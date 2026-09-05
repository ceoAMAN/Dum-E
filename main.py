from __future__ import annotations
import argparse
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List
import configs
from apex_nadir_convolution import ApexNadirConvolution
from central import CentralModel
from experts import ExpertPool
from gating import GateModel, TripleKSelector, MaskingSchedule
from inference import InferenceEngine
from memory import RoutingMemory, SessionTracker
from meta import MAMLOptimiser
from splitter import measure_expert_ram_mb


@dataclass
class SystemComponents:
    gate: GateModel
    expert_pool: ExpertPool
    central: CentralModel
    convolution: ApexNadirConvolution
    routing_memory: RoutingMemory
    session_tracker: SessionTracker
    maml: MAMLOptimiser
    inference_engine: InferenceEngine
    triple_k: TripleKSelector
    masking_schedule: MaskingSchedule
    r_out_mean_seed: float
@dataclass
class DeadTimeState:
    active: bool = False
    pending_timeline_a_inputs: List[str] = field(default_factory=list)
    last_outer_loop_token: int = 0
    total_tokens_processed: int = 0
    inference_active: bool = False
    last_domain: str = "general"
    last_k_used: int = 0
    last_reconstruction_entropy: float = 0.0
def boot_system() -> SystemComponents:
    configs.validate_config()
    convolution = ApexNadirConvolution(
        calibration_path=configs.CALIBRATION_PATH,
        latency_store_path=configs.LATENCY_STORE_PATH,
    )
    convolution.load()
    routing_memory = RoutingMemory()
    routing_memory.load(configs.ROUTING_MEMORY_PATH)
    session_tracker = SessionTracker()
    session_tracker.load(configs.SESSION_TRACKER_PATH)
    gate = GateModel()
    gate.load()
    central = CentralModel()
    measure_expert_ram_mb()   # measure real per-expert RAM so X/Y geometry is grounded in fact
    expert_pool = ExpertPool(convolution=convolution, session_tracker=session_tracker)
    triple_k = TripleKSelector(convolution=convolution)
    masking_schedule = MaskingSchedule()
    maml = MAMLOptimiser(gate_model=gate.model)
    maml.load()

    inference_engine = InferenceEngine(
        gate=gate,
        expert_pool=expert_pool,
        central=central,
        convolution=convolution,
        routing_memory=routing_memory,
        session_tracker=session_tracker,
        triple_k=triple_k,
        masking_schedule=masking_schedule,
        maml=maml,
    )

    # If the previous process died mid-batch, its in-flight marker is still on
    # disk. Claim it now, before any work: a Metal OOM aborts uncatchably, so
    # this is the only moment the crash can be turned into evidence.
    inference_engine.diagnostics.claim_crashed_run()

    # FEED THE MEMORY GOVERNOR. set_memory_baseline / set_usable / observe_memory
    # / can_fit_expert were all unreachable, so _mem_usable_mb stayed 0.0,
    # peak_util() returned 0.0 by its own guard, and can_fit_expert() answered
    # True unconditionally — a ceiling computed against nothing.
    #
    # Honest about what this measures: CentralModel() only sets model = None, so
    # the 4B is NOT resident here and this base is gate-plus-runtime, not the
    # full footprint. `_mem_base_mb` has no reader anywhere in the system; the
    # value that does work is the usable ceiling, and inference refreshes that
    # from a live reading every batch. The seed matters only for the window
    # before the first observed peak.
    from splitter import get_active_memory_mb, get_available_ram_mb
    base_mb = get_active_memory_mb()
    usable_mb = base_mb + get_available_ram_mb()
    inference_engine.diagnostics.set_memory_baseline(base_mb, usable_mb)
    print(f"[boot] memory governor: resident {base_mb:.0f} MB (gate only — central "
          f"loads lazily), usable ceiling {usable_mb:.0f} MB")

    r_out_mean_seed = configs.MAX_SEQ_LEN / configs.K_DEFAULT
    return SystemComponents(
        gate=gate,
        expert_pool=expert_pool,
        central=central,
        convolution=convolution,
        routing_memory=routing_memory,
        session_tracker=session_tracker,
        maml=maml,
        inference_engine=inference_engine,
        triple_k=triple_k,
        masking_schedule=masking_schedule,
        r_out_mean_seed=r_out_mean_seed,
    )
def pretrain_central(components: SystemComponents, samples, total_token_budget: int,
                     print_every: int = 50) -> Dict[str, Any]:
    """PHASE 1 — Central trains alone on half the total token budget.

    Of Ť total tokens, Ť/2 go to Central on its own; the remaining Ť run the
    joint phase where the experts work and Central synthesises. Central is
    trained FIRST and then stops taking gradients, which is what makes r_i
    comparable across a run: r_i is measured against Central's loss, so a Central
    that keeps moving is a yardstick that keeps moving, and expert scores from
    early and late batches stop meaning the same thing.

    `samples` is any iterable of (text, target_ids) — data-source agnostic, like
    run_calibration. Pair Dum-E with any corpus; the engine never names one.
    Stops when the budget is spent or the samples run out, then persists Central
    (which is the step that never existed: load() guards on a checkpoint that was
    never written, so every boot silently reloaded the stock model)."""
    import mlx.optimizers as optim
    from training import apply_central_pretrain

    budget = max(1, int(total_token_budget) // 2)
    optimizer = optim.Adam(learning_rate=configs.LEARNING_RATE)
    spent, steps, losses = 0, 0, []
    for text, target_ids in samples:
        if spent >= budget:
            break
        if not target_ids:
            continue
        out = apply_central_pretrain(components.central, optimizer, text, list(target_ids))
        spent += len(target_ids)
        steps += 1
        if out["ce"] > 0.0:
            losses.append(out["ce"])
        if print_every and steps % print_every == 0:
            recent = sum(losses[-print_every:]) / max(1, len(losses[-print_every:]))
            print(f"[central] step {steps} | {spent}/{budget} tokens | ce {recent:.4f}")
    components.central.save()
    first = sum(losses[:10]) / max(1, len(losses[:10])) if losses else 0.0
    last = sum(losses[-10:]) / max(1, len(losses[-10:])) if losses else 0.0
    print(f"[central] phase 1 done: {steps} steps, {spent} tokens, ce {first:.4f} -> {last:.4f}, saved")
    return {"steps": steps, "tokens": spent, "ce_first": first, "ce_last": last}
def run_calibration(components: SystemComponents, calibration_batches):
    """Fit each expert's apex/nadir/latency curves from caller-supplied calibration
    data — an iterable of (expert_id, dict) with token_counts / quality_scores /
    gradient_coherence / wall_times keys. Data-source agnostic: pair Dum-E with any
    corpus and feed measurements here (or let a training loop fit curves live)."""
    for expert_id, calibration_data in calibration_batches:
        components.convolution.fit_curves_from_calibration(expert_id, calibration_data)
    components.convolution.save()
def session_reset(components: SystemComponents, dead_state: DeadTimeState):
    # Persist BEFORE reset. reset() now only clears the session counters, but
    # saving first makes the ordering explicit rather than incidental.
    try:
        components.session_tracker.save(configs.SESSION_TRACKER_PATH)
    except Exception as e:
        print(f"[warn] session tracker save failed: {e}")
    components.session_tracker.reset()
    components.routing_memory.save(configs.ROUTING_MEMORY_PATH)
    # Persist the gate. save_route_head existed but nothing called it, so even
    # once the gate started learning every update would have been discarded at
    # session end — the head would reload from its old checkpoint next boot.
    try:
        components.gate.save_route_head()
    except Exception as e:
        print(f"[warn] route_head save failed: {e}")
    components.maml.save()
    components.convolution.save_latency_store()
    # save() as well as save_latency_store(): the latency store holds only the
    # per-expert cost coefficients. ALLOC(T), the goldilocks points it was fitted
    # from, and central_min_cap live in the calibration file, and nothing on the
    # live path was writing it — so every curve the session fitted was lost.
    components.convolution.save()
    components.maml.log_k_velocity_all_domains()
    dead_state.pending_timeline_a_inputs.clear()
    dead_state.last_outer_loop_token = 0
async def dead_time_orchestrator(components: SystemComponents, dead_state: DeadTimeState):
    while True:
        await asyncio.sleep(0.1)
        if dead_state.inference_active:
            continue
        if configs.DEPLOYMENT:
            # Under deployment, we still run the pending Timeline A shadow inputs
            # to verify that the dead-time B cycle runs asynchronously without blocking,
            # but we skip MAML parameter updates, routing syncs, stuck expert reassignment, and other training/optimization logic.
            if dead_state.pending_timeline_a_inputs:
                pending = dead_state.pending_timeline_a_inputs.copy()
                dead_state.pending_timeline_a_inputs.clear()
                for text in pending:
                    components.inference_engine.run(
                        text,
                        send_to_user=False,
                        force_timeline_b=True,
                        min_experts=max(1, components.routing_memory.get_domain_mean_k())
                    )
            continue
            
        if components.maml.should_run_outer_loop(dead_state.total_tokens_processed, dead_state.last_outer_loop_token):
            components.maml.run_outer_step_from_metrics(
                domain=dead_state.last_domain,
                k_value=dead_state.last_k_used,
                reconstruction_entropy=dead_state.last_reconstruction_entropy,
                timeline_a_rate=components.session_tracker.get_timeline_a_rate(),
                cluster_count=len(components.routing_memory.clusters),
            )
            dead_state.last_outer_loop_token = dead_state.total_tokens_processed
            # Persist state only after meaningful work (MAML step), not every 0.1s
            components.routing_memory.sync(configs.ROUTING_MEMORY_PATH)
            components.convolution.save_latency_store()
        for expert_id in range(configs.EXPERT_POOL_SIZE):
            domain = components.session_tracker.get_dominant_domain(expert_id)
            if components.expert_pool.check_stuck_expert(expert_id, domain, dead_state.total_tokens_processed, components.convolution):
                new_domain = components.session_tracker.find_migration_target(expert_id, components.convolution)
                components.expert_pool.reassign_expert(expert_id, new_domain)
        if dead_state.pending_timeline_a_inputs:
            pending = dead_state.pending_timeline_a_inputs.copy()
            dead_state.pending_timeline_a_inputs.clear()
            for text in pending:
                components.inference_engine.run(
                    text,
                    send_to_user=False,
                    force_timeline_b=True,
                    min_experts=max(1, components.routing_memory.get_domain_mean_k())
                )
def process_input(text: str, components: SystemComponents, dead_state: DeadTimeState) -> Dict[str, Any]:
    dead_state.inference_active = True
    try:
        result = components.inference_engine.run(text, send_to_user=True)
        if result.timeline == "A":
            dead_state.pending_timeline_a_inputs.append(text)
        token_count = result.token_count
        dead_state.total_tokens_processed += token_count
        dead_state.last_domain = result.domain
        dead_state.last_k_used = result.k_used
        dead_state.last_reconstruction_entropy = result.reconstruction_entropy
        components.maml.record_k(result.domain, result.k_used, dead_state.total_tokens_processed)
        components.session_tracker.log_warmup(dead_state.total_tokens_processed)
    finally:
        dead_state.inference_active = False
    return {
        "timeline": result.timeline,
        "output_text": result.output_text,
        "k_used": result.k_used,
        "experts_activated": result.experts_activated,
    }
def run_pending_timeline_a_shadows(components: SystemComponents, dead_state: DeadTimeState):
    if not dead_state.pending_timeline_a_inputs:
        return
    pending = dead_state.pending_timeline_a_inputs.copy()
    dead_state.pending_timeline_a_inputs.clear()
    for text in pending:
        components.inference_engine.run(
            text,
            send_to_user=False,
            force_timeline_b=True,
            min_experts=max(1, components.routing_memory.get_domain_mean_k())
        )
async def run_interactive(components: SystemComponents, dead_state: DeadTimeState, max_turns: int = 0):
    orchestrator_task = asyncio.create_task(dead_time_orchestrator(components, dead_state))

    turns = 0
    try:
        while True:
            if max_turns and turns >= max_turns:
                break

            try:
                loop = asyncio.get_event_loop()
                user_in = await loop.run_in_executor(None, lambda: input("Dum-E> ").strip())
            except (EOFError, KeyboardInterrupt):
                break
            if not user_in:
                continue

            result = process_input(user_in, components, dead_state)
            print(result["output_text"])
            turns += 1
    finally:
        orchestrator_task.cancel()
        session_reset(components, dead_state)
def run_cli():
    parser = argparse.ArgumentParser(description="Dum-E core engine.")
    parser.add_argument("--prompt", type=str, help="Run a single prompt and exit.")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat loop.")
    parser.add_argument("--max-turns", type=int, default=0, help="Stop after N turns.")
    parser.add_argument("--json", action="store_true", help="Print full result dict.")
    parser.add_argument("--deployment", action="store_true", help="Run in deployment mode.")
    args = parser.parse_args()
    if args.deployment:
        configs.DEPLOYMENT = True
    components = boot_system()
    dead_state = DeadTimeState()
    if args.prompt:
        result = process_input(args.prompt, components, dead_state)
        run_pending_timeline_a_shadows(components, dead_state)
        if args.json:
            print(result)
        else:
            print(result["output_text"])
        session_reset(components, dead_state)
        return
    if args.interactive:
        asyncio.run(run_interactive(components, dead_state, max_turns=args.max_turns))
        return
    result = process_input("Test input for Dum-E.", components, dead_state)
    run_pending_timeline_a_shadows(components, dead_state)
    print(result["output_text"])
    session_reset(components, dead_state)
if __name__ == "__main__":
    run_cli()
