from __future__ import annotations
import argparse
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import configs
from apex_nadir_convolution import ApexNadirConvolution
from central import CentralModel
from data import authenticate_huggingface, DomainLabelledStream
from experts import ExpertPool
from gating import GateModel, TripleKSelector, MaskingSchedule
from inference import InferenceEngine
from memory import RoutingMemory, SessionTracker
from meta import MAMLOptimiser
from bios import BIOS
from conversational_memory import ConversationalMemory
from code_executor import CodeExecutor
from self_correction import SelfCorrection
from proactive import ProactiveEngine
from k_velocity import KVelocity
from voice import VoicePipeline, VoiceConfig
from autonomous import AutonomousExecutor
from emotional import EmotionalCalibration
from predictive import PredictiveAssistant
from world_model import WorldModel
from vision_module import VisionSystem


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
    # BRIAN capability stack
    bios: Optional[BIOS] = None
    conv_memory: Optional[ConversationalMemory] = None
    code_executor: Optional[CodeExecutor] = None
    self_correction: Optional[SelfCorrection] = None
    proactive: Optional[ProactiveEngine] = None
    k_velocity: Optional[KVelocity] = None
    voice: Optional[VoicePipeline] = None
    autonomous: Optional[AutonomousExecutor] = None
    emotional: Optional[EmotionalCalibration] = None
    predictive: Optional[PredictiveAssistant] = None
    world_model: Optional[WorldModel] = None
    vision: Optional[VisionSystem] = None
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
    authenticate_huggingface()
    convolution = ApexNadirConvolution(
        calibration_path=configs.CALIBRATION_PATH,
        latency_store_path=configs.LATENCY_STORE_PATH,
    )
    convolution.load()
    routing_memory = RoutingMemory()
    routing_memory.load(configs.ROUTING_MEMORY_PATH)
    session_tracker = SessionTracker()
    gate = GateModel()
    gate.load()
    central = CentralModel()
    expert_pool = ExpertPool(convolution=convolution, session_tracker=session_tracker)
    triple_k = TripleKSelector(convolution=convolution)
    masking_schedule = MaskingSchedule()
    maml = MAMLOptimiser(gate_model=gate.model)
    maml.load()

    # --- Boot BRIAN capability stack ---
    bios = BIOS() if configs.BIOS_ENABLED else None

    conv_memory = ConversationalMemory(
        store_dir=configs.CONV_MEMORY_DIR,
        max_conversations=configs.CONV_MEMORY_MAX_CONVERSATIONS,
    )

    code_executor = CodeExecutor(timeout=configs.CODE_EXEC_TIMEOUT)

    self_correction = SelfCorrection(
        confidence_floor=configs.SELF_CORRECTION_CONFIDENCE_FLOOR,
        entropy_ceiling=configs.SELF_CORRECTION_ENTROPY_CEILING,
        max_retries=configs.SELF_CORRECTION_MAX_RETRIES,
    ) if configs.SELF_CORRECTION_ENABLED else None

    proactive = ProactiveEngine(
        check_interval=configs.PROACTIVE_CHECK_INTERVAL,
        cpu_threshold=configs.PROACTIVE_CPU_THRESHOLD,
        ram_threshold_mb=configs.PROACTIVE_RAM_THRESHOLD_MB,
        thermal_threshold=configs.PROACTIVE_THERMAL_THRESHOLD,
        disk_threshold_gb=configs.PROACTIVE_DISK_THRESHOLD_GB,
        battery_threshold=configs.PROACTIVE_BATTERY_THRESHOLD,
    ) if configs.PROACTIVE_ENABLED else None

    k_velocity_engine = KVelocity(
        decay_rate=configs.K_VELOCITY_DECAY,
        learning_rate=configs.K_VELOCITY_LR,
        history_file=configs.K_VELOCITY_HISTORY,
    )

    voice = None
    if configs.VOICE_ENABLED:
        voice = VoicePipeline(VoiceConfig(
            whisper_model=configs.WHISPER_MODEL,
            tts_voice=configs.TTS_VOICE,
            say_voice=configs.SAY_VOICE,
            hindi_voice=configs.HINDI_VOICE,
            sample_rate=configs.VOICE_SAMPLE_RATE,
            vad_threshold=configs.VOICE_VAD_THRESHOLD,
        ))

    emotional = EmotionalCalibration() if configs.EMOTIONAL_ENABLED else None

    predictive = PredictiveAssistant(
        history_file=configs.PREDICTIVE_HISTORY,
    ) if configs.PREDICTIVE_ENABLED else None

    world_model_inst = WorldModel(
        store_path=configs.WORLD_MODEL_PATH,
        max_entities=configs.WORLD_MODEL_MAX_ENTITIES,
    ) if configs.WORLD_MODEL_ENABLED else None

    vision = VisionSystem(output_dir=configs.VISION_OUTPUT_DIR) if configs.VISION_ENABLED else None

    autonomous = AutonomousExecutor()

    inference_engine = InferenceEngine(
        gate=gate,
        expert_pool=expert_pool,
        central=central,
        convolution=convolution,
        routing_memory=routing_memory,
        session_tracker=session_tracker,
        triple_k=triple_k,
        masking_schedule=masking_schedule,
        bios=bios,
        conv_memory=conv_memory,
        code_executor=code_executor,
        self_correction=self_correction,
        k_velocity=k_velocity_engine,
        emotional=emotional,
        predictive=predictive,
        world_model=world_model_inst,
        vision=vision,
    )

    # Wire autonomous handlers
    def _inference_handler(description, params, context):
        prompt = f"{context}\n\n{description}" if context else description
        result = inference_engine.run(prompt, send_to_user=True)
        return {"output": result.output_text}

    def _bios_handler(description, params, context):
        if bios:
            cmd_result = bios.execute(params.get("raw_text", description))
            return {"output": cmd_result.stdout or cmd_result.stderr}
        return {"error": "BIOS not available."}

    def _code_handler(description, params, context):
        lang = params.get("language", "python")
        result = code_executor.execute(description, lang)
        return {"output": result.stdout, "error": result.stderr if result.returncode != 0 else ""}

    def _search_handler(description, params, context):
        from tools import query_search_with_fallback
        return {"output": query_search_with_fallback(params.get("query", description))}

    autonomous.register_handler("inference", _inference_handler)
    autonomous.register_handler("bios", _bios_handler)
    autonomous.register_handler("code", _code_handler)
    autonomous.register_handler("search", _search_handler)
    autonomous.register_handler("tool", _search_handler)
    autonomous.register_handler("file", _bios_handler)

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
        bios=bios,
        conv_memory=conv_memory,
        code_executor=code_executor,
        self_correction=self_correction,
        proactive=proactive,
        k_velocity=k_velocity_engine,
        voice=voice,
        autonomous=autonomous,
        emotional=emotional,
        predictive=predictive,
        world_model=world_model_inst,
        vision=vision,
    )
def run_universal_buffet(components: SystemComponents):
    stream = DomainLabelledStream(dataset_ids=configs.DATASET_IDS)
    for expert_id in range(configs.EXPERT_POOL_SIZE):
        calibration_data: Dict[str, Any] = {}
        for domain_batch in stream.iter_calibration_batches(expert_id):
            calibration_data.update(domain_batch)
        components.convolution.fit_curves_from_calibration(expert_id, calibration_data)
    components.convolution.save()
def session_reset(components: SystemComponents, dead_state: DeadTimeState):
    components.session_tracker.reset()
    components.routing_memory.save(configs.ROUTING_MEMORY_PATH)
    components.maml.save()
    components.convolution.save_latency_store()
    components.maml.log_k_velocity_all_domains()
    dead_state.pending_timeline_a_inputs.clear()
    dead_state.last_outer_loop_token = 0
    # Persist BRIAN state
    if components.k_velocity:
        components.k_velocity.save()
    if components.predictive:
        components.predictive.save()
    if components.world_model:
        components.world_model.save()
    if components.proactive:
        components.proactive.stop()
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

    # Start proactive monitoring
    proactive_task = None
    if components.proactive:
        proactive_task = asyncio.create_task(components.proactive.start())

    turns = 0
    try:
        while True:
            if max_turns and turns >= max_turns:
                break

            # Check for proactive alerts
            if components.proactive:
                alerts = components.proactive.get_pending_alerts()
                for alert in alerts:
                    print(f"\n[BRIAN] {alert.message}")

            try:
                loop = asyncio.get_event_loop()
                user_in = await loop.run_in_executor(None, lambda: input("Sturnus> ").strip())
            except (EOFError, KeyboardInterrupt):
                break
            if not user_in:
                continue

            # Check for autonomous multi-step tasks
            if components.autonomous and components.autonomous.is_multi_step(user_in):
                task = components.autonomous.plan(user_in)
                print(f"[BRIAN] Breaking into {len(task.steps)} steps...")
                for step in task.steps:
                    print(f"  {step.index + 1}. {step.description} [{step.action_type}]")
                task = components.autonomous.execute(task)
                print(task.final_output)
            else:
                result = process_input(user_in, components, dead_state)
                print(result["output_text"])
            turns += 1
    finally:
        if proactive_task:
            proactive_task.cancel()
        orchestrator_task.cancel()
        session_reset(components, dead_state)
def run_cli():
    parser = argparse.ArgumentParser(description="Sturnus inference engine.")
    parser.add_argument("--prompt", type=str, help="Run a single prompt and exit.")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat loop.")
    parser.add_argument("--max-turns", type=int, default=0, help="Stop after N turns.")
    parser.add_argument("--json", action="store_true", help="Print full result dict.")
    parser.add_argument("--buffet", action="store_true", help="Run Universal Buffet calibration.")
    parser.add_argument("--deployment", action="store_true", help="Run in deployment mode.")
    args = parser.parse_args()
    if args.deployment:
        configs.DEPLOYMENT = True
    components = boot_system()
    dead_state = DeadTimeState()
    if args.buffet:
        run_universal_buffet(components)
        return
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
    result = process_input("Test input for Sturnus.", components, dead_state)
    run_pending_timeline_a_shadows(components, dead_state)
    print(result["output_text"])
    session_reset(components, dead_state)
if __name__ == "__main__":
    run_cli()
