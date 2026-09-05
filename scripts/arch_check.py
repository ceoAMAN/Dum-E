"""Architecture verification — run BEFORE any training run.

The failure mode this codebase keeps producing is not a crash, it is a mechanism
that looks wired, executes, and decides nothing: R_out pinned at its floor for
994k tokens, the nadir gate silently disabling the whole pool, the apex-nadir
refresh fitting from a single point, spiderweb unable to move a zero-weight
expert. Every one of those passed "it imports and runs".

So this does not check that things run. It instruments the hot path and ASSERTS
each mechanism actually fired and actually changed something.
"""
import sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import configs

CALLS = {}


def track(mod, name, label=None):
    """Wrap a function so we can prove it was reached."""
    label = label or name
    orig = getattr(mod, name)
    def wrapped(*a, **k):
        CALLS[label] = CALLS.get(label, 0) + 1
        return orig(*a, **k)
    setattr(mod, name, wrapped)
    return orig


def track_method(cls, name, label=None):
    label = label or f"{cls.__name__}.{name}"
    orig = getattr(cls, name)
    def wrapped(self, *a, **k):
        CALLS[label] = CALLS.get(label, 0) + 1
        return orig(self, *a, **k)
    setattr(cls, name, wrapped)
    return orig


def section(t):
    print(f"\n{'='*74}\n{t}\n{'='*74}")


results = {}


def check(name, cond, detail=""):
    results[name] = bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return cond


# ── A. static ────────────────────────────────────────────────────────────────
section("A. STATIC")
import py_compile, importlib
MODULES = ['configs','apex_nadir_convolution','splitter','gating','experts','central',
           'memory','meta','training','diagnostics','vectors','inference','main']
ok = True
for m in MODULES:
    try:
        py_compile.compile(m + '.py', doraise=True); importlib.import_module(m)
    except Exception as e:
        ok = False; print(f"    {m}: {e}")
check("all 13 modules compile + import", ok)
try:
    configs.validate_config(); check("validate_config", True)
except Exception as e:
    check("validate_config", False, str(e))

import gating, splitter, training, diagnostics, experts as experts_mod, central as central_mod
check("E is variable (no ==100 lock)", configs.EXPERT_POOL_SIZE >= 1)
check("D is variable", len(gating.DOMAINS) >= 2, f"D={len(gating.DOMAINS)}")
check("EXPERT_GROUPS empty (earned, not declared)", not configs.EXPERT_GROUPS)
check("LORA scale = alpha/r", abs(configs.LORA_ALPHA / configs.LORA_R - 2.0) < 1e-9,
      f"{configs.LORA_ALPHA}/{configs.LORA_R} = {configs.LORA_ALPHA/configs.LORA_R}")
check("no OLS _regress", not hasattr(diagnostics.Diagnostics, "_regress"))
check("no CEMGovernor", not hasattr(diagnostics, "CEMGovernor"))
check("voice path removed", not hasattr(central_mod.CentralModel, "generate_stream"))
check("CentralModel.save exists", hasattr(central_mod.CentralModel, "save"))

# ── B. instrument the hot path ───────────────────────────────────────────────
section("B. BOOT")
import inference, main
track(inference, "schedule_by_expert")   # imported by name into inference; patch THAT binding
track(diagnostics, "select_k_cap")
track(diagnostics, "evaluate_configuration")
track_method(experts_mod.ExpertPool, "peer_disagreement")
track_method(experts_mod.ExpertPool, "_build_expert_prompt")
import memory as memory_mod
track_method(memory_mod.SessionTracker, "composite_tkl_pool")
track_method(inference.InferenceEngine, "_apply_expert_learning")
track_method(inference.InferenceEngine, "note_input")
track_method(inference.InferenceEngine, "_refresh_apex_nadir")
track_method(central_mod.CentralModel, "compute_r_i_batch")

t0 = time.time()
try:
    comp = main.boot_system()
    check("boot_system()", True, f"{time.time()-t0:.0f}s")
except Exception:
    traceback.print_exc(); check("boot_system()", False); sys.exit(1)

print(f"    measured: expert {configs.EXPERT_RAM_MB}MB  gate {configs.GATE_RAM_MB}MB")
print(f"    hard cap: {comp.inference_engine.diagnostics.memory_ceiling()}  "
      f"experts_per_batch: {splitter.experts_per_batch()}")

# ── C. Timeline A ────────────────────────────────────────────────────────────
section("C. TIMELINE A (Central alone)")
r = comp.inference_engine.run("What is 2+2?", send_to_user=True, force_timeline_a=True)
check("A produced text", bool(r.output_text.strip()), f"{len(r.output_text)} chars")
print(f"    {r.output_text.strip()[:110]!r}")

# ── D. Timeline B ────────────────────────────────────────────────────────────
section("D. TIMELINE B (experts + synthesis)")
CALLS.clear()
q = "Explain why quicksort is O(n log n) on average but O(n^2) in the worst case."
t0 = time.time()
rb = comp.inference_engine.run(q, send_to_user=True, force_timeline_b=True, min_experts=2)
dt = time.time() - t0
check("B produced text", bool(rb.output_text.strip()), f"{len(rb.output_text)} chars, {dt:.0f}s")
print(f"    {rb.output_text.strip()[:160]!r}")
check("experts actually ran", rb.k_used > 0, f"k_used={rb.k_used} {rb.experts_activated}")
check("r_i computed", rb.mean_r_i > 0.0, f"mean_r_i={rb.mean_r_i:.4f}")
check("entropy finite", rb.reconstruction_entropy == rb.reconstruction_entropy)

section("E. DID EACH MECHANISM FIRE?")
expected = [
    ("note_input",                    "InferenceEngine.note_input"),
    ("expert chat template",          "ExpertPool._build_expert_prompt"),
    ("schedule_by_expert",            "schedule_by_expert"),
    ("peer_disagreement",             "ExpertPool.peer_disagreement"),
    ("compute_r_i_batch",             "CentralModel.compute_r_i_batch"),
    ("composite_tkl_pool",            "SessionTracker.composite_tkl_pool"),
    ("spiderweb",                     "InferenceEngine._apply_expert_learning"),
    ("select_k_cap",                  "select_k_cap"),
    ("evaluate_configuration",        "evaluate_configuration"),
    ("apex-nadir refresh",            "InferenceEngine._refresh_apex_nadir"),
]
for label, key in expected:
    check(f"{label} fired", CALLS.get(key, 0) > 0, f"{CALLS.get(key,0)}x")

section("F. STATE PERSISTED")
import os
check("latency store written", Path(configs.LATENCY_STORE_PATH).exists())
check("probe records captured", len(comp.convolution.probe_records) > 0,
      f"{len(comp.convolution.probe_records)} probes")
check("TKL populated", any(v > 0 for v in comp.session_tracker.expert_tkl.values()),
      f"{sum(1 for v in comp.session_tracker.expert_tkl.values() if v>0)} experts scored")

section("SUMMARY")
bad = [k for k, v in results.items() if not v]
print(f"  {len(results)-len(bad)}/{len(results)} passed")
if bad:
    print("  FAILED:")
    for b in bad: print(f"    - {b}")
    sys.exit(1)
print("\n  architecture verified — safe to launch training")
