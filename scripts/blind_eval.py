"""Blind quality eval — does the MoE pipeline actually change the answer?

Reading the code reveals two things that decide what this eval *can* show:

  1. Central is FROZEN. Training updates the gate (routing) and the experts
     (hidden-state alignment), but never Central. The user-facing reply is
     always produced by base Central.
  2. Since the audit A.2.4 fix, expert_forward(..., generate_text=True) produces
     a REAL generated answer (not the old argmax echo of the input), and the
     Timeline-B reply now injects that expert text into Central's prompt. This
     harness calls experts with generate_text=True so column B measures the same
     expert-conditioned reply the deployed Timeline-B path now produces.

The meaningful question this answers: does conditioning on what the experts
produce make the answer better or worse? That quantifies whether the (now wired)
MoE machinery is a net quality lever or injected noise — the A.5 next step.

Configs compared per query:
  A. central_alone        — Central.generate(question)             [deployed reply]
  B. central_plus_experts — Central.generate(question + expert text)[does it help?]
  C. baseline (optional)  — an external model via --baseline-model  [RAM-gated]

Outputs logs/blind_eval.jsonl (one record per query, all replies + the raw
expert text so a human/GPT judge can see what the experts contributed) and a
ready-to-paste GPT-judge prompt in logs/blind_eval_judge_prompt.txt.

  python scripts/blind_eval.py --n 8 --max-tokens 128
  python scripts/blind_eval.py --n 20 --baseline-model mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx

import configs

QUERIES = [
    ("code", "Write a Python function that checks whether a string is a palindrome."),
    ("code", "How do I remove duplicate entries from a list in Python while keeping order?"),
    ("code", "Explain what a Python decorator is and give a small example."),
    ("code", "Write a function to compute the greatest common divisor of two integers."),
    ("code", "How do I read a CSV file and sum one column in Python?"),
    ("reasoning", "A shirt costs $40 after a 20% discount. What was the original price?"),
    ("reasoning", "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?"),
    ("reasoning", "What is the derivative of f(x) = 3x^3 - 5x + 2?"),
    ("reasoning", "How many ways can you arrange the letters in the word 'apple'?"),
    ("reasoning", "Two trains leave 300 km apart heading toward each other at 50 and 100 km/h. When do they meet?"),
    ("knowledge", "Why is the sky blue?"),
    ("knowledge", "What were the main causes of the Industrial Revolution?"),
    ("knowledge", "Explain the difference between mitosis and meiosis."),
    ("knowledge", "Who was Ada Lovelace and why is she significant?"),
    ("knowledge", "What is the greenhouse effect?"),
    ("general", "Give me three tips for staying productive while working from home."),
    ("general", "Suggest a simple dinner I can make with chicken, rice, and broccoli."),
    ("general", "How can I start meditating as a complete beginner?"),
    ("general", "Recommend three classic novels for someone new to reading fiction."),
    ("general", "What's a good way to apologize to a friend after a small argument?"),
]


def truncate(s: str, n: int = 400) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def run_pipeline_experts(gate, expert_pool, central, triple_k, masking, session_tracker, question, k):
    """Run the Timeline-B MoE machinery and return (expert_texts, n_experts).
    Faithful to scripts/finetune.py's forward, but inference-only."""
    token_ids = gate.tokenizer.encode(question)[: configs.MAX_SEQ_LEN]
    if len(token_ids) < configs.FRAGMENT_MIN:
        token_ids = (token_ids * (configs.FRAGMENT_MIN // max(len(token_ids), 1) + 1))[: configs.FRAGMENT_MIN]
    tokens = mx.array(token_ids)
    n_tokens = len(token_ids)
    gate_out = gate.forward(tokens)
    selected = triple_k.select_experts(gate_out, session_tracker, masking, 0)[:k]
    ids_to_load = [s.expert_id for s in selected if s.expert_id not in expert_pool.loaded_experts]
    try:
        if ids_to_load:
            expert_pool.load_experts(ids_to_load)
    except RuntimeError:
        return [], 0
    selected = [s for s in selected if s.expert_id in expert_pool.loaded_experts]
    if not selected:
        return [], 0
    frag = max(configs.FRAGMENT_MIN, n_tokens // max(len(selected), 1))
    texts = []
    for i, sel in enumerate(selected):
        fs, fe = i * frag, min(i * frag + frag, n_tokens)
        if fs >= n_tokens:
            break
        ft = tokens[fs:fe]
        if ft.shape[0] < configs.FRAGMENT_MIN:
            continue
        eo = expert_pool.expert_forward(sel.expert_id, ft, generate_text=True)
        texts.append(eo.output_text)
    return texts, len(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of queries (max 20)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--baseline-model", default="", help="external HF/MLX model id for a 3rd column")
    args = ap.parse_args()

    queries = QUERIES[: max(1, min(args.n, len(QUERIES)))]
    print(f"[blind] {len(queries)} queries | max_tokens={args.max_tokens}")

    from gating import GateModel, TripleKSelector, MaskingSchedule
    from memory import SessionTracker
    from apex_nadir_convolution import ApexNadirConvolution
    from experts import ExpertPool
    from central import CentralModel
    from splitter import get_available_ram_mb

    convolution = ApexNadirConvolution(configs.CALIBRATION_PATH, configs.LATENCY_STORE_PATH)
    convolution.load()
    gate = GateModel(); gate.load()
    triple_k = TripleKSelector(convolution=convolution)
    masking = MaskingSchedule()
    session_tracker = SessionTracker()
    boot_ram = get_available_ram_mb()
    hw_max = min(max(1, int(max(0, boot_ram - 4000) / configs.EXPERT_RAM_MB)), configs.K_MAX)
    expert_pool = ExpertPool(convolution=convolution, session_tracker=session_tracker, max_loaded=max(2, hw_max))
    central = CentralModel(); central.load()
    print(f"[blind] booted (expert_cap={hw_max})")

    # Optional external baseline (e.g. Mixtral). Skipped gracefully on OOM/RAM.
    baseline_model = baseline_tok = None
    if args.baseline_model:
        try:
            from mlx_lm import load as mlx_load
            print(f"[blind] loading baseline {args.baseline_model} ...")
            baseline_model, baseline_tok = mlx_load(args.baseline_model)
            print("[blind] baseline loaded")
        except Exception as e:
            print(f"[blind] baseline unavailable ({type(e).__name__}: {e}) — skipping column C. "
                  f"(Mixtral-8x7B-4bit needs ~26GB; infeasible on 16GB.)")
            baseline_model = None

    from mlx_lm import generate as mlx_generate

    records = []
    identical = 0
    for idx, (domain, q) in enumerate(queries, 1):
        t0 = time.time()
        # A. deployed reply
        central_alone = central.generate(q, max_tokens=args.max_tokens)
        # MoE machinery: what do the experts produce?
        expert_texts, n_exp = run_pipeline_experts(
            gate, expert_pool, central, triple_k, masking, session_tracker, q, args.k)
        # B. inject expert text into the prompt
        if expert_texts:
            joined = "\n".join(t for t in expert_texts if t and t.strip())[: configs.MAX_SEQ_LEN * 4]
            aug = f"{q}\n\n[Reference notes]\n{joined}"
        else:
            aug = q
        central_plus = central.generate(aug, max_tokens=args.max_tokens)
        # C. external baseline
        baseline_reply = ""
        if baseline_model is not None:
            try:
                baseline_reply = mlx_generate(baseline_model, baseline_tok,
                                               prompt=central.format_prompt(q), max_tokens=args.max_tokens)
            except Exception as e:
                baseline_reply = f"(baseline error: {e})"

        same = central_alone.strip() == central_plus.strip()
        identical += int(same)
        rec = {
            "id": idx, "domain": domain, "question": q,
            "n_experts": n_exp,
            "expert_text_sample": truncate(expert_texts[0]) if expert_texts else "",
            "central_alone": central_alone.strip(),
            "central_plus_experts": central_plus.strip(),
            "baseline": baseline_reply.strip(),
            "A_equals_B": same,
            "secs": round(time.time() - t0, 1),
        }
        records.append(rec)
        print(f"  [{idx}/{len(queries)}] {domain:9s} | experts={n_exp} | A==B:{same} | {rec['secs']}s")
        print(f"      expert text: {rec['expert_text_sample'][:120]}")

    Path("logs").mkdir(exist_ok=True)
    with open("logs/blind_eval.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # GPT-judge scaffold (only meaningful where A != B or a baseline exists)
    judge = (
        "You are a strict, impartial judge. For each question you are given anonymized "
        "candidate answers labelled by letter. Pick the single best answer on correctness, "
        "completeness, and clarity. Respond with the winning letter and one sentence of "
        "justification. Do not reward verbosity.\n\n"
    )
    for r in records:
        judge += f"### Q{r['id']} ({r['domain']}): {r['question']}\n"
        judge += f"[A] {truncate(r['central_alone'], 600)}\n"
        judge += f"[B] {truncate(r['central_plus_experts'], 600)}\n"
        if r["baseline"]:
            judge += f"[C] {truncate(r['baseline'], 600)}\n"
        judge += "\n"
    Path("logs/blind_eval_judge_prompt.txt").write_text(judge)

    print("\n=== Blind eval summary ===")
    print(f"  queries                 : {len(records)}")
    print(f"  A==B (expert text made NO difference) : {identical}/{len(records)} = {identical/len(records):.0%}")
    print(f"  → If high, the MoE experts do not change the user-facing answer in the")
    print(f"    current wiring (training-only apparatus). See logs/blind_eval.jsonl.")
    print(f"  wrote logs/blind_eval.jsonl and logs/blind_eval_judge_prompt.txt")


if __name__ == "__main__":
    main()
