from __future__ import annotations
import os
import random
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional
import mlx.core as mx
import configs
@dataclass
class Sample:
    source: str
    text: str
    raw: Dict[str, Any]
    # The question/answer split, from extract_pair. `text` stays exactly what it
    # was (the flattened row) so every existing consumer is unaffected; these are
    # additive. `answer` is the ground truth y that the grounded delta,
    # expert class standing and Central's trust score are all computed against —
    # without it none of those can exist at all.
    prompt: Optional[str] = None
    answer: Optional[str] = None
    # `verifiable` is the externally checkable final value where one exists — the
    # string after "####" for gsm8k, the correct option text for ai2_arc — and ""
    # otherwise. A STRING, not a flag: the value is what an exact-match check
    # compares against, and collapsing it to a bool throws that away while
    # looking like it works.
    verifiable: str = ""

    @property
    def has_target(self) -> bool:
        return bool(self.answer)

    @property
    def has_verifiable(self) -> bool:
        return bool(self.verifiable)
_TEXT_KEYS = ("text", "content", "code", "prompt", "response", "instruction")


def extract_pair(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Split one dataset row into {prompt, answer, verifiable} against the REAL
    schema of each configured dataset.

    Why this exists: _extract_text() below flattens a row into one undifferentiated
    string, which destroys the question/answer boundary. Nothing downstream could
    then produce `target_ids`, so grounded delta (central.compute_grounded_r_i)
    has never had a caller and every quality signal in the system fell back to
    cosine agreement with Central's own hidden state.

    Its format guessing was also silently WRONG on three of the eight configured
    datasets — verified by running it against live rows:

      CodeAlpaca_20K  has prompt/completion, matches no branch, falls through to
                      _TEXT_KEYS and returns the PROMPT ALONE. The dataset mapped
                      to the `code` domain contained no code.
      ai2_arc         choices live in a dict, which the string-only fallback drops;
                      the row ID leaks in as text and the answer is a bare letter
                      referring to nothing.
      sciq            the three distractors are concatenated alongside the correct
                      answer with nothing marking which is which — training on
                      wrong answers presented as content.

    `verifiable` is the externally checkable answer where one exists (a final
    number, a multiple-choice option) — the anti-hallucination signal, since a
    confidently-worded wrong number scores zero against it no matter how fluent.
    Empty string when the answer is open-ended and only CE applies.

    Returns None when the row matches no known schema; the caller falls back to
    _extract_text rather than inventing a split."""
    def _s(key: str, default: str = "") -> str:
        v = example.get(key, default)
        return v.strip() if isinstance(v, str) else default

    # ── CodeAlpaca_20K: prompt / completion ─────────────────────────────────
    if "completion" in example and "prompt" in example:
        p, a = _s("prompt"), _s("completion")
        if p and a:
            return {"prompt": p, "answer": a, "verifiable": ""}

    # ── alpaca instruction / input / output ─────────────────────────────────
    if "instruction" in example and "output" in example:
        instr, inp, out = _s("instruction"), _s("input") or _s("context"), _s("output")
        if instr and out:
            return {"prompt": f"{instr}\n{inp}".strip(), "answer": out, "verifiable": ""}

    # ── ai2_arc: question + choices{text,label} + answerKey ─────────────────
    ch = example.get("choices")
    if "answerKey" in example and isinstance(ch, dict) and "text" in ch and "label" in ch:
        q = _s("question"); key = _s("answerKey")
        texts, labels = list(ch["text"]), [str(l) for l in ch["label"]]
        if q and texts and key in labels:
            correct = texts[labels.index(key)]
            rendered = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            # Choices belong in the PROMPT. Dropping them (the old behaviour) left
            # a bare letter answering a question with no options to choose from.
            return {"prompt": f"{q}\n{rendered}", "answer": correct, "verifiable": correct}

    # ── sciq: question + correct_answer + distractor1..3 ────────────────────
    if "correct_answer" in example and "question" in example:
        q, correct = _s("question"), _s("correct_answer")
        if q and correct:
            # Distractors are deliberately NOT included. They are wrong answers;
            # the old path concatenated them into the text unlabelled.
            return {"prompt": q, "answer": correct, "verifiable": correct}

    # ── gsm8k: question + answer, final value after "####" ──────────────────
    if "question" in example and "answer" in example:
        q, a = _s("question"), _s("answer")
        if q and a:
            final = a.split("####")[-1].strip() if "####" in a else ""
            return {"prompt": q, "answer": a, "verifiable": final}

    # ── MetaMathQA: query + response, final value after "The answer is:" ────
    if "query" in example and "response" in example:
        q, a = _s("query"), _s("response")
        if q and a:
            marker = "The answer is:"
            final = a.split(marker)[-1].strip() if marker in a else ""
            return {"prompt": q, "answer": a, "verifiable": final}

    # ── chat formats: last assistant turn is the answer, everything before is
    #    the prompt. Covers SlimOrca (conversations/from/value) and UltraChat
    #    (messages/role/content).
    for field, role_key, text_key, assistant in (
        ("conversations", "from", "value", ("gpt", "assistant")),
        ("messages", "role", "content", ("assistant", "gpt")),
    ):
        conv = example.get(field)
        if not isinstance(conv, list) or not conv:
            continue
        turns = [(str(m.get(role_key, "")), str(m.get(text_key, "")))
                 for m in conv if isinstance(m, dict)]
        last = next((i for i in range(len(turns) - 1, -1, -1)
                     if turns[i][0].lower() in assistant and turns[i][1].strip()), None)
        if last is None or last == 0:
            continue
        prompt = "\n".join(f"{r}: {t.strip()}" for r, t in turns[:last] if t.strip())
        if prompt and turns[last][1].strip():
            return {"prompt": prompt, "answer": turns[last][1].strip(), "verifiable": ""}

    return None
_DATASET_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}
def authenticate_huggingface():
    token = configs.HF_TOKEN
    if not token:
        token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "HuggingFace token not found. "
            "Set the HF_TOKEN environment variable before running: "
            "export HF_TOKEN='hf_your_token_here'"
        )
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
def get_tokenizer(model_id: str):
    if model_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_id]
    from mlx_lm import load as mlx_load
    _, tokenizer = mlx_load(model_id)
    _TOKENIZER_CACHE[model_id] = tokenizer
    return tokenizer
def _extract_text(example: Dict[str, Any]) -> str:
    if "messages" in example and isinstance(example["messages"], list):
        parts = []
        for m in example["messages"]:
            role = m.get("role", m.get("from", ""))
            value = m.get("content", m.get("value", ""))
            if isinstance(value, str) and value.strip():
                parts.append(f"{role}: {value}")
        if parts:
            return "\n".join(parts)
    # "conversations" (plural, ShareGPT) or "conversation" (singular, Agent-FLAN).
    conv = example.get("conversations")
    if conv is None:
        conv = example.get("conversation")
    if isinstance(conv, list) and conv:
        parts = []
        for m in conv:
            if not isinstance(m, dict):
                parts.append(str(m))
                continue
            role = m.get("from", m.get("role", ""))
            value = m.get("value", m.get("content", ""))
            if isinstance(value, str) and value.strip():
                parts.append(f"{role}: {value}")
        if parts:
            return "\n".join(parts)
    if "instruction" in example and "output" in example:
        parts = []
        instruction = example.get("instruction")
        input_text = example.get("input", example.get("context", ""))
        output = example.get("output")
        if isinstance(instruction, str) and instruction.strip():
            parts.append(f"instruction: {instruction}")
        if isinstance(input_text, str) and input_text.strip():
            parts.append(f"input: {input_text}")
        if isinstance(output, str) and output.strip():
            parts.append(f"output: {output}")
        if parts:
            return "\n".join(parts)
    if "instruction" in example and "response" in example:
        parts = []
        instruction = example.get("instruction")
        context = example.get("context", "")
        response = example.get("response")
        if isinstance(instruction, str) and instruction.strip():
            parts.append(f"instruction: {instruction}")
        if isinstance(context, str) and context.strip():
            parts.append(f"context: {context}")
        if isinstance(response, str) and response.strip():
            parts.append(f"response: {response}")
        if parts:
            return "\n".join(parts)
    if "question" in example and "response" in example:
        parts = []
        system_prompt = example.get("system_prompt", "")
        question = example.get("question")
        response = example.get("response")
        if isinstance(system_prompt, str) and system_prompt.strip():
            parts.append(f"system: {system_prompt}")
        if isinstance(question, str) and question.strip():
            parts.append(f"question: {question}")
        if isinstance(response, str) and response.strip():
            parts.append(f"response: {response}")
        if parts:
            return "\n".join(parts)
    if "question" in example and "answer" in example:
        parts = []
        question = example.get("question")
        answer = example.get("answer")
        if isinstance(question, str) and question.strip():
            parts.append(f"question: {question}")
        if isinstance(answer, str) and answer.strip():
            parts.append(f"answer: {answer}")
        if parts:
            return "\n".join(parts)
    if "query" in example and "response" in example:
        parts = []
        query = example.get("query")
        response = example.get("response")
        if isinstance(query, str) and query.strip():
            parts.append(f"query: {query}")
        if isinstance(response, str) and response.strip():
            parts.append(f"response: {response}")
        if parts:
            return "\n".join(parts)
    if "problem" in example and "generated_solution" in example:
        parts = []
        problem = example.get("problem")
        solution = example.get("generated_solution")
        expected_answer = example.get("expected_answer", "")
        if isinstance(problem, str) and problem.strip():
            parts.append(f"problem: {problem}")
        if isinstance(solution, str) and solution.strip():
            parts.append(f"solution: {solution}")
        if isinstance(expected_answer, str) and expected_answer.strip():
            parts.append(f"answer: {expected_answer}")
        if parts:
            return "\n".join(parts)
    # xlam function-calling format: instruction + tool definitions + answers
    if "functions" in example and "answers" in example:
        parts = []
        instr = example.get("instruction", "")
        if isinstance(instr, str) and instr.strip():
            parts.append(f"instruction: {instr}")
        functions = example.get("functions", "")
        if isinstance(functions, str) and functions.strip():
            parts.append(f"tools: {functions}")
        answers = example.get("answers", "")
        if isinstance(answers, str) and answers.strip():
            parts.append(f"tool_calls: {answers}")
        if parts:
            return "\n".join(parts)

    # glaive function-calling format: system + raw chat string
    if "chat" in example:
        parts = []
        system = example.get("system", "")
        if isinstance(system, str) and system.strip():
            parts.append(system.strip())
        chat = example.get("chat", "")
        if isinstance(chat, str) and chat.strip():
            parts.append(chat.strip())
        if parts:
            return "\n".join(parts)

    # agent trajectory format (AgentInstruct / Agent-FLAN ReAct traces)
    if "trajectory" in example and isinstance(example["trajectory"], list):
        parts = []
        instr = example.get("instruction", example.get("task", ""))
        if isinstance(instr, str) and instr.strip():
            parts.append(f"task: {instr}")
        for step in example["trajectory"][:8]:
            if isinstance(step, dict):
                for key in ("thought", "reasoning", "action", "observation", "result"):
                    v = step.get(key, "")
                    if isinstance(v, str) and v.strip():
                        parts.append(f"{key}: {v.strip()[:300]}")
        if parts:
            return "\n".join(parts)

    for key in _TEXT_KEYS:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    parts = [str(v) for v in example.values() if isinstance(v, str) and v.strip()]
    if parts:
        return "\n".join(parts)
    return str(example)
def _load_stream(dataset_key: str):
    if dataset_key in _DATASET_CACHE:
        return _DATASET_CACHE[dataset_key]
    from datasets import load_dataset
    dataset_spec = configs.DATASET_IDS[dataset_key]
    if len(dataset_spec) == 3:
        dataset_id, dataset_cfg, dataset_split = dataset_spec
    else:
        dataset_id, dataset_cfg = dataset_spec
        dataset_split = "train"
        if dataset_key == "ultrachat":
            dataset_split = "train_sft"
    print(f"[data] Opening stream {dataset_key} ({dataset_id})")
    kwargs: Dict[str, Any] = {"split": dataset_split, "streaming": True}
    if dataset_cfg:
        if isinstance(dataset_cfg, dict):
            kwargs["data_files"] = dataset_cfg
        else:
            kwargs["name"] = dataset_cfg
    if configs.HF_TOKEN:
        kwargs["token"] = configs.HF_TOKEN
    ds = load_dataset(dataset_id, **kwargs)
    _DATASET_CACHE[dataset_key] = ds
    return ds
def _weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    r = rng.random()
    cumulative = 0.0
    for key, weight in weights.items():
        cumulative += weight
        if r <= cumulative:
            return key
    return list(weights.keys())[-1]
def iter_dataset_samples(dataset_key: str) -> Iterator[Sample]:
    ds = _load_stream(dataset_key)
    for row in ds:
        text = _extract_text(row)
        # extract_pair understands each configured dataset's REAL schema;
        # _extract_text flattens the row and loses the question/answer boundary.
        # Both are produced: `text` for the unchanged consumers, the pair for
        # everything that needs a ground truth.
        pair = extract_pair(row) or {}
        yield Sample(source=dataset_key, text=text, raw=row,
                     prompt=pair.get("prompt"), answer=pair.get("answer"),
                     verifiable=(pair.get("verifiable") or ""))
def _next_with_timeout(
    stream: Iterator[Sample],
    dataset_key: str,
    timeout: Optional[float] = None,
) -> Sample:
    result: Dict[str, Any] = {}
    done = threading.Event()
    wait_timeout = timeout if timeout is not None else configs.DATASET_SAMPLE_TIMEOUT
    def runner():
        try:
            result["sample"] = next(stream)
        except BaseException as e:
            result["error"] = e
        finally:
            done.set()
    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    if not done.wait(wait_timeout):
        raise TimeoutError(f"{dataset_key} timed out after {wait_timeout:.0f}s")
    error = result.get("error")
    if error is not None:
        raise error
    return result["sample"]
def iter_mixture_samples(seed: int = 42) -> Iterator[Sample]:
    rng = random.Random(seed)
    streams = {}
    cold_streams = set()
    failed_keys = set()
    # Only initialise streams with positive weight. Zero-weight datasets (e.g.
    # disabled because they hang on the HF stream open) must NOT be opened at all,
    # otherwise their slow/timeout load still stalls boot even though they're never
    # sampled.
    enabled_keys = [k for k, w in configs.DATASET_WEIGHTS.items() if w > 0.0]
    print(f"[data] Initialising mixture streams: {enabled_keys}")
    for key in enabled_keys:
        try:
            streams[key] = iter_dataset_samples(key)
            cold_streams.add(key)
        except Exception as e:
            print(f"[data] Failed to load {key}: {e}")
            failed_keys.add(key)
    active_weights = {k: v for k, v in configs.DATASET_WEIGHTS.items()
                      if k not in failed_keys and v > 0.0}
    if not active_weights:
        raise RuntimeError("All datasets failed to load.")
    total = sum(active_weights.values())
    active_weights = {k: v / total for k, v in active_weights.items()}
    while True:
        chosen = _weighted_choice(rng, active_weights)
        try:
            timeout = configs.DATASET_BOOT_TIMEOUT if chosen in cold_streams else None
            sample = _next_with_timeout(streams[chosen], chosen, timeout=timeout)
            cold_streams.discard(chosen)
            yield sample
        except StopIteration:
            try:
                streams[chosen] = iter_dataset_samples(chosen)
                cold_streams.add(chosen)
                sample = _next_with_timeout(
                    streams[chosen],
                    chosen,
                    timeout=configs.DATASET_BOOT_TIMEOUT,
                )
                cold_streams.discard(chosen)
                yield sample
            except Exception:
                failed_keys.add(chosen)
                active_weights = {k: v for k, v in configs.DATASET_WEIGHTS.items() if k not in failed_keys and v > 0.0}
                if not active_weights:
                    return
                total = sum(active_weights.values())
                active_weights = {k: v / total for k, v in active_weights.items()}
        except Exception as e:
            print(f"[data] Error reading {chosen}: {e}")
            failed_keys.add(chosen)
            active_weights = {k: v for k, v in configs.DATASET_WEIGHTS.items() if k not in failed_keys and v > 0.0}
            if not active_weights:
                return
            total = sum(active_weights.values())
            active_weights = {k: v / total for k, v in active_weights.items()}
            continue
def tokenize_for_gate(texts: List[str], max_length: int = configs.MAX_SEQ_LEN) -> List[mx.array]:
    tokenizer = get_tokenizer(configs.GATE_MODEL_ID)
    results = []
    for text in texts:
        ids = tokenizer.encode(text[:max_length * 6])[:max_length]
        results.append(mx.array(ids))
    return results
def tokenize_for_expert(texts: List[str], max_length: int = configs.MAX_SEQ_LEN) -> List[mx.array]:
    tokenizer = get_tokenizer(configs.EXPERT_MODEL_ID)
    results = []
    for text in texts:
        ids = tokenizer.encode(text[:max_length * 6])[:max_length]
        results.append(mx.array(ids))
    return results
def tokenize_for_central(texts: List[str], max_length: int = configs.MAX_SEQ_LEN) -> List[mx.array]:
    tokenizer = get_tokenizer(configs.CENTRAL_MODEL_ID)
    results = []
    for text in texts:
        ids = tokenizer.encode(text[:max_length * 6])[:max_length]
        results.append(mx.array(ids))
    return results
class StreamingDataset:
    def __init__(self, model_id: str, batch_size: int = 4, max_length: int = configs.MAX_SEQ_LEN, seed: int = 42):
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
    def __iter__(self) -> Iterator[List[mx.array]]:
        tokenizer = get_tokenizer(self.model_id)
        samples = iter_mixture_samples(seed=self.seed)
        while True:
            batch_texts = []
            for _ in range(self.batch_size):
                batch_texts.append(next(samples).text)
            batch_tokens = []
            for text in batch_texts:
                ids = tokenizer.encode(text[:self.max_length * 6])[:self.max_length]
                batch_tokens.append(mx.array(ids))
            yield batch_tokens
class DomainLabelledStream:
    def __init__(self, dataset_ids: Optional[Dict] = None):
        self.dataset_ids = dataset_ids or configs.DATASET_IDS
    def iter_calibration_batches(self, expert_id: int) -> Iterator[Dict[str, Any]]:
        for domain_key in self.dataset_ids:
            try:
                stream = iter_dataset_samples(domain_key)
                token_counts = []
                quality_scores = []
                gradient_coherence = []
                wall_times = []
                for i, sample in enumerate(stream):
                    if i >= 50:
                        break
                    text = sample.text
                    tc = len(text.split())
                    token_counts.append(tc)
                    quality_scores.append(1.0 / (1.0 + 0.01 * abs(tc - 128)))
                    gradient_coherence.append(min(1.0, tc / 64.0))
                    wall_times.append(tc * 0.001)
                if token_counts:
                    yield {
                        "domain": domain_key,
                        "expert_id": expert_id,
                        "token_counts": token_counts,
                        "quality_scores": quality_scores,
                        "gradient_coherence": gradient_coherence,
                        "wall_times": wall_times,
                    }
            except Exception:
                continue
def iter_mixture_token_batches(
    model_id: str,
    batch_size: int = 4,
    max_length: int = configs.MAX_SEQ_LEN,
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    tokenizer = get_tokenizer(model_id)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    samples = iter_mixture_samples(seed=seed)
    while True:
        batch_ids = []
        for _ in range(batch_size):
            text = next(samples).text
            ids = tokenizer.encode(text[:max_length * 6])[:max_length]
            batch_ids.append(ids)
        longest = max(len(ids) for ids in batch_ids)
        input_ids = []
        attention_mask = []
        for ids in batch_ids:
            pad_len = longest - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        yield {
            "input_ids": mx.array(input_ids),
            "attention_mask": mx.array(attention_mask),
        }
def iter_group_token_batches(
    group_name: str,
    model_id: str,
    batch_size: int = 4,
    max_length: int = configs.MAX_SEQ_LEN,
) -> Iterator[Dict[str, Any]]:
    domain_dataset_map = {
        "code": "codeparrot_clean",
        "reasoning": "gsm8k",
        "knowledge": "wikitext",
        "general": "ultrachat",
    }
    dataset_key = domain_dataset_map.get(group_name)
    if dataset_key is None or dataset_key not in configs.DATASET_WEIGHTS:
        dataset_key = list(configs.DATASET_WEIGHTS.keys())[0]
    tokenizer = get_tokenizer(model_id)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    stream = iter_dataset_samples(dataset_key)
    while True:
        batch_ids = []
        for _ in range(batch_size):
            try:
                text = next(stream).text
            except StopIteration:
                stream = iter_dataset_samples(dataset_key)
                text = next(stream).text
            ids = tokenizer.encode(text[:max_length * 6])[:max_length]
            batch_ids.append(ids)
        longest = max(len(ids) for ids in batch_ids)
        input_ids = []
        attention_mask = []
        for ids in batch_ids:
            pad_len = longest - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        yield {
            "input_ids": mx.array(input_ids),
            "attention_mask": mx.array(attention_mask),
        }
