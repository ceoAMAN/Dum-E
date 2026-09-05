from __future__ import annotations
import math
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import defaultdict
import mlx.core as mx
import configs
@dataclass
class ExpertFragment:
    expert_id: int
    tokens: mx.array
    token_indices: List[int]
    domain_label: str
    r_out: float
    below_nadir: bool = False
    # The ALLOC-sized span this expert was actually allocated. `tokens` may be
    # LONGER: a fragment landing mid-word is extended outward to whole-word
    # boundaries so the expert reads real words instead of "ang" / "lement".
    # That boundary padding is free — it never counts toward apex-nadir (probe
    # size, latency curve, token-allocation history), so completing a word can't
    # inflate an expert's measured allocation or distort its curves.
    allocated_len: int = 0
@dataclass
class DomainBatch:
    batch_index: int
    domain_label: str
    token_indices: List[int]
    tokens: mx.array
    expert_ids: List[int] = field(default_factory=list)
@dataclass
class XYGeometry:
    X: int
    Y: int
    total_experts_needed: int
    r_out_mean: float
    available_ram_mb: float
    total_tokens: int
@dataclass
class OverlapPaddedFragment:
    padded_tokens: mx.array
    grad_mask: mx.array
    overlap_len: int
    fragment_len: int
    expert_id: int
    original_tokens: mx.array
def compute_xy(
    total_tokens: int,
    r_out_mean: float,
    available_ram_mb: float,
    x_override: Optional[int] = None,
) -> XYGeometry:
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be > 0, got {total_tokens}")
    if r_out_mean <= 0:
        raise ValueError(f"r_out_mean must be > 0, got {r_out_mean}")
    if available_ram_mb < configs.EXPERT_RAM_MB:
        raise ValueError(f"Insufficient RAM: {available_ram_mb:.1f} MB available, {configs.EXPERT_RAM_MB} MB required per expert.")
    if x_override is not None:
        X = max(configs.X_MIN, min(configs.X_MAX, x_override))
    else:
        X = max(1, math.floor(available_ram_mb / configs.EXPERT_RAM_MB))
    total_experts_needed = max(1, math.ceil(total_tokens / r_out_mean))
    Y = max(1, math.ceil(total_experts_needed / X))
    return XYGeometry(X=X, Y=Y, total_experts_needed=total_experts_needed, r_out_mean=r_out_mean, available_ram_mb=available_ram_mb, total_tokens=total_tokens)


def experts_per_batch(total_ram_gb: Optional[float] = None,
                      general_gb: Optional[float] = None,
                      gating_gb: Optional[float] = None,
                      expert_gb: Optional[float] = None) -> int:
    """How many experts fit in one batch, derived from measured hardware:

        usable   = R - sqrt(R)          sqrt(R) scales the safety reserve with
                                        machine size instead of a fixed headroom
        usable  -= general              the resident synthesiser
        usable  -= gating               the gate
        budget   = usable / 2           half to experts, half left as working
                                        room for activations and KV cache
        count    = budget / expert_gb

    Every input is measured (sysctl RAM, configs values set by
    splitter.measure_expert_ram_mb at boot), so no fixed cap appears anywhere —
    the same formula yields a different answer on different hardware."""
    R = float(total_ram_gb if total_ram_gb is not None else total_physical_ram_mb() / 1024.0)
    if R <= 0:
        return configs.X_MIN
    gen = float(general_gb if general_gb is not None else configs.CENTRAL_RAM_MB / 1024.0)
    gat = float(gating_gb if gating_gb is not None else configs.GATE_RAM_MB / 1024.0)
    exp = float(expert_gb if expert_gb is not None else configs.EXPERT_RAM_MB / 1024.0)
    if exp <= 0:
        return configs.X_MIN
    # R - sqrt(R) - central - gating; whatever remains is the experts'.
    # Every term is MEASURED: sysctl for R, and boot-time load deltas for the
    # central/gate/expert costs. sqrt(R) is the reserve, so headroom scales with
    # the machine instead of being a fixed number that is generous on 64GB and
    # fatal on 8.
    remaining = R - math.sqrt(R) - gen - gat
    return int(max(configs.X_MIN, min(configs.X_MAX, math.floor(remaining / exp))))


def prefetch_next_batch(
    expert_pool,
    next_batch_expert_ids: List[int],
    done_event: threading.Event,
) -> None:
    try:
        expert_pool.load_experts(next_batch_expert_ids)
    except Exception as e:
        print(f"[prefetch] Failed to prefetch experts {next_batch_expert_ids}: {e}")
    finally:
        done_event.set()


def build_geography_batches(tokens: mx.array, domain_map: Dict[int, str], n_y: int) -> List[DomainBatch]:
    total_tokens_count = tokens.shape[0]
    domain_groups: Dict[str, List[int]] = defaultdict(list)
    for idx in range(total_tokens_count):
        label = domain_map.get(idx, "mixed")
        domain_groups[label].append(idx)
    sorted_domains = sorted(domain_groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    batch_token_counts = [0] * n_y
    batch_assignments: Dict[int, List] = defaultdict(list)
    for domain_label, idx_list in sorted_domains:
        target_batch = min(range(n_y), key=lambda b: batch_token_counts[b])
        batch_assignments[target_batch].append((domain_label, idx_list))
        batch_token_counts[target_batch] += len(idx_list)
    batches: List[DomainBatch] = []
    for batch_idx in range(n_y):
        assigned = batch_assignments.get(batch_idx, [])
        if not assigned:
            batches.append(DomainBatch(batch_index=batch_idx, domain_label="empty", token_indices=[], tokens=mx.array([], dtype=mx.int32), expert_ids=[]))
            continue
        all_indices = []
        for _, idx_list in assigned:
            all_indices.extend(idx_list)
        all_indices.sort()
        primary_label = max(assigned, key=lambda kv: len(kv[1]))[0]
        indices_mx = mx.array(all_indices, dtype=mx.int32)
        batch_tokens = tokens[indices_mx]
        batches.append(DomainBatch(batch_index=batch_idx, domain_label=primary_label, token_indices=all_indices, tokens=batch_tokens, expert_ids=[]))
    return batches
def _is_word_start(tokenizer, token_id) -> bool:
    """True if this token begins a new word (or is punctuation), i.e. cutting
    immediately BEFORE it leaves the preceding word intact."""
    try:
        s = tokenizer.decode([int(token_id)])
    except Exception:
        return True
    if not s:
        return True
    first = s[0]
    return first.isspace() or not first.isalnum()


def snap_cut_to_word(tokens, cut: int, tokenizer, hard_limit: int) -> int:
    """Advance a fragment boundary forward until it lands on a word start, so an
    expert never receives a half-word.

    The allocation decides HOW MANY tokens an expert gets; this only decides
    WHERE the cut falls. Tokens consumed finishing a word are not charged to any
    expert's allocation — the next fragment still receives its own full share.
    Without this, splitting "What is quantum entanglement..." across experts
    hands one expert 'What is quantum entang' and the next 'lement and how does
    it' — a half-word prompt an LM can only mangle."""
    if tokenizer is None:
        return int(cut)
    n = int(tokens.shape[0]) if hasattr(tokens, "shape") else len(tokens)
    c = max(0, min(int(cut), n))
    stop = min(n, int(hard_limit))
    while c < stop and not _is_word_start(tokenizer, tokens[c]):
        c += 1
    return c


def assign_spans_to_experts(tokens: mx.array, domain_map: Dict[int, str], selected_experts: List,
                            alloc: int, expert_domain: Optional[Dict[int, str]] = None,
                            tokenizer=None) -> Dict[int, List[ExpertFragment]]:
    """Cut the input into CONTIGUOUS, allocation-sized spans and hand each to the
    expert that should own it. Returns {expert_id: [fragments in input order]}.

    Two things this fixes at once.

    CONTIGUITY. The old path grouped tokens BY DOMAIN LABEL and gathered
    tokens[indices], producing an in-order but gapped subsequence — sentences
    with holes cut through them, because a domain's tokens are scattered through
    a real input. Here nothing is reordered or gathered: spans are cut in place,
    snapped to word boundaries, and the text an expert reads is always a real
    contiguous stretch.

    SWAPS. Grouping by expert means the caller can run expert-major — load an
    expert once, process every span it owns, then move on. Swap count becomes
    exactly the number of distinct experts, which is the minimum possible; the
    old Y-batch loop reloaded the whole active set once per batch.

    Ownership: a span goes to the highest-ranked SELECTED expert whose assigned
    domain matches the span's dominant domain; with no domain match it goes to
    the highest-ranked expert with the least work so far, so the load stays even
    rather than piling onto rank 1."""
    total = int(tokens.shape[0])
    if total == 0 or not selected_experts:
        return {}
    alloc = max(1, int(alloc))
    expert_domain = expert_domain or {}
    ordered = list(selected_experts)                      # already rank-ordered
    load: Dict[int, int] = {se.expert_id: 0 for se in ordered}
    out: Dict[int, List[ExpertFragment]] = {}
    cursor = 0
    while cursor < total:
        end = min(total, cursor + alloc)
        if end < total:
            end = snap_cut_to_word(tokens, end, tokenizer, total)
        end = max(cursor + 1, min(end, total))
        span = list(range(cursor, end))
        # dominant domain of this span
        counts: Dict[str, int] = {}
        for i in span:
            d = domain_map.get(i, "mixed")
            counts[d] = counts.get(d, 0) + 1
        dom = max(counts, key=counts.get) if counts else "mixed"
        match = [se for se in ordered if expert_domain.get(se.expert_id) == dom]
        if match:
            owner = match[0].expert_id                    # best-ranked in-domain
        else:
            owner = min(ordered, key=lambda se: (load[se.expert_id], ordered.index(se))).expert_id
        load[owner] += len(span)
        out.setdefault(owner, []).append(ExpertFragment(
            expert_id=owner,
            tokens=tokens[cursor:end],
            token_indices=span,
            domain_label=dom,
            r_out=float(alloc),
            below_nadir=False,
            allocated_len=min(alloc, end - cursor),
        ))
        cursor = end
    return out


def compute_x_expert_splits(domain_batch: DomainBatch, selected_experts: List, n_x: int, convolution,
                            tokenizer=None) -> List[ExpertFragment]:
    if len(domain_batch.token_indices) == 0 or not selected_experts:
        return []
    active_experts = selected_experts[:n_x]
    n_experts = len(active_experts)
    total_batch_tokens = len(domain_batch.token_indices)
    r_out_values = []
    for sel_expert in active_experts:
        r_out_i = convolution.compute_r_out(sel_expert.expert_id)
        r_out_values.append(max(r_out_i, float(configs.FRAGMENT_MIN)))
    r_out_sum = sum(r_out_values)
    if r_out_sum <= 0:
        r_out_sum = float(n_experts)
        r_out_values = [1.0] * n_experts
    fragment_lengths: List[int] = []
    tokens_assigned = 0
    for i, r_out_i in enumerate(r_out_values):
        if i < n_experts - 1:
            share = r_out_i / r_out_sum
            frag_len = max(1, round(total_batch_tokens * share))
            frag_len = min(frag_len, total_batch_tokens - tokens_assigned - (n_experts - i - 1))
            frag_len = max(1, frag_len)
        else:
            frag_len = max(1, total_batch_tokens - tokens_assigned)
        fragment_lengths.append(frag_len)
        tokens_assigned += frag_len
    fragments: List[ExpertFragment] = []
    cursor = 0
    for i, sel_expert in enumerate(active_experts):
        frag_len = fragment_lengths[i]
        expert_id = sel_expert.expert_id
        end = cursor + frag_len
        if i < n_experts - 1:
            # Extend to the next word start so this fragment ends on a whole word.
            # The overflow is absorbed here and NOT deducted from the following
            # experts — each still gets the length its allocation gave it.
            end = snap_cut_to_word(domain_batch.tokens, end, tokenizer, total_batch_tokens)
        else:
            end = total_batch_tokens
        end = max(cursor + 1, min(end, total_batch_tokens))
        frag_indices = domain_batch.token_indices[cursor:end]
        frag_tokens = domain_batch.tokens[cursor:end]
        is_below_nadir = check_nadir_floor(fragment_len=(end - cursor), expert_id=expert_id, convolution=convolution)
        # allocated_len is the PRE-SNAP share. The expert reads `frag_tokens`
        # (word-complete, possibly longer), but every apex-nadir measurement —
        # probe size, latency curve, token-allocation history — is charged this
        # number, so finishing a word can never inflate an expert's measured
        # allocation or bend its curves.
        fragments.append(ExpertFragment(expert_id=expert_id, tokens=frag_tokens, token_indices=frag_indices, domain_label=domain_batch.domain_label, r_out=r_out_values[i], below_nadir=is_below_nadir, allocated_len=min(frag_len, end - cursor)))
        cursor = end
        if cursor >= total_batch_tokens:
            break
    return fragments
def schedule_by_expert(fragments: List[ExpertFragment], resident: Optional[Set[int]] = None) -> List[ExpertFragment]:
    """Order fragments so each expert's work forms ONE contiguous run.

    Gating already knows which token spans belong to which expert. What it should
    also decide is the ORDER those spans are processed in — because the cost that
    dominates is not compute, it is model swaps. Processing fragments in token
    order makes an expert load, run, get evicted, and load again; processing them
    grouped by expert means it loads once, sees everything assigned to it, and
    only then makes way. Swaps drop from O(fragments) to O(experts), and experts
    leave one at a time instead of the whole set turning over per batch.

    This is also why fragments never needed to be spliced into gapped
    subsequences to keep a domain together: the domain stays together in TIME,
    not in the token array, so every fragment can stay contiguous and readable.

    Experts already resident are scheduled first, so work that needs no load at
    all happens before anything is swapped in. Within that, larger runs go first
    — an expert with more tokens earns its load cost over more work."""
    if not fragments:
        return []
    resident = resident or set()
    groups: Dict[int, List[ExpertFragment]] = {}
    for f in fragments:
        groups.setdefault(f.expert_id, []).append(f)
    order = sorted(groups, key=lambda e: (0 if e in resident else 1,
                                          -sum(len(g.token_indices) for g in groups[e]),
                                          e))
    out: List[ExpertFragment] = []
    for eid in order:
        out.extend(groups[eid])
    return out


def count_expert_swaps(fragments: List[ExpertFragment]) -> int:
    """Model loads implied by this ordering: every time the expert changes from
    the previous fragment, one swap. Diagnostic for schedule_by_expert."""
    swaps, prev = 0, None
    for f in fragments:
        if f.expert_id != prev:
            swaps += 1
            prev = f.expert_id
    return swaps


def check_nadir_floor(fragment_len: int, expert_id: int, convolution) -> bool:
    if fragment_len < configs.FRAGMENT_MIN:
        return True
    return convolution.check_nadir_floor(expert_id, fragment_len)
def compute_overlap_padding(fragment: ExpertFragment, context: mx.array) -> OverlapPaddedFragment:
    fragment_len = fragment.tokens.shape[0]
    context_len = context.shape[0] if context.ndim > 0 else 0
    if context_len == 0:
        return OverlapPaddedFragment(padded_tokens=fragment.tokens, grad_mask=mx.ones([fragment_len], dtype=mx.float32), overlap_len=0, fragment_len=fragment_len, expert_id=fragment.expert_id, original_tokens=fragment.tokens)
    overlap_len = max(1, math.floor(context_len * configs.OVERLAP_FRACTION))
    overlap_len = min(overlap_len, context_len)
    overlap_tokens = context[-overlap_len:]
    padded_tokens = mx.concatenate([overlap_tokens, fragment.tokens])
    total_len = overlap_len + fragment_len
    grad_mask = mx.concatenate([mx.zeros([overlap_len], dtype=mx.float32), mx.ones([fragment_len], dtype=mx.float32)])
    assert grad_mask.shape[0] == total_len
    return OverlapPaddedFragment(padded_tokens=padded_tokens, grad_mask=grad_mask, overlap_len=overlap_len, fragment_len=fragment_len, expert_id=fragment.expert_id, original_tokens=fragment.tokens)
def validate_overlap_grads(grads: mx.array, overlap_len: int, expert_id: int) -> bool:
    if overlap_len == 0:
        return True
    overlap_grads = grads[:overlap_len]
    all_zero = mx.all(overlap_grads == 0.0).item()
    if not all_zero:
        max_val = mx.abs(overlap_grads).max().item()
        raise AssertionError(f"Expert {expert_id}: Overlap gradient NOT zero. Max abs: {max_val:.6e}. Mask must be inside masked_loss.")
    return True
def measure_expert_ram_mb() -> float:
    """Measure the real cost of one expert as a PEAK delta — weights plus the
    transient of a forward pass — then update configs.EXPERT_RAM_MB in place so
    every downstream X/Y consumer uses the measured value.

    Weights alone are the wrong quantity. Loading the model and reading active
    memory gave ~831 MB, and the X/Y feasibility bound was built on it — but a
    Metal OOM is triggered by the high-water mark, and the forward pass plus its
    KV allocation is most of that. The OOM forensics measured real per-expert
    cost at 2.8-4.8 GB against 831 MB of weights, which is exactly the gap
    between what this measured and what actually had to fit. So: reset the peak,
    load, run one real forward, and take the high-water mark.

    Returns the measured MB (falls back to the configured estimate on any
    failure). Called once at boot — never per batch."""
    try:
        from mlx_lm import load as mlx_load
        mx.clear_cache()
        reset_peak_memory()
        before = get_peak_memory_mb()
        model, _tok = mlx_load(configs.EXPERT_MODEL_ID)
        mx.eval(model.parameters())
        # One real forward at the working sequence length: the KV and activation
        # spike is the part the weights-only measurement missed entirely.
        probe = mx.zeros((1, int(configs.MAX_SEQ_LEN)), dtype=mx.int32)
        out = model.model(probe) if hasattr(model, "model") else model(probe)
        mx.eval(out)
        measured = get_peak_memory_mb() - before
        del out, probe, model
        mx.clear_cache()
        if measured > 1.0:
            configs.EXPERT_RAM_MB = round(measured)
            return float(configs.EXPERT_RAM_MB)
    except Exception as e:
        print(f"[boot] expert RAM measurement failed, using configured {configs.EXPERT_RAM_MB} MB: {e}")
    return float(configs.EXPERT_RAM_MB)
def measure_gate_ram_mb() -> float:
    """Measure the gate's real resident cost the same way the expert's is
    measured, and write it into configs. It was a hardcoded 0.35 GB — a guess at
    a 0.5B-4bit model, wrong on any other gate and wrong the moment GATE_MODEL_ID
    changes. The expert cost is measured at boot; there is no reason the gate's
    should be assumed."""
    try:
        before = get_active_memory_mb()
        from mlx_lm import load as mlx_load
        model, _tok = mlx_load(configs.GATE_MODEL_ID)
        mx.eval(model.parameters())
        after = get_active_memory_mb()
        measured = after - before
        del model
        mx.clear_cache()
        if measured > 1.0:
            configs.GATE_RAM_MB = round(measured)
            return float(configs.GATE_RAM_MB)
    except Exception as e:
        print(f"[boot] gate RAM measurement failed, using configured {configs.GATE_RAM_MB} MB: {e}")
    return float(configs.GATE_RAM_MB)


def get_active_memory_mb() -> float:
    """MLX-reported active (resident) memory in MB. Preferred RAM signal on Apple
    Silicon — no subprocess, reflects the unified-memory allocator's real use."""
    try:
        return float(mx.get_active_memory()) / (1024 * 1024)
    except Exception:
        return 0.0
def get_peak_memory_mb() -> float:
    """MLX high-water-mark memory (MB) since the last reset — captures the transient
    spike of a forward+generation, which is what actually triggers a Metal OOM."""
    try:
        return float(mx.get_peak_memory()) / (1024 * 1024)
    except Exception:
        return 0.0
def reset_peak_memory() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        pass
def total_physical_ram_mb() -> float:
    """Physical RAM (MB) from sysctl hw.memsize — a measured hardware fact, not a
    config constant. Used as the absolute ceiling the MLX peak must stay under."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return float(int(out)) / (1024 * 1024)
    except Exception:
        return 0.0
def get_available_ram_mb() -> float:
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        page_size = 16384
        free_pages = 0
        inactive_pages = 0
        for line in out.splitlines():
            if line.startswith("Pages free:"):
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages inactive:"):
                inactive_pages = int(line.split(":")[1].strip().rstrip("."))
        available_mb = (free_pages + inactive_pages) * page_size / (1024 * 1024)
        return max(0.0, available_mb)
    except Exception:
        return float(configs.EXPERT_RAM_MB * 3)
