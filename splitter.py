from __future__ import annotations
import math
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict
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
def compute_xy(total_tokens: int, r_out_mean: float, available_ram_mb: float) -> XYGeometry:
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be > 0, got {total_tokens}")
    if r_out_mean <= 0:
        raise ValueError(f"r_out_mean must be > 0, got {r_out_mean}")
    if available_ram_mb < configs.EXPERT_RAM_MB:
        raise ValueError(f"Insufficient RAM: {available_ram_mb:.1f} MB available, {configs.EXPERT_RAM_MB} MB required per expert.")
    X = max(1, math.floor(available_ram_mb / configs.EXPERT_RAM_MB))
    total_experts_needed = max(1, math.ceil(total_tokens / r_out_mean))
    Y = max(1, math.ceil(total_experts_needed / X))
    return XYGeometry(X=X, Y=Y, total_experts_needed=total_experts_needed, r_out_mean=r_out_mean, available_ram_mb=available_ram_mb, total_tokens=total_tokens)
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
def compute_x_expert_splits(domain_batch: DomainBatch, selected_experts: List, n_x: int, convolution) -> List[ExpertFragment]:
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
        frag_indices = domain_batch.token_indices[cursor:cursor + frag_len]
        frag_tokens = domain_batch.tokens[cursor:cursor + frag_len]
        is_below_nadir = check_nadir_floor(fragment_len=frag_len, expert_id=expert_id, convolution=convolution)
        fragments.append(ExpertFragment(expert_id=expert_id, tokens=frag_tokens, token_indices=frag_indices, domain_label=domain_batch.domain_label, r_out=r_out_values[i], below_nadir=is_below_nadir))
        cursor += frag_len
    return fragments
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
        usable_mb = max(0.0, available_mb)
        print(f"[splitter] RAM: {available_mb:.0f} MB currently available")
        return usable_mb
    except Exception:
        return float(configs.EXPERT_RAM_MB * 3)
