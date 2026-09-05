from __future__ import annotations
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import configs
from apex_nadir_convolution import ApexNadirConvolution

# Domain slots, in the order the gate's leading domain-logit dims map to and the
# L_dom one-hot target uses. Single source of truth shared by routing +
# topography. D = len(DOMAINS) is a VARIABLE: everything that consumes it slices
# by len(DOMAINS), never by a literal, so a domain can be added or removed here
# alone. configs.DOMAINS overrides it when set.
DOMAINS: List[str] = list(getattr(configs, "DOMAINS", None) or
                          ["code", "reasoning", "knowledge", "general"])
@dataclass
class DomainTopography:
    domain_map: Dict[int, str]
    domain_proportions: Dict[str, float]
    total_tokens: int
@dataclass
class GateOutput:
    hidden_states: mx.array
    k_per_token: int
    domain_logits: mx.array
    timeline_flag: str
    confidence: float
    # Learned per-expert routing preference from GateNet.route_head (shape
    # [EXPERT_POOL_SIZE]). None only on the NaN-guard fallback path. select_experts
    # blends this with apex-nadir distance-to-peak.
    route_logits: Optional[mx.array] = None
@dataclass
class SelectedExpert:
    expert_id: int
    distance_to_peak: float
    domain: str
    is_alpha: bool
class GateNet(nn.Module):
    """The trainable gate as one unit, so a single value_and_grad call updates the
    backbone's LoRA params and the expert-routing head together. `backbone` is the
    LoRA'd Qwen (only its LoRA adapters are unfrozen); `route_head` maps the pooled
    gate hidden state to one preference logit per expert — the differentiable path
    that L_eff/L_rel use to teach the gate which experts to prefer."""
    def __init__(self, backbone: nn.Module, d_model: int, n_experts: int):
        super().__init__()
        self.backbone = backbone
        self.route_head = nn.Linear(d_model, n_experts)

class GateModel:
    def __init__(self):
        self.model = None        # LoRA'd Qwen backbone (also held by self.net.backbone)
        self.tokenizer = None
        self.net = None          # GateNet: backbone + route_head, the trainable unit
        self._loaded = False
        # The gate had no optimiser anywhere in the system, so apply_gate_gradients
        # had nothing to be called with and route_head stayed at its random init
        # for the life of the process — while route_pref was the entire expert
        # ranking (spread 3.2e-2 against a 1e-6 jitter). Built lazily: it must be
        # created after load() so it sees the real parameter tree.
        self._optimizer = None

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = optim.Adam(learning_rate=configs.LEARNING_RATE)
        return self._optimizer
    @property
    def route_head(self):
        return self.net.route_head if self.net is not None else None
    def _route_head_path(self) -> Path:
        return Path(configs.CHECKPOINT_DIR) / "gate" / "route_head.safetensors"
    def load(self):
        if self._loaded:
            return
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.utils import linear_to_lora_layers
        self.model, self.tokenizer = mlx_load(configs.GATE_MODEL_ID)
        self.model.freeze()
        lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA / configs.LORA_R, "dropout": configs.LORA_DROPOUT}
        num_layers = len(self.model.layers) if hasattr(self.model, "layers") else len(self.model.model.layers)
        linear_to_lora_layers(self.model, num_layers, lora_config)
        weights_path = Path(configs.CHECKPOINT_DIR) / "gate" / "weights.safetensors"
        if weights_path.exists():
            self.model.load_weights(str(weights_path), strict=False)
        # Wrap backbone + a fresh routing head into the trainable unit. The head's
        # own checkpoint is restored separately so it survives restarts.
        self.net = GateNet(self.model, configs.GATE_D_MODEL, configs.EXPERT_POOL_SIZE)
        rh_path = self._route_head_path()
        if rh_path.exists():
            self._load_route_head(rh_path)
        mx.eval(self.net.route_head.parameters())
        self.model.train()
        self._loaded = True
    def _load_route_head(self, path: Path):
        """Restore the routing head, GROWING or SHRINKING it to the current pool.

        route_head is the only parameter whose shape depends on E, so it is the
        one thing that breaks when the pool changes size. `load_weights(...,
        strict=False)` does not merely skip a mismatched tensor — it REPLACES it,
        silently resizing a 120-row head back to the saved 100 rows. The new
        experts then never receive a routing preference (a bounds guard in
        select_experts hides it), and the next save_route_head persists the
        shrunken head for good.

        Instead: copy the overlapping rows and leave the rest at init. Adding
        experts preserves everything the gate learned about the existing ones and
        gives the new arrivals a fresh, unbiased start; removing experts keeps the
        rows that still correspond to live experts."""
        from safetensors.numpy import load_file
        try:
            saved = load_file(str(path))
        except Exception as e:
            print(f"[warn] route_head load failed ({e}); keeping fresh init")
            return
        head = self.net.route_head
        for name, cur in (("weight", head.weight), ("bias", getattr(head, "bias", None))):
            if cur is None or name not in saved:
                continue
            old = mx.array(saved[name])
            if old.shape == cur.shape:
                setattr(head, name, old)
                continue
            n = min(int(old.shape[0]), int(cur.shape[0]))
            if old.ndim == 2 and cur.ndim == 2 and int(old.shape[1]) != int(cur.shape[1]):
                print(f"[warn] route_head.{name} d_model {old.shape[1]} != {cur.shape[1]}; keeping fresh init")
                continue
            merged = mx.concatenate([old[:n], cur[n:]], axis=0) if n < int(cur.shape[0]) else old[:int(cur.shape[0])]
            setattr(head, name, merged)
            print(f"[boot] route_head.{name} resized {tuple(old.shape)} -> {tuple(cur.shape)}: "
                  f"kept {n} learned rows, {max(0, int(cur.shape[0]) - n)} new experts at init")

    def save_route_head(self):
        from mlx.utils import tree_flatten
        p = self._route_head_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(p), dict(tree_flatten(self.net.route_head.parameters())))
    def _backbone(self, tokens: mx.array) -> mx.array:
        """One full backbone pass → per-token hidden states (1, T, D). Shared by
        forward(), look_ahead() and forward_with_topography() so a single request
        never runs the backbone twice."""
        self.load()
        if hasattr(self.model, 'model'):
            hidden = self.model.model(tokens.reshape(1, -1))
        else:
            hidden = self.model(tokens.reshape(1, -1))
        mx.eval(hidden)
        return hidden
    def _topography_from_hidden(self, hidden: mx.array, total: int) -> DomainTopography:
        domain_map: Dict[int, str] = {}
        if total == 0:
            return DomainTopography(domain_map={}, domain_proportions={}, total_tokens=0)
        domain_counts: Dict[str, int] = {}
        chunk_size = max(1, total // 10)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk_hidden = hidden[0, start:end, :]
            mean_hidden = mx.mean(chunk_hidden, axis=0)
            domain = self._domain_from_mean_hidden(mean_hidden)
            for idx in range(start, end):
                domain_map[idx] = domain
            domain_counts[domain] = domain_counts.get(domain, 0) + (end - start)
        domain_proportions = {d: c / total for d, c in domain_counts.items()}
        return DomainTopography(domain_map=domain_map, domain_proportions=domain_proportions, total_tokens=total)
    def look_ahead(self, tokens: mx.array) -> DomainTopography:
        total = int(tokens.shape[0])
        if total == 0:
            return DomainTopography(domain_map={}, domain_proportions={}, total_tokens=0)
        return self._topography_from_hidden(self._backbone(tokens), total)
    def forward_with_topography(self, tokens: mx.array):
        """Fused: ONE backbone pass → (GateOutput, DomainTopography). Timeline B
        previously paid for two full gate passes per request (forward() then
        look_ahead()); this collapses them into one — the single biggest routing
        overhead on the inference path."""
        hidden = self._backbone(tokens)
        gate_out = self.forward(tokens, hidden=hidden)
        topo = self._topography_from_hidden(hidden, int(tokens.shape[0]))
        return gate_out, topo
    def _domain_from_mean_hidden(self, mean_hidden: mx.array) -> str:
        """Domain of a (chunk) mean hidden state from the gate's LEARNED domain head:
        the same z-score → domain_logits[:D] argmax that forward() uses for routing.
        No hardcoded variance/magnitude thresholds — the trained gate decides."""
        h = mx.clip(mean_hidden, -1e4, 1e4)
        mu = mx.mean(h)
        sigma = mx.sqrt(mx.mean((h - mu) ** 2) + 1e-8)
        domain_logits = ((h - mu) / (sigma + 1e-8))[:len(DOMAINS)]
        return DOMAINS[int(mx.argmax(domain_logits).item())]
    def forward(self, tokens: mx.array, hidden: mx.array = None) -> GateOutput:
        self.load()
        # Reuse a precomputed backbone pass when the caller already ran one
        # (forward_with_topography). Otherwise run it here. Either way the gate
        # backbone executes exactly once per request.
        if hidden is None:
            hidden = self._backbone(tokens)
        mean_hidden = mx.mean(hidden[0], axis=0)
        mean_hidden = mx.clip(mean_hidden, -1e4, 1e4)
        mx.eval(mean_hidden)
        # Guard against NaN hidden states (corrupt checkpoint or exploding backbone).
        import math as _math
        spot_vals = mean_hidden[:8].tolist()
        if any(not _math.isfinite(v) for v in spot_vals):
            domain_logits = mx.zeros(len(DOMAINS))
            k = configs.K_DEFAULT
            return GateOutput(
                hidden_states=mean_hidden,
                k_per_token=k,
                domain_logits=domain_logits,
                timeline_flag="B",
                confidence=0.0,
            )
        # z-score normalise across ALL hidden dims before slicing domain logits.
        # Raw backbone activations have large magnitudes (e.g. ±100) → softmax
        # saturates → confidence locks at 1.0 before any training. After
        # normalisation values are ~N(0,1), entropy is meaningful, and the gate
        # can actually learn to discriminate domains via l_dom gradients.
        mu = mx.mean(mean_hidden)
        sigma = mx.sqrt(mx.mean((mean_hidden - mu) ** 2) + 1e-8)
        normed = (mean_hidden - mu) / (sigma + 1e-8)
        domain_logits = normed[:len(DOMAINS)]   # D slots, one per DOMAINS entry
        # Learned per-expert routing preference from the same pooled hidden state.
        route_logits = self.net.route_head(mean_hidden)
        mx.eval(domain_logits, route_logits)
        probs = mx.softmax(domain_logits)
        mx.eval(probs)
        entropy = -float(mx.sum(probs * mx.log(probs + 1e-10)).item())
        max_entropy = math.log(max(2, len(DOMAINS)))   # D domain classes
        confidence = max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))
        if confidence > configs.FAST_PATH_THRESHOLD:
            timeline = "A"
        else:
            timeline = "B"
        k = max(configs.K_MIN, min(configs.K_MAX, int((1.0 - confidence) * configs.K_MAX)))
        return GateOutput(hidden_states=mean_hidden, k_per_token=k, domain_logits=domain_logits, timeline_flag=timeline, confidence=confidence, route_logits=route_logits)
    def parameters(self) -> dict:
        if self.model is not None:
            return self.model.parameters()
        return {}
class MaskingSchedule:
    def __init__(self):
        self._last_masked: Set[int] = set()
    def get_masked_experts(self, alpha_experts: List[int], batch_id: int) -> Set[int]:
        if not alpha_experts:
            return set()
        candidates = [e for e in alpha_experts if e not in self._last_masked]
        if not candidates:
            candidates = alpha_experts
        mask_count = max(1, len(candidates) // 3)
        masked = set(candidates[:mask_count])
        self._last_masked = masked
        return masked
class DomainRegistry:
    """Which experts belong to which domain — earned, not declared.

    Every expert starts UNASSIGNED. Membership is granted by measured
    performance during the curriculum (general rounds -> per-domain rounds ->
    assign where it specialises), and can be revoked by migration. Until an
    expert is assigned it stays available to every domain, which is the correct
    cold start: with no evidence, any expert may serve any domain.

    Sizing, all derived from the pool and the domain count:

        min_pool(D)     = ceil(sqrt(E / D))     the floor every domain keeps
        new_domain_seed = sqrt(E) / 2           experts seeded into a new domain
        target(domain)  = max(min_pool, share of E by that domain's frequency)

    so frequent domains grow larger pools and rare ones cannot be starved below
    the floor."""

    def __init__(self, pool_size: Optional[int] = None):
        self.pool_size = int(pool_size or configs.EXPERT_POOL_SIZE)
        self.assignment: Dict[int, Optional[str]] = {i: None for i in range(self.pool_size)}
        self.domain_tokens: Dict[str, int] = {}

    # ── membership ─────────────────────────────────────────────────────────
    def unassigned(self) -> List[int]:
        return [e for e, d in self.assignment.items() if d is None]

    def members(self, domain: str) -> List[int]:
        return [e for e, d in self.assignment.items() if d == domain]

    def domains(self) -> List[str]:
        return sorted({d for d in self.assignment.values() if d})

    def assign(self, expert_id: int, domain: str) -> None:
        self.assignment[int(expert_id)] = domain

    def release(self, expert_id: int) -> None:
        """Back to unassigned — an expert that failed in its domain is available
        to every domain again rather than being stranded."""
        self.assignment[int(expert_id)] = None

    # ── sizing ─────────────────────────────────────────────────────────────
    def min_pool(self, n_domains: Optional[int] = None) -> int:
        """ceil(sqrt(E / D)) — the UPPER bracket here, unlike the NL selection
        rule, because this is a floor on how small a domain's pool may get."""
        d = max(1, int(n_domains if n_domains is not None else len(self.domains()) or 1))
        return max(1, math.ceil(math.sqrt(self.pool_size / d)))

    def new_domain_seed(self) -> int:
        """sqrt(E) / 2 experts seeded when a domain first appears."""
        return max(1, int(math.sqrt(self.pool_size) // 2))

    def record_domain_tokens(self, domain: str, tokens: int) -> None:
        self.domain_tokens[domain] = self.domain_tokens.get(domain, 0) + int(tokens)

    def target_size(self, domain: str) -> int:
        """Frequent domains earn bigger pools; the min_pool floor protects rare
        ones. Share is measured from observed tokens, not declared."""
        total = sum(self.domain_tokens.values())
        floor = self.min_pool()
        if total <= 0:
            return floor
        share = self.domain_tokens.get(domain, 0) / total
        return max(floor, int(round(self.pool_size * share)))

    def seed_domain(self, domain: str, candidates: List[int]) -> List[int]:
        """Seed a NEW domain with sqrt(E)/2 experts drawn from `candidates` —
        the caller supplies the middle tier (strong but underused) plus any
        migrants, so seeding never costs a domain its top performers."""
        need = self.new_domain_seed()
        picked = [e for e in candidates if self.assignment.get(e) != domain][:need]
        for e in picked:
            self.assign(e, domain)
        return picked


class Curriculum:
    """Per-expert training schedule that produces domain assignment.

    Every count is algebra over E (pool size) and D (domain count), read LIVE
    from configs/DOMAINS, so growing the pool or adding a domain reshapes the
    schedule with no code change:

        phase 1  general      E              rounds on the general domain
        phase 2  sampling     floor(sqrt(E)) rounds on EACH non-general domain
        phase 3  specialise   2E             rounds on the domain it scored best

        total per expert = 3E + sqrt(E)*(D-1)      (330 at E=100, D=4)

    A round is one training batch. Phase 2 is deliberately short: it is a probe
    to find where an expert belongs, not training — the real investment is the 2E
    rounds afterwards, spent only on the domain that earned them.

    Scores recorded here decide assignment, so they are only as meaningful as the
    quality signal feeding them; with an ungrounded r_i the machinery is correct
    and the verdict is not."""

    GENERAL = "general"

    def __init__(self, pool_size: Optional[int] = None, registry: Optional["DomainRegistry"] = None):
        self._pool_size = pool_size
        self.registry = registry
        self.rounds: Dict[int, Dict[str, int]] = {}          # expert -> domain -> rounds done
        self._failed: Dict[int, set] = {}                    # domains an expert could not hold
        self._in_transit: set = set()                        # passing THROUGH general, mid-migration
        self.migrations: List[Dict] = []                     # lifecycle: every move, before/after
        self._rng = random.Random(0)
        self.scores: Dict[int, Dict[str, List[float]]] = {}  # expert -> domain -> scores

    # ── derived counts (never stored, so configs stays the source of truth) ──
    @property
    def E(self) -> int:
        return int(self._pool_size or configs.EXPERT_POOL_SIZE)

    def general_rounds(self) -> int:
        return self.E

    def domain_rounds(self) -> int:
        return max(1, math.floor(math.sqrt(self.E)))

    def specialise_rounds(self) -> int:
        return 2 * self.E

    def sampling_domains(self) -> List[str]:
        return [d for d in DOMAINS if d != self.GENERAL]

    def total_rounds(self) -> int:
        return 3 * self.E + self.domain_rounds() * len(self.sampling_domains())

    # ── progress ────────────────────────────────────────────────────────────
    def done(self, expert_id: int, domain: str) -> int:
        return self.rounds.get(int(expert_id), {}).get(domain, 0)

    def record(self, expert_id: int, domain: str, score: Optional[float] = None) -> None:
        e = int(expert_id)
        self.rounds.setdefault(e, {})[domain] = self.done(e, domain) + 1
        if score is not None and math.isfinite(float(score)):
            self.scores.setdefault(e, {}).setdefault(domain, []).append(float(score))

    def phase(self, expert_id: int) -> str:
        """'general' -> 'sampling' -> 'specialise' -> 'done'."""
        e = int(expert_id)
        if self.done(e, self.GENERAL) < self.general_rounds():
            return "general"
        need = self.domain_rounds()
        if any(self.done(e, d) < need for d in self.sampling_domains()):
            return "sampling"
        assigned = self.registry.assignment.get(e) if self.registry else None
        if assigned is None:
            return "specialise"          # ready to be assigned
        extra = self.done(e, assigned) - (need if assigned in self.sampling_domains() else self.general_rounds())
        return "done" if extra >= self.specialise_rounds() else "specialise"

    def next_domain(self, expert_id: int) -> str:
        """Which domain this expert should train on right now."""
        e = int(expert_id)
        ph = self.phase(e)
        if ph == "general":
            return self.GENERAL
        if ph == "sampling":
            need = self.domain_rounds()
            pending = [d for d in self.sampling_domains() if self.done(e, d) < need]
            # Least-covered first, so sampling stays balanced if it is interrupted.
            return min(pending, key=lambda d: self.done(e, d))
        assigned = self.registry.assignment.get(e) if self.registry else None
        return assigned or self.best_domain(e) or self.GENERAL

    def domain_mean(self, expert_id: int, domain: str) -> Optional[float]:
        vals = self.scores.get(int(expert_id), {}).get(domain, [])
        return float(np.mean(vals)) if vals else None

    def best_domain(self, expert_id: int) -> Optional[str]:
        """Where this expert measured best. General is a legitimate destination —
        it was measured too, over E rounds — so an expert that is genuinely a
        generalist is not forced into a specialism it never earned."""
        means = {d: self.domain_mean(expert_id, d) for d in DOMAINS}
        means = {d: v for d, v in means.items() if v is not None}
        return max(means, key=means.get) if means else None

    # ── migration lifecycle statistics ─────────────────────────────────────
    def open_migration(self, expert_id: int, from_domain: str, to_domain: str) -> None:
        """Log a migration as it starts, with the score the expert held where it
        is leaving. Pairs with close_migration once the trial in the new place is
        served."""
        e = int(expert_id)
        before = self.domain_mean(e, from_domain)
        self.migrations.append({
            "expert_id": e, "from": from_domain, "to": to_domain,
            "before": before, "after": None,
        })

    def close_migration(self, expert_id: int, domain: str) -> Optional[float]:
        """Complete the most recent open migration for this expert, recording what
        it scored after settling. Returns the delta."""
        e = int(expert_id)
        after = self.domain_mean(e, domain)
        for rec in reversed(self.migrations):
            if rec["expert_id"] == e and rec["to"] == domain and rec["after"] is None:
                rec["after"] = after
                if rec["before"] is None or after is None:
                    return None
                return float(after - rec["before"])
        return None

    def migration_delta(self, from_domain: str, to_domain: str) -> Optional[float]:
        """The pool's AVERAGE change in score for experts that made this exact
        transition. This is the lifecycle signal: an expert's own sampling score
        says where it might fit, but the accumulated history of everyone who
        actually made the move says whether that kind of move pays. Returns None
        until at least one completed migration exists for the pair."""
        deltas = [r["after"] - r["before"] for r in self.migrations
                  if r["from"] == from_domain and r["to"] == to_domain
                  and r["before"] is not None and r["after"] is not None]
        return float(np.mean(deltas)) if deltas else None

    def lifecycle(self, expert_id: int) -> List[Dict]:
        """This expert's full migration history — where it has been, what it
        scored on arrival and departure, in order."""
        return [r for r in self.migrations if r["expert_id"] == int(expert_id)]

    # ── tenure & migration ─────────────────────────────────────────────────
    def trial_length(self) -> int:
        """NU(E) = ceil(sqrt(E)) — the successor approximation. An expert holds a
        domain place for this many rounds before its tenure is judged. Upper
        bracket here (unlike the NL selection rule) because it is a grace period:
        round it down and you evict on thinner evidence."""
        return max(1, math.ceil(math.sqrt(self.E)))

    def tenure_verdict(self, expert_id: int, domain: str, peers: List[int]) -> bool:
        """True = keep the place. An expert earns its domain by outranking AT
        LEAST ONE expert already ranked there. Beating nobody after a full trial
        means it has no claim — not that it is bad, just that this is not its
        domain. Judged only once the trial is served."""
        e = int(expert_id)
        if self.done(e, domain) < self.trial_length():
            return True                      # trial not served yet
        mine = self.domain_mean(e, domain)
        if mine is None:
            return True
        others = [self.domain_mean(p, domain) for p in peers if p != e]
        others = [s for s in others if s is not None]
        if not others:
            # SOLE OCCUPANT: no peer to displace, so judge against the pool's mean
            # on this domain — the same bar assign_pool uses. Returning True here
            # instead means an expert alone in a domain holds it forever however
            # badly it scores, and a domain seeded with one weak expert can never
            # correct itself.
            thr = self.domain_thresholds(list(self.scores)).get(domain)
            return thr is None or mine >= thr
        return any(mine > s for s in others)

    def next_migration_step(self, expert_id: int, current_domain: str) -> str:
        """One hop of the migration cycle: domain1 -> general -> domain2.

        Never a direct domain-to-domain jump. An expert that failed its tenure
        re-generalises first, then re-specialises — so it arrives at the next
        domain having been reset by general work rather than carrying the shape
        of the domain it just failed.

        The next specialism is chosen by CORRELATION with the first-phase
        sampling scores: the sampling round already measured this expert on every
        domain, so that record says where else it might fit. Ties break randomly.
        Domains it has already failed are excluded, so it cannot cycle back into
        a place it could not hold."""
        e = int(expert_id)
        failed = self._failed.setdefault(e, set())
        if current_domain != self.GENERAL:
            failed.add(current_domain)
            self._in_transit.add(e)          # mark: general is a waypoint, not a home
            return self.GENERAL              # step 1: always via general
        # step 2: general -> the best-correlated domain it has not failed
        cands = {d: self.domain_mean(e, d) for d in DOMAINS
                 if d != self.GENERAL and d not in failed}
        cands = {d: v for d, v in cands.items() if v is not None}
        self._in_transit.discard(e)          # arriving at domain2 ends the transit
        if not cands:
            failed.clear()                   # exhausted every domain: start over
            return self.GENERAL
        # Blend the expert's OWN sampling score for the candidate with the pool's
        # average outcome for this transition. The first says where this expert
        # might fit; the second says whether that kind of move has ever paid for
        # anyone. A domain the expert looks suited to, but which everyone who
        # moved there got worse in, is worth less than it appears.
        scored = {}
        for d, own in cands.items():
            hist = self.migration_delta(current_domain, d)
            scored[d] = own + (hist if hist is not None else 0.0)
        best = max(scored.values())
        top = [d for d, v in scored.items() if v >= best - 1e-9]
        return self._rng.choice(sorted(top))

    def review_tenure(self, registry: "DomainRegistry", expert_ids: Optional[List[int]] = None) -> Dict[int, str]:
        """Judge every assigned expert's tenure and move those that failed one hop
        along the cycle. Returns {expert_id: new_domain} for those that moved."""
        moved: Dict[int, str] = {}
        ids = expert_ids if expert_ids is not None else list(registry.assignment)
        for e in ids:
            dom = registry.assignment.get(e)
            if dom is None:
                continue
            # An expert passing THROUGH general is judged only on time served, not
            # on rank: alone in general it would outrank nobody, tenure_verdict
            # would return KEEP, and the migration would stall there forever
            # instead of continuing to domain2.
            in_transit = e in self._in_transit and dom == self.GENERAL
            if in_transit:
                if self.done(e, dom) < self.trial_length():
                    continue                 # still serving its general stint
            elif self.tenure_verdict(e, dom, registry.members(dom)):
                continue
            nxt = self.next_migration_step(e, dom)
            self.close_migration(e, dom)     # settle the record for the place being left
            self.open_migration(e, dom, nxt)
            registry.assign(e, nxt)
            self.rounds.setdefault(e, {})[nxt] = 0     # fresh trial in the new place
            moved[e] = nxt
        return moved

    def domain_thresholds(self, expert_ids: List[int]) -> Dict[str, float]:
        """Threshold per domain = the mean of every expert's score in THAT domain.

        The bar is pool-relative, which normalises for domain difficulty: if one
        domain yields systematically higher raw scores simply because its data is
        easier, its bar rises with it. Comparing an expert's own scores across
        domains (per-expert argmax) has no such correction and would pile the
        whole pool into whichever domain happens to score highest."""
        out: Dict[str, float] = {}
        for d in DOMAINS:
            vals = [self.domain_mean(e, d) for e in expert_ids]
            vals = [v for v in vals if v is not None]
            if vals:
                out[d] = float(np.mean(vals))
        return out

    def assign_pool(self, expert_ids: List[int],
                    target_sizes: Optional[Dict[str, int]] = None) -> Dict[str, List[int]]:
        """Assign every expert to one domain, pool-relative.

          1. threshold_d = mean score of all experts on domain d
          2. margin_d(e) = score_d(e) - threshold_d      (comparable ACROSS
             domains precisely because each threshold absorbs its own difficulty)
          3. each expert provisionally goes to its largest-margin domain
          4. domain over target -> drop the experts NEAREST the threshold
             (smallest margin); they return to the unassigned pool
          5. domain under target -> fill from the unassigned, taking those that
             fell BELOW the bar but nearest to it (highest score first)

        Steps 4-5 are why the threshold is a bar and not a hard filter: it ranks,
        and the pool's target size decides how far down the ranking to cut."""
        thresholds = self.domain_thresholds(expert_ids)
        if not thresholds:
            return {}
        targets = target_sizes or {}
        picked: Dict[str, List[int]] = {d: [] for d in thresholds}
        margin: Dict[int, Dict[str, float]] = {}
        for e in expert_ids:
            m = {}
            for d, thr in thresholds.items():
                s = self.domain_mean(e, d)
                if s is not None:
                    m[d] = s - thr
            if m:
                margin[e] = m
                picked[max(m, key=m.get)].append(e)
        unassigned: List[int] = [e for e in expert_ids if e not in margin]
        # 4. trim the over-full, releasing the marginal ones
        for d, members in picked.items():
            cap = targets.get(d)
            if cap is None or len(members) <= cap:
                continue
            members.sort(key=lambda e: margin[e][d], reverse=True)
            unassigned.extend(members[cap:])
            picked[d] = members[:cap]
        # 5. fill the under-full from those just below the bar
        for d in picked:
            cap = targets.get(d)
            if cap is None or len(picked[d]) >= cap:
                continue
            pool = [e for e in unassigned if self.domain_mean(e, d) is not None]
            pool.sort(key=lambda e: self.domain_mean(e, d), reverse=True)  # nearest the bar first
            take = pool[: cap - len(picked[d])]
            picked[d].extend(take)
            unassigned = [e for e in unassigned if e not in take]
        if self.registry is not None:
            for d, members in picked.items():
                for e in members:
                    self.registry.assign(e, d)
            for e in unassigned:
                self.registry.release(e)
        return picked

    def assign_if_ready(self, expert_id: int) -> Optional[str]:
        """Assign after sampling completes. Returns the domain, or None if the
        expert is still sampling / already assigned."""
        e = int(expert_id)
        if self.registry is None or self.registry.assignment.get(e) is not None:
            return None
        if self.phase(e) != "specialise":
            return None
        best = self.best_domain(e)
        if best:
            self.registry.assign(e, best)
        return best


class TripleKSelector:
    def __init__(self, convolution: ApexNadirConvolution):
        self.convolution = convolution
        self.k_d: Dict[str, List[int]] = {}
        self.k_pd: Dict[str, List[int]] = {}
        self.k_all: List[int] = list(range(configs.EXPERT_POOL_SIZE))
        self.alpha_experts: Dict[str, List[int]] = {}
        self.beta_experts: Dict[str, List[int]] = {}
        self._rng = random.Random(42)
        self._auto_seed()
    def _auto_seed(self):
        if hasattr(configs, "EXPERT_GROUPS") and configs.EXPERT_GROUPS:
            self.seed_from_calibration(configs.EXPERT_GROUPS)
    def seed_from_calibration(self, domain_assignments: Dict[str, List[int]]):
        self.k_d = dict(domain_assignments)
        for domain, experts in domain_assignments.items():
            n_alpha = max(1, len(experts) // 3)
            self.alpha_experts[domain] = experts[:n_alpha]
            self.beta_experts[domain] = experts[n_alpha:]
            self.k_pd[domain] = list(experts)
    def select_experts(self, gate_output: GateOutput, session_tracker, masking_schedule: MaskingSchedule, batch_id: int = 0, loaded_experts=None) -> List[SelectedExpert]:
        domain = "general"
        logits = gate_output.domain_logits
        if logits is not None and logits.shape[0] >= len(DOMAINS):
            domain = DOMAINS[int(mx.argmax(logits[:len(DOMAINS)]).item())]
        # k=0 means "Timeline A" (no experts) — but that path never calls
        # select_experts. Whenever we ARE selecting (Timeline B / training), we need
        # at least one expert; otherwise selected[:0] == [] and the caller skips the
        # batch entirely (no gate/expert gradient). A confident gate emits k=0, which
        # was silently zeroing ~75% of training batches.
        k = max(1, gate_output.k_per_token)
        domain_pool = self.k_d.get(domain, self.k_all)
        masked = masking_schedule.get_masked_experts(self.alpha_experts.get(domain, []), batch_id)
        # The gate's learned routing preference (one softmax pull). Blended into the
        # ranking below: apex-nadir distance-to-peak keeps things grounded, the head
        # tilts selection toward experts it has learned are efficient/fresh.
        route_pref = None
        if gate_output.route_logits is not None:
            route_pref = mx.softmax(gate_output.route_logits).tolist()
        candidates = []
        for eid in domain_pool:
            if eid in masked:
                continue
            current_alloc = session_tracker.get_current_allocation(eid)
            dist = self.convolution.get_distance_to_peak(eid, current_alloc)
            if loaded_experts and eid in loaded_experts:
                dist *= 0.1  # Huge discount for RAM-resident experts to avoid SSD latency
            if route_pref is not None and eid < len(route_pref):
                dist -= configs.ROUTE_BIAS_W * route_pref[eid]   # learned preference lowers rank-distance
            jitter = self._rng.random() * 1e-6
            is_alpha = eid in self.alpha_experts.get(domain, [])
            candidates.append(SelectedExpert(expert_id=eid, distance_to_peak=dist + jitter, domain=domain, is_alpha=is_alpha))
        self._rng.shuffle(candidates)
        candidates.sort(key=lambda e: e.distance_to_peak)
        selected = candidates[:k]
        if len(selected) < k:
            for eid in self.k_all:
                if len(selected) >= k:
                    break
                if any(s.expert_id == eid for s in selected):
                    continue
                if eid in masked:
                    continue
                current_alloc = session_tracker.get_current_allocation(eid)
                dist = self.convolution.get_distance_to_peak(eid, current_alloc)
                selected.append(SelectedExpert(expert_id=eid, distance_to_peak=dist, domain=domain, is_alpha=False))
        return selected[:k]
