from __future__ import annotations

import os
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import numpy as np

import configs


@dataclass
class SystemSnapshot:
    batch_index: int
    tokens_processed: int
    time_in_bound: float
    thermal_state: float
    ram_headroom_mb: float
    ssd_read_rate_mb: float
    k_used: int
    x_used: int


class Diagnostics:
    def __init__(self):
        self.history: List[SystemSnapshot] = []
        self.x_next: int = configs.X_MAX
        self._batch_index: int = 0
        self._thermal_estimate: float = 58.0
        self._thermal_is_exact: bool = False
        # ── live memory governor (set via set_memory_baseline) ──
        self._mem_base_mb: float = 0.0          # resident central+gate, measured
        self._mem_usable_mb: float = 0.0        # ceiling the MLX peak may reach, measured
        self._mem_samples: deque = deque(maxlen=64)   # (x_used, peak_overhead_mb) for the OLS fit
        self._mem_shared_mb: float = 0.0        # fitted shared spike (7B forward), intercept a
        self._mem_per_expert_mb: float = float(configs.EXPERT_RAM_MB)  # fitted per-expert cost, slope b

    def set_memory_baseline(self, base_mb: float, usable_mb: float) -> None:
        """Record measured resident base (central+gate) and the usable peak ceiling
        (base + currently-available RAM). The per-expert / shared-spike split is then
        learned live by OLS from (x_used, peak) samples."""
        self._mem_base_mb = base_mb
        self._mem_usable_mb = usable_mb
        self._mem_samples.clear()
        self._mem_shared_mb = 0.0
        self._mem_per_expert_mb = float(configs.EXPERT_RAM_MB)

    def observe_memory(self, x_used: int) -> None:
        """Record this batch's real peak overhead (peak - base) against the expert
        count, then reset the high-water mark. Feeds the OLS cost model below."""
        from splitter import get_peak_memory_mb, reset_peak_memory
        peak = get_peak_memory_mb()
        reset_peak_memory()
        if x_used > 0 and peak > self._mem_base_mb:
            self._mem_samples.append((float(x_used), peak - self._mem_base_mb))
        self._fit_cost_model()

    def _fit_cost_model(self) -> None:
        """OLS fit of peak_overhead ≈ shared + per_expert · x. The intercept is the
        7B-forward spike shared across experts; the slope is the TRUE marginal cost
        of one more expert. With <2 distinct x values, fall back to a conservative
        single-point estimate (all overhead charged per-expert)."""
        if not self._mem_samples:
            return
        xs = [s[0] for s in self._mem_samples]
        # An expert can never cost less than its resident weights, so the slope is
        # floored at the measured weight RAM — this kills the degenerate near-zero/
        # negative slopes that noisy real samples produce (which would blow the
        # ceiling up to nonsense).
        floor = float(configs.EXPERT_RAM_MB)
        if len(self._mem_samples) >= 3 and len(set(xs)) >= 2:
            X = np.array([[1.0, x] for x, _ in self._mem_samples], dtype=np.float64)
            y = np.array([o for _, o in self._mem_samples], dtype=np.float64)
            (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
            self._mem_per_expert_mb = max(floor, float(b))
            # Keep the implied shared spike consistent with the floored slope so the
            # ceiling stays grounded in the actual observed peaks.
            mean_x = float(np.mean(xs))
            mean_o = float(np.mean([o for _, o in self._mem_samples]))
            self._mem_shared_mb = max(0.0, mean_o - self._mem_per_expert_mb * mean_x)
        else:
            x, o = self._mem_samples[-1]
            self._mem_shared_mb = 0.0
            self._mem_per_expert_mb = max(floor, o / max(x, 1.0))

    def memory_ceiling(self) -> int:
        """Max experts that fit under the measured usable-RAM ceiling, given the
        fitted shared spike + per-expert cost."""
        if self._mem_usable_mb <= 0.0:
            return configs.X_MAX
        room = max(0.0, self._mem_usable_mb - self._mem_base_mb - self._mem_shared_mb)
        fits = int(room / max(self._mem_per_expert_mb, 1.0))
        return max(configs.X_MIN, min(configs.X_MAX, fits))   # soft ceiling X_MAX

    def _time_pressure(self, recent_time: float) -> bool:
        """True if recent batch time is well above the run's fastest (unthrottled)
        batch — a RELATIVE, self-calibrating slowdown signal (replaces a hardcoded
        absolute-seconds wall that mis-fired on every 7B+generation batch)."""
        if len(self.history) < 3:
            return False
        times = [s.time_in_bound for s in self.history if s.time_in_bound > 0.0]
        if not times:
            return False
        return recent_time > 2.0 * min(times)

    def _recent_tps(self) -> Optional[float]:
        """Tokens/sec of the most recent batch (per-batch delta / its wall time)."""
        if not self.history:
            return None
        last = self.history[-1]
        prev = self.history[-2].tokens_processed if len(self.history) >= 2 else 0
        delta = last.tokens_processed - prev
        return delta / last.time_in_bound if last.time_in_bound > 0 and delta > 0 else None

    def recommended_x(self, x_used_last: int) -> int:
        """Experts to run next batch — derived live, no hard cap. Three governors:
          - MEMORY: never exceed the OLS-fitted ceiling (OOM safety, primary).
          - TEMPERATURE: back off as the SoC nears its throttle point.
          - PROCESSING: back off if throughput COLLAPSES relative to the best seen
            (a self-calibrating signal — real degradation, not a hardcoded latency).
        Healthy → cautious +1 ramp, re-measuring cost at each step so it can't
        overshoot into an OOM."""
        mem_cap = self.memory_ceiling()
        if mem_cap < x_used_last:                       # memory tightened — drop to fit now
            return max(configs.X_MIN, mem_cap)
        thermal_hot = bool(self.history) and self.history[-1].thermal_state > configs.THERMAL_THROTTLE_TEMP * configs.THERMAL_BACKOFF_FRAC
        tps = self._recent_tps()
        if tps is not None:
            self._best_tps = max(getattr(self, "_best_tps", 0.0), tps)
        throughput_collapse = (
            tps is not None and getattr(self, "_best_tps", 0.0) > 0.0
            and tps < configs.THROUGHPUT_COLLAPSE_FRAC * self._best_tps
        )
        if thermal_hot or throughput_collapse:
            return max(configs.X_MIN, min(mem_cap, x_used_last - 1))   # back off under real pressure
        return max(configs.X_MIN, min(mem_cap, x_used_last + 1))       # cautious ramp up (mem_cap <= X_MAX)

    def update(self, tokens_processed: int, time_in_bound: float, x_used: int, k_used: int) -> int:
        ram_headroom_mb = self._read_ram()
        ssd_read_rate_mb = self._read_ssd_rate()
        snap = SystemSnapshot(
            batch_index=self._batch_index,
            tokens_processed=tokens_processed,
            time_in_bound=time_in_bound,
            thermal_state=self._read_thermal(time_in_bound, x_used, k_used, ram_headroom_mb, ssd_read_rate_mb),
            ram_headroom_mb=ram_headroom_mb,
            ssd_read_rate_mb=ssd_read_rate_mb,
            k_used=k_used,
            x_used=x_used,
        )
        self.history.append(snap)
        self._batch_index += 1
        self.x_next = self._regress()
        return self.x_next

    def validate_thermal_regression(self) -> dict:
        if not self.history:
            return {
                "history_len": 0,
                "x_next": self.x_next,
                "bounded": configs.X_MIN <= self.x_next <= configs.X_MAX,
                "thermal_guard_active": False,
                "recent_avg_thermal": 0.0,
                "recent_avg_time_in_bound": 0.0,
                "thermal_source": "none",
            }
        latest = self.history[-1]
        recent = self.history[-3:]
        recent_avg_thermal = sum(snap.thermal_state for snap in recent) / len(recent)
        recent_avg_time = sum(snap.time_in_bound for snap in recent) / len(recent)
        thermal_guard_active = (
            latest.thermal_state > configs.THERMAL_THROTTLE_TEMP * 0.9
            or (
                not self._thermal_is_exact
                and len(self.history) >= 3
                and (
                    self._time_pressure(recent_avg_time)
                    or recent_avg_thermal > configs.THERMAL_THROTTLE_TEMP * 0.82
                )
            )
        )
        return {
            "history_len": len(self.history),
            "latest_batch_index": latest.batch_index,
            "x_used": latest.x_used,
            "x_next": self.x_next,
            "bounded": configs.X_MIN <= self.x_next <= configs.X_MAX,
            "thermal_guard_active": thermal_guard_active,
            "thermal": latest.thermal_state,
            "recent_avg_thermal": recent_avg_thermal,
            "recent_avg_time_in_bound": recent_avg_time,
            "ram_mb": latest.ram_headroom_mb,
            "ssd_read_rate_mb": latest.ssd_read_rate_mb,
            "thermal_source": "direct" if self._thermal_is_exact else "proxy_or_estimate",
        }

    def _read_thermal(
        self,
        time_in_bound: float,
        x_used: int,
        k_used: int,
        ram_headroom_mb: float,
        ssd_read_rate_mb: float,
    ) -> float:
        direct = self._read_powermetrics_thermal()
        if direct is not None:
            self._thermal_estimate = direct
            self._thermal_is_exact = True
            return direct
        proxy = self._read_pmset_thermal_proxy()
        if proxy is not None:
            self._thermal_estimate = proxy
            self._thermal_is_exact = False
            return proxy
        self._thermal_is_exact = False
        return self._estimate_thermal(time_in_bound, x_used, k_used, ram_headroom_mb, ssd_read_rate_mb)

    def _read_powermetrics_thermal(self) -> Optional[float]:
        commands = []
        if os.geteuid() == 0:
            commands.append(["powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"])
        else:
            commands.append(["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"])
            commands.append(["powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"])
        for command in commands:
            try:
                out = subprocess.check_output(
                    command,
                    text=True,
                    timeout=2,
                    stderr=subprocess.DEVNULL,
                )
                for line in out.splitlines():
                    if "CPU die temperature" in line:
                        return float(line.split(":")[1].strip().split()[0])
            except Exception:
                continue
        return None

    def _read_pmset_thermal_proxy(self) -> Optional[float]:
        # thermlog is deliberately excluded — it blocks for the full timeout
        # on unthrottled systems waiting for events that never arrive.
        try:
            out = subprocess.check_output(
                ["pmset", "-g", "therm"],
                text=True,
                timeout=2,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return None
        sched_limits = [int(value) for value in re.findall(r"(?:Scheduler|Speed)_Limit[^0-9]*(\d+)", out, flags=re.IGNORECASE)]
        warning_levels = [int(value) for value in re.findall(r"warning level[^0-9]*(\d+)", out, flags=re.IGNORECASE)]
        if sched_limits:
            limit = min(sched_limits)
            warning = max(warning_levels) if warning_levels else 0
            return max(self._thermal_estimate, min(95.0, 55.0 + (100 - limit) * 0.55 + warning * 4.0))
        if warning_levels:
            warning = max(warning_levels)
            return max(self._thermal_estimate, min(95.0, 62.0 + warning * 6.0))
        return None

    def _estimate_thermal(
        self,
        time_in_bound: float,
        x_used: int,
        k_used: int,
        ram_headroom_mb: float,
        ssd_read_rate_mb: float,
    ) -> float:
        x_scale = min(1.0, x_used / max(float(configs.X_MAX), 1.0))
        k_scale = min(1.0, k_used / max(float(configs.K_MAX), 1.0))
        time_scale = min(1.0, time_in_bound / 4.0)
        ram_capacity = max(float(configs.EXPERT_RAM_MB * configs.X_MAX), 1.0)
        ram_scale = 1.0 - min(1.0, ram_headroom_mb / ram_capacity)
        io_scale = min(1.0, ssd_read_rate_mb / 2048.0)
        target = 49.0 + 34.0 * (0.3 * x_scale + 0.2 * k_scale + 0.28 * time_scale + 0.17 * ram_scale + 0.05 * io_scale)
        previous = self.history[-1].thermal_state if self.history else self._thermal_estimate
        alpha = 0.3 if target >= previous else 0.12
        estimate = previous + (target - previous) * alpha
        self._thermal_estimate = max(42.0, min(92.0, estimate))
        return self._thermal_estimate

    def _read_ram(self) -> float:
        try:
            out = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
            page_size = 16384
            free = inactive = 0
            for line in out.splitlines():
                if line.startswith("Pages free:"):
                    free = int(line.split(":")[1].strip().rstrip("."))
                elif line.startswith("Pages inactive:"):
                    inactive = int(line.split(":")[1].strip().rstrip("."))
            return (free + inactive) * page_size / (1024 * 1024)
        except Exception:
            return float(configs.EXPERT_RAM_MB * 3)

    def _read_ssd_rate(self) -> float:
        try:
            out = subprocess.check_output(["iostat", "-d", "-K", "disk0"], text=True, stderr=subprocess.DEVNULL)
            lines = [
                line
                for line in out.splitlines()
                if line.strip() and not line.lstrip().startswith("disk")
            ]
            if lines:
                parts = lines[-1].split()
                if len(parts) >= 3:
                    return float(parts[2])
        except Exception:
            pass
        return 0.0

    def _regress(self) -> int:
        if len(self.history) < 3:
            return self.x_next

        x_mat = mx.array(
            [
                [
                    snap.thermal_state,
                    snap.ram_headroom_mb,
                    snap.ssd_read_rate_mb,
                    snap.time_in_bound,
                    float(snap.k_used),
                    1.0,
                ]
                for snap in self.history
            ],
            dtype=mx.float32,
        )
        y_vec = mx.array([float(snap.x_used) for snap in self.history], dtype=mx.float32)

        xtx = mx.matmul(x_mat.T, x_mat)
        xty = mx.matmul(x_mat.T, y_vec)

        try:
            ridge = mx.eye(xtx.shape[0]) * 1e-4
            beta = mx.matmul(mx.linalg.inv(xtx + ridge), xty)
            mx.eval(beta)

            latest = self.history[-1]
            current = mx.array(
                [
                    [
                        latest.thermal_state,
                        latest.ram_headroom_mb,
                        latest.ssd_read_rate_mb,
                        latest.time_in_bound,
                        float(latest.k_used),
                        1.0,
                    ]
                ],
                dtype=mx.float32,
            )
            x_pred = mx.matmul(current, beta).item()

            if latest.thermal_state > configs.THERMAL_THROTTLE_TEMP * 0.9:
                x_pred = min(x_pred, 2.0)

            if not self._thermal_is_exact and len(self.history) >= 3:
                recent = self.history[-3:]
                recent_time = sum(snap.time_in_bound for snap in recent) / len(recent)
                recent_thermal = sum(snap.thermal_state for snap in recent) / len(recent)
                if self._time_pressure(recent_time) or recent_thermal > configs.THERMAL_THROTTLE_TEMP * 0.82:
                    x_pred = min(x_pred, max(float(configs.X_MIN), float(recent[-1].x_used - 1)))

            return max(configs.X_MIN, min(configs.X_MAX, round(x_pred)))
        except Exception:
            return self.x_next
