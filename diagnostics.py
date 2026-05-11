from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx

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
                    recent_avg_time > 2.5
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
                if recent_time > 2.5 or recent_thermal > configs.THERMAL_THROTTLE_TEMP * 0.82:
                    x_pred = min(x_pred, max(float(configs.X_MIN), float(recent[-1].x_used - 1)))

            return max(configs.X_MIN, min(configs.X_MAX, round(x_pred)))
        except Exception:
            return self.x_next
