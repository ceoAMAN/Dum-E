from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import configs
@dataclass
class ExpertCurves:
    apex_coeffs: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, -0.001]))
    nadir_coeffs: np.ndarray = field(default_factory=lambda: np.array([float(configs.FRAGMENT_MIN), 0.0]))
    latency_coeffs: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.001]))
    apex_ceiling: float = 256.0
    nadir_floor: float = float(configs.FRAGMENT_MIN)
    r_out_cached: Optional[float] = None
class ApexNadirConvolution:
    def __init__(self, calibration_path: str, latency_store_path: str):
        self.calibration_path = calibration_path
        self.latency_store_path = latency_store_path
        self.expert_curves: Dict[int, ExpertCurves] = {}
        self._init_default_curves()
    def _init_default_curves(self):
        for i in range(configs.EXPERT_POOL_SIZE):
            self.expert_curves[i] = ExpertCurves()
    def _eval_poly(self, coeffs: np.ndarray, x: float) -> float:
        return float(sum(c * (x ** i) for i, c in enumerate(coeffs)))
    def _eval_apex(self, expert_id: int, token_count: float) -> float:
        curves = self.expert_curves[expert_id]
        return max(0.0, self._eval_poly(curves.apex_coeffs, token_count))
    def _eval_nadir(self, expert_id: int, token_count: float) -> float:
        curves = self.expert_curves[expert_id]
        return max(0.0, self._eval_poly(curves.nadir_coeffs, token_count))
    def _eval_latency(self, expert_id: int, token_count: float) -> float:
        curves = self.expert_curves[expert_id]
        return max(1e-6, self._eval_poly(curves.latency_coeffs, token_count))
    def compute_r_out(self, expert_id: int) -> float:
        curves = self.expert_curves[expert_id]
        if curves.r_out_cached is not None:
            return curves.r_out_cached
        floor = max(configs.FRAGMENT_MIN, int(curves.nadir_floor))
        ceiling = max(floor + 1, int(curves.apex_ceiling))
        best_t = floor
        best_ratio = -1.0
        step = max(1, (ceiling - floor) // 100)
        for t in range(floor, ceiling + 1, step):
            s_c = self._eval_apex(expert_id, float(t))
            c_e = self._eval_latency(expert_id, float(t))
            ratio = s_c / c_e
            if ratio > best_ratio:
                best_ratio = ratio
                best_t = t
        search_lo = max(floor, best_t - step)
        search_hi = min(ceiling, best_t + step)
        for t in range(search_lo, search_hi + 1):
            s_c = self._eval_apex(expert_id, float(t))
            c_e = self._eval_latency(expert_id, float(t))
            ratio = s_c / c_e
            if ratio > best_ratio:
                best_ratio = ratio
                best_t = t
        r_out = float(max(configs.FRAGMENT_MIN, best_t))
        curves.r_out_cached = r_out
        return r_out
    def compute_r_out_mean(self, expert_ids: List[int]) -> float:
        if not expert_ids:
            return float(configs.MAX_SEQ_LEN) / configs.K_DEFAULT
        return float(np.mean([self.compute_r_out(eid) for eid in expert_ids]))
    def update_latency(self, expert_id: int, token_count: int, wall_time: float):
        curves = self.expert_curves[expert_id]
        measured_rate = wall_time / max(token_count, 1)
        old_slope = curves.latency_coeffs[1] if len(curves.latency_coeffs) > 1 else 0.001
        new_slope = configs.EMA_DECAY * old_slope + (1.0 - configs.EMA_DECAY) * measured_rate
        curves.latency_coeffs = np.array([0.0, new_slope])
        curves.r_out_cached = None
    def check_monopoly_ceiling(self, expert_id: int, current_allocation: int) -> bool:
        curves = self.expert_curves[expert_id]
        return current_allocation > curves.apex_ceiling * configs.MONOPOLY_THRESHOLD
    def check_nadir_floor(self, expert_id: int, fragment_size: int) -> bool:
        curves = self.expert_curves[expert_id]
        return fragment_size < curves.nadir_floor
    def get_distance_to_peak(self, expert_id: int, current_allocation: int) -> float:
        r_out = self.compute_r_out(expert_id)
        if r_out < 1e-6:
            return float('inf')
        return abs(current_allocation - r_out) / r_out
    def fit_curves_from_calibration(self, expert_id: int, calibration_data: dict):
        curves = self.expert_curves[expert_id]
        if "token_counts" in calibration_data and "quality_scores" in calibration_data:
            tc = np.array(calibration_data["token_counts"], dtype=np.float64)
            qs = np.array(calibration_data["quality_scores"], dtype=np.float64)
            if len(tc) >= 3:
                X = np.vstack([np.ones_like(tc), tc, tc ** 2]).T
                coeffs, _, _, _ = np.linalg.lstsq(X, qs, rcond=None)
                curves.apex_coeffs = coeffs
                if coeffs[2] < 0:
                    curves.apex_ceiling = max(float(configs.FRAGMENT_MIN), float(-coeffs[1] / (2 * coeffs[2])))
                else:
                    curves.apex_ceiling = float(tc.max())
        if "gradient_coherence" in calibration_data and "token_counts" in calibration_data:
            gc = np.array(calibration_data["gradient_coherence"], dtype=np.float64)
            tc = np.array(calibration_data["token_counts"], dtype=np.float64)
            valid = tc[gc > 0.1]
            if len(valid) > 0:
                curves.nadir_floor = max(float(configs.FRAGMENT_MIN), float(valid.min()))
            curves.nadir_coeffs = np.array([curves.nadir_floor, 0.0])
        if "wall_times" in calibration_data and "token_counts" in calibration_data:
            wt = np.array(calibration_data["wall_times"], dtype=np.float64)
            tc = np.array(calibration_data["token_counts"], dtype=np.float64)
            if len(tc) > 0:
                curves.latency_coeffs = np.array([0.0, float(np.mean(wt / np.maximum(tc, 1)))])
        curves.r_out_cached = None
    def reset_r_t_curve(self, expert_id: int):
        curves = self.expert_curves[expert_id]
        curves.latency_coeffs = np.array([0.0, 0.001])
        curves.r_out_cached = None
    def save(self):
        from pathlib import Path
        Path(self.calibration_path).parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for eid, curves in self.expert_curves.items():
            data[f"apex_{eid}"] = curves.apex_coeffs
            data[f"nadir_{eid}"] = curves.nadir_coeffs
            data[f"latency_{eid}"] = curves.latency_coeffs
            data[f"meta_{eid}"] = np.array([curves.apex_ceiling, curves.nadir_floor])
        np.savez(self.calibration_path, **data)
    def save_latency_store(self):
        from pathlib import Path
        Path(self.latency_store_path).parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for eid, curves in self.expert_curves.items():
            data[f"latency_{eid}"] = curves.latency_coeffs
        np.savez(self.latency_store_path, **data)
    def load(self):
        try:
            data = np.load(self.calibration_path, allow_pickle=False)
            for eid in range(configs.EXPERT_POOL_SIZE):
                curves = self.expert_curves[eid]
                if f"apex_{eid}" in data:
                    curves.apex_coeffs = data[f"apex_{eid}"]
                if f"nadir_{eid}" in data:
                    curves.nadir_coeffs = data[f"nadir_{eid}"]
                if f"latency_{eid}" in data:
                    curves.latency_coeffs = data[f"latency_{eid}"]
                if f"meta_{eid}" in data:
                    meta = data[f"meta_{eid}"]
                    curves.apex_ceiling = float(meta[0])
                    curves.nadir_floor = float(meta[1])
                curves.r_out_cached = None
        except FileNotFoundError:
            pass
        try:
            data = np.load(self.latency_store_path, allow_pickle=False)
            for eid in range(configs.EXPERT_POOL_SIZE):
                if f"latency_{eid}" in data:
                    self.expert_curves[eid].latency_coeffs = data[f"latency_{eid}"]
                    self.expert_curves[eid].r_out_cached = None
        except FileNotFoundError:
            pass
