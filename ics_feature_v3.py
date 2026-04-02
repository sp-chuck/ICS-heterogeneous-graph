import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureV3Config:
    # Period mining parameters
    w0: int = 16
    w_max: int = 256
    delta: int = 4
    delta_w: int = 2
    entropy_bins: int = 16
    constant_threshold: float = 1e-6
    large_period_value: float = 1e6
    convexity_drop_ratio: float = 0.02
    min_period_autocorr: float = 0.3

    # Real metadata vocabularies built from list sensors (stable within one run)
    device_types: Tuple[str, ...] = ("UNKNOWN",)
    processes: Tuple[str, ...] = ("PUNK",)
    sub_processes: Tuple[str, ...] = ("SPUNK",)
    controller_count: int = 6
    # device_name -> (device_type, process, sub_process)
    metadata_map: Optional[Dict[str, Tuple[str, str, str]]] = None

    def full_feature_names(self) -> List[str]:
        names: List[str] = []
        names.extend([f"general_device_type_{x.lower()}" for x in self.device_types])
        names.extend([f"general_process_{x.lower()}" for x in self.processes])
        names.extend([f"general_sub_process_{x.lower()}" for x in self.sub_processes])
        names.extend([f"control_plc{i}" for i in range(1, self.controller_count + 1)])
        names.extend(
            [
                "measurement_periodicity_periodic",
                "measurement_periodicity_constant",
                "measurement_periodicity_non_periodic",
                "measurement_period_value",
                "measurement_period_max",
                "measurement_period_min",
                "measurement_period_entropy_mean",
                "measurement_period_entropy_var",
            ]
        )
        return names


def _split_prefix_number(device_name: str) -> Tuple[str, str]:
    s = (device_name or "").strip().upper()
    m = re.match(r"^([A-Z]+)(\d+)", s)
    if not m:
        return "UNKNOWN", ""
    return m.group(1), m.group(2)


def _infer_meta_from_device_name(device_name: str) -> Tuple[str, str, str]:
    prefix, number = _split_prefix_number(device_name)
    device_type = prefix if prefix else "UNKNOWN"

    if number:
        process = f"P{number[0]}"
        if len(number) >= 2:
            sub_process = f"SP{number[:2]}"
        else:
            sub_process = "SPUNK"
    else:
        process = "PUNK"
        sub_process = "SPUNK"

    return device_type, process, sub_process


def build_real_metadata_config(
    sensors: List[str],
    base: Optional[FeatureV3Config] = None,
) -> FeatureV3Config:
    cfg = base or FeatureV3Config()

    metadata_map: Dict[str, Tuple[str, str, str]] = {}
    device_types = set()
    processes = set()
    sub_processes = set()

    for s in sensors:
        meta = _infer_meta_from_device_name(s)
        metadata_map[s] = meta
        device_types.add(meta[0])
        processes.add(meta[1])
        sub_processes.add(meta[2])

    if "UNKNOWN" not in device_types:
        device_types.add("UNKNOWN")
    if "PUNK" not in processes:
        processes.add("PUNK")
    if "SPUNK" not in sub_processes:
        sub_processes.add("SPUNK")

    cfg.device_types = tuple(sorted(device_types))
    cfg.processes = tuple(sorted(processes))
    cfg.sub_processes = tuple(sorted(sub_processes))
    cfg.metadata_map = metadata_map
    return cfg


def one_hot(value: str, vocab: Sequence[str]) -> np.ndarray:
    out = np.zeros(len(vocab), dtype=np.float32)
    try:
        idx = list(vocab).index(value)
    except ValueError:
        idx = len(vocab) - 1
    out[idx] = 1.0
    return out


def extract_general_process_features(
    device_name: str,
    plc_name: Optional[str],
    config: FeatureV3Config,
) -> np.ndarray:
    _ = plc_name
    if config.metadata_map and device_name in config.metadata_map:
        device_type, process, sub_process = config.metadata_map[device_name]
    else:
        device_type, process, sub_process = _infer_meta_from_device_name(device_name)

    return np.concatenate(
        [
            one_hot(device_type, config.device_types),
            one_hot(process, config.processes),
            one_hot(sub_process, config.sub_processes),
        ],
        axis=0,
    ).astype(np.float32)


def extract_control_features(plc_name: Optional[str], config: FeatureV3Config) -> np.ndarray:
    out = np.zeros(config.controller_count, dtype=np.float32)
    if plc_name and plc_name.upper().startswith("PLC"):
        idx_text = plc_name[3:]
        if idx_text.isdigit():
            idx = int(idx_text) - 1
            if 0 <= idx < config.controller_count:
                out[idx] = 1.0
    return out


def window_entropy(window: np.ndarray, bins: int = 16) -> float:
    if window.size == 0:
        return 0.0
    if np.allclose(window, window[0]):
        return 0.0

    hist, _ = np.histogram(window, bins=bins, density=False)
    total = float(np.sum(hist))
    if total <= 0:
        return 0.0
    prob = hist.astype(np.float64) / total
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log(prob + 1e-12)))


def build_windows(signal: np.ndarray, length: int, delta_w: int) -> List[np.ndarray]:
    if length <= 0 or delta_w <= 0 or signal.size < length:
        return []
    return [signal[s : s + length] for s in range(0, signal.size - length + 1, delta_w)]


def find_local_minimum_with_convexity_check(
    scores: List[Tuple[int, float]],
    drop_ratio: float = 0.02,
) -> Optional[int]:
    if len(scores) < 3:
        return None

    ells = np.array([x[0] for x in scores], dtype=np.int64)
    vals = np.array([x[1] for x in scores], dtype=np.float64)

    for i in range(1, len(vals) - 1):
        left = vals[i - 1]
        cur = vals[i]
        right = vals[i + 1]

        is_local_min = (cur <= left) and (cur <= right)
        if not is_local_min:
            continue

        drop = left - cur
        rise = right - cur
        scale = max(abs(cur), 1e-9)
        if drop / scale < drop_ratio:
            continue
        if rise / scale < drop_ratio:
            continue

        return int(ells[i])

    return None


def sequence_period_mining(
    signal: np.ndarray,
    config: FeatureV3Config,
) -> Tuple[Optional[int], List[Tuple[int, float]]]:
    n = signal.size
    if n <= 2:
        return None, []

    w0 = max(2, int(config.w0))
    w_max = min(int(config.w_max), max(2, n // 2))
    delta = max(1, int(config.delta))
    delta_w = max(1, int(config.delta_w))

    if w0 > w_max:
        return None, []

    scores: List[Tuple[int, float]] = []
    for ell in range(w0, w_max + 1, delta):
        windows = build_windows(signal, ell, delta_w)
        if not windows:
            continue
        entropies = [window_entropy(w, bins=config.entropy_bins) for w in windows]
        var_entropy = float(np.var(np.asarray(entropies, dtype=np.float64)))
        scores.append((ell, var_entropy))

    t = find_local_minimum_with_convexity_check(scores, drop_ratio=config.convexity_drop_ratio)
    return t, scores


def extract_measurement_features(signal: np.ndarray, config: FeatureV3Config) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return np.zeros(8, dtype=np.float32)

    if float(np.std(x)) < float(config.constant_threshold):
        c = float(np.mean(x))
        return np.array([0, 1, 0, 0, c, c, 0, 0], dtype=np.float32)

    t, _ = sequence_period_mining(x, config)
    if t is None or t <= 1 or t >= x.size:
        global_entropy = window_entropy(x, bins=config.entropy_bins)
        return np.array(
            [0, 0, 1, config.large_period_value, float(np.max(x)), float(np.min(x)), global_entropy, 0.0],
            dtype=np.float32,
        )

    n = x.size
    lag_candidates = {int(t)}
    if t - 1 > 1:
        lag_candidates.add(int(t - 1))
    if t + 1 < n:
        lag_candidates.add(int(t + 1))
    for k in (2, 3):
        lk = int(t * k)
        if 1 < lk < n:
            lag_candidates.add(lk)

    best_abs_corr = 0.0
    for lag in sorted(lag_candidates):
        x1 = x[:-lag]
        x2 = x[lag:]
        if x1.size < 2 or x2.size < 2:
            continue
        c = float(np.corrcoef(x1, x2)[0, 1])
        if np.isnan(c):
            continue
        best_abs_corr = max(best_abs_corr, abs(c))

    if best_abs_corr < float(config.min_period_autocorr):
        global_entropy = window_entropy(x, bins=config.entropy_bins)
        return np.array(
            [0, 0, 1, config.large_period_value, float(np.max(x)), float(np.min(x)), global_entropy, 0.0],
            dtype=np.float32,
        )

    windows = build_windows(x, int(t), max(1, int(config.delta_w)))
    if not windows:
        global_entropy = window_entropy(x, bins=config.entropy_bins)
        return np.array(
            [0, 0, 1, config.large_period_value, float(np.max(x)), float(np.min(x)), global_entropy, 0.0],
            dtype=np.float32,
        )

    maxima = np.asarray([np.max(w) for w in windows], dtype=np.float64)
    minima = np.asarray([np.min(w) for w in windows], dtype=np.float64)
    ent = np.asarray([window_entropy(w, bins=config.entropy_bins) for w in windows], dtype=np.float64)

    rho_max = float(np.mean(maxima))
    rho_min = float(np.mean(minima))
    mu_e = float(np.mean(ent))
    var_e = float(np.mean((ent - mu_e) ** 2))

    return np.array([1, 0, 0, float(t), rho_max, rho_min, mu_e, var_e], dtype=np.float32)


def build_node_feature(
    device_name: str,
    plc_name: Optional[str],
    signal: np.ndarray,
    config: FeatureV3Config,
) -> np.ndarray:
    general = extract_general_process_features(device_name=device_name, plc_name=plc_name, config=config)
    control = extract_control_features(plc_name=plc_name, config=config)
    measurement = extract_measurement_features(signal=signal, config=config)
    return np.concatenate([general, control, measurement], axis=0).astype(np.float32)


def build_device_feature_map_v3(
    normal_df: pd.DataFrame,
    sensors: List[str],
    device_to_plc: Dict[str, str],
    config: FeatureV3Config,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for s in sensors:
        plc_name = device_to_plc.get(s)
        series = pd.to_numeric(normal_df[s], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        out[s] = build_node_feature(device_name=s, plc_name=plc_name, signal=series, config=config)
    return out
