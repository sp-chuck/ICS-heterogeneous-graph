import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ics_feature_v3 import (
    FeatureV3Config,
    build_device_feature_map_v3,
    build_real_metadata_config,
    extract_measurement_features,
)


@dataclass
class EdgeRecord:
    src_global: int
    dst_global: int
    src_type: str
    dst_type: str
    edge_type: str
    weight: float


@dataclass
class FlowRuleStats:
    rule_adjacent_stage_fit: int = 0
    rule_same_stage_fit_lit: int = 0
    rule_adjacent_device_name: int = 0
    rule_periodic_similarity: int = 0
    rule_ait_excluded: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 3-layer heterogeneous graph for SWAT: CRP/PLC/device with physical+flow edges."
    )
    parser.add_argument("--data_dir", type=str, default="/data/spz/data/swat_whole")
    parser.add_argument("--out_dir", type=str, default="/data/spz/AHGA")
    parser.add_argument(
        "--static_subdir",
        type=str,
        default="static_graph",
        help="Static graph outputs will be written to out_dir/static_subdir.",
    )
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--test_file", type=str, default="test.csv")
    parser.add_argument("--list_file", type=str, default="list.txt")
    parser.add_argument("--graph_file", type=str, default="graph100.txt")
    parser.add_argument("--physical_threshold", type=float, required=True)
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=0.75,
        help="Deprecated in rule-based flow mode. Kept for backward compatibility only.",
    )
    parser.add_argument(
        "--period_similarity_threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold for periodic-feature based flow edges (same stage only).",
    )
    parser.add_argument("--window_size", type=int, default=4300)
    parser.add_argument("--balance_factor", type=float, default=4.0)
    parser.add_argument(
        "--feature_version",
        type=str,
        default="v3",
        choices=["v2", "v3"],
        help="Node feature version: v2(legacy statistical) or v3(general+control+measurement).",
    )
    parser.add_argument("--period_w0", type=int, default=16)
    parser.add_argument("--period_w_max", type=int, default=256)
    parser.add_argument("--period_delta", type=int, default=4)
    parser.add_argument("--period_delta_w", type=int, default=2)
    parser.add_argument("--entropy_bins", type=int, default=16)
    parser.add_argument("--constant_threshold", type=float, default=1e-6)
    parser.add_argument("--large_period_value", type=float, default=1e6)
    parser.add_argument("--period_convexity_drop_ratio", type=float, default=0.02)
    parser.add_argument("--period_min_autocorr", type=float, default=0.3)
    parser.add_argument(
        "--reuse_node_features",
        action="store_true",
        help="Reuse existing hetero_node_features.csv to skip static node feature recomputation.",
    )
    parser.add_argument(
        "--reuse_node_features_file",
        type=str,
        default="",
        help="Optional path to an existing hetero_node_features.csv. Defaults to out_dir/static_subdir/hetero_node_features.csv.",
    )
    parser.add_argument(
        "--legacy_window_mode",
        action="store_true",
        help="Use DataPreprocess_Data-style weighted block features and device-index selection.",
    )
    parser.add_argument(
        "--legacy_block_size",
        type=int,
        default=100,
        help="Block size for legacy weighted aggregation (default: 100).",
    )
    parser.add_argument(
        "--legacy_device_indexes",
        type=str,
        default="1,2,3,4,6,7,8,9,10,13,15,17,18,19,20,21,22,23,25,26,27,28,29,35,36,37,39,40,41,42,45,46,47,48,50",
        help="0-based column indexes applied to train feature columns when legacy_window_mode is enabled.",
    )
    parser.add_argument("--directed", action="store_true", help="If set, keep one-direction edges when applicable.")
    parser.add_argument("--with_self_loop", action="store_true")
    parser.add_argument("--export_homo_view", action="store_true")
    parser.add_argument(
        "--temporal_mode",
        action="store_true",
        help="Enable temporal heterogeneous graph snapshots.",
    )
    parser.add_argument(
        "--temporal_window",
        type=int,
        default=256,
        help="Rows per temporal snapshot window.",
    )
    parser.add_argument(
        "--temporal_stride",
        type=int,
        default=128,
        help="Sliding stride between snapshot windows.",
    )
    parser.add_argument(
        "--max_snapshots",
        type=int,
        default=0,
        help="If >0, cap the number of generated temporal snapshots.",
    )
    parser.add_argument(
        "--build_test_temporal",
        action="store_true",
        help="Build temporal snapshots from test_file for anomaly detection/evaluation.",
    )
    parser.add_argument(
        "--test_temporal_subdir",
        type=str,
        default="temporal_snapshots_test",
        help="Output subdirectory name for temporal snapshots generated from test_file.",
    )
    parser.add_argument(
        "--fail_on_no_attack_snapshots",
        action="store_true",
        help="Fail if temporal snapshot labels contain no attack windows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_sensor_list(path: Path) -> List[str]:
    sensors = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                sensors.append(name)
    if not sensors:
        raise ValueError(f"No sensor names found in {path}")
    return sensors


def clean_train_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed") or c == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    if "attack" not in df.columns:
        raise ValueError("Column 'attack' not found in train.csv")
    return df


def align_sensor_columns(df: pd.DataFrame, sensor_list: List[str]) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    feature_cols = [c for c in df.columns if c != "attack"]

    missing = [s for s in sensor_list if s not in feature_cols]
    aligned = [s for s in sensor_list if s in feature_cols]
    extras = [c for c in feature_cols if c not in sensor_list]

    if not aligned:
        raise ValueError("No aligned sensor columns between list.txt and train.csv")

    out = df[aligned + ["attack"]].copy()
    return out, aligned, missing, extras


def parse_legacy_device_indexes(index_text: str) -> List[int]:
    idx = []
    for token in index_text.split(","):
        t = token.strip()
        if not t:
            continue
        idx.append(int(t))
    if not idx:
        raise ValueError("legacy_device_indexes resolved to an empty list")
    return idx


def select_columns_by_indexes(feature_cols: List[str], indexes: List[int]) -> List[str]:
    selected = []
    seen = set()
    for i in indexes:
        if 0 <= i < len(feature_cols):
            name = feature_cols[i]
            if name not in seen:
                selected.append(name)
                seen.add(name)
    if not selected:
        raise ValueError("No valid device columns selected by legacy indexes")
    return selected


def infer_plc_index(device_name: str) -> int:
    m = re.search(r"(\d+)", device_name)
    if not m:
        return 1

    number = int(m.group(1))
    first_digit = int(str(number)[0])
    if 1 <= first_digit <= 6:
        return first_digit

    bucket = number // 100
    if 1 <= bucket <= 6:
        return bucket

    return 1


def make_exp_weights(length: int, balance_factor: float) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.float64)

    idx = np.arange(length, dtype=np.float64)
    # Keep AHGA-style recency bias: newer points get larger weights.
    vals = np.exp(-((length - 1 - idx) * balance_factor) / max(length, 1))
    vals_sum = vals.sum()
    if vals_sum == 0:
        return np.ones(length, dtype=np.float64) / length
    return vals / vals_sum


def feature_vector(series: pd.Series, window_size: int, balance_factor: float) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if x.size == 0:
        return np.zeros(7, dtype=np.float32)

    if x.size > window_size:
        x_tail = x[-window_size:]
    else:
        x_tail = x

    w = make_exp_weights(len(x_tail), balance_factor)
    wmean = float(np.dot(x_tail, w))

    mean = float(np.mean(x))
    std = float(np.std(x))
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    last = float(x[-1])

    if len(x) >= 2:
        t = np.arange(len(x), dtype=np.float64)
        slope = float(np.polyfit(t, x, deg=1)[0])
    else:
        slope = 0.0

    return np.array([mean, std, min_v, max_v, last, slope, wmean], dtype=np.float32)


def feature_vector_legacy_block(
    series: pd.Series,
    window_size: int,
    balance_factor: float,
    block_size: int,
) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("legacy_block_size must be positive")

    x = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if x.size >= window_size:
        x_tail = x[-window_size:]
    else:
        pad = np.zeros(window_size - x.size, dtype=np.float64)
        x_tail = np.concatenate([pad, x], axis=0)

    w = make_exp_weights(window_size, balance_factor)
    weighted = x_tail * w

    out = []
    for start in range(0, window_size, block_size):
        end = min(start + block_size, window_size)
        out.append(float(np.sum(weighted[start:end])))

    return np.array(out, dtype=np.float32)


def device_features(
    normal_df: pd.DataFrame,
    sensors: List[str],
    device_to_plc: Dict[str, str],
    window_size: int,
    balance_factor: float,
    legacy_window_mode: bool = False,
    legacy_block_size: int = 100,
    feature_version: str = "v3",
    feature_v3_config: Optional[FeatureV3Config] = None,
) -> Dict[str, np.ndarray]:
    if feature_version == "v3" and not legacy_window_mode:
        cfg = feature_v3_config or FeatureV3Config()
        return build_device_feature_map_v3(
            normal_df=normal_df,
            sensors=sensors,
            device_to_plc=device_to_plc,
            config=cfg,
        )

    feats = {}
    for s in sensors:
        if legacy_window_mode:
            feats[s] = feature_vector_legacy_block(
                normal_df[s],
                window_size=window_size,
                balance_factor=balance_factor,
                block_size=legacy_block_size,
            )
        else:
            feats[s] = feature_vector(normal_df[s], window_size=window_size, balance_factor=balance_factor)
    return feats


def build_node_schema(sensors: List[str]) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, str], Dict[str, int]]:
    records = []
    global_id = 0

    records.append(
        {
            "global_id": global_id,
            "node_name": "CRP0",
            "node_type": "crp",
            "plc_name": "",
            "plc_index": 0,
        }
    )
    crp_id = global_id
    global_id += 1

    plc_name_to_global_id = {}
    for i in range(1, 7):
        name = f"PLC{i}"
        plc_name_to_global_id[name] = global_id
        records.append(
            {
                "global_id": global_id,
                "node_name": name,
                "node_type": "plc",
                "plc_name": name,
                "plc_index": i,
            }
        )
        global_id += 1

    device_name_to_global_id = {}
    device_to_plc = {}
    for dev in sensors:
        plc_idx = infer_plc_index(dev)
        plc_name = f"PLC{plc_idx}"
        device_name_to_global_id[dev] = global_id
        device_to_plc[dev] = plc_name
        records.append(
            {
                "global_id": global_id,
                "node_name": dev,
                "node_type": "device",
                "plc_name": plc_name,
                "plc_index": plc_idx,
            }
        )
        global_id += 1

    mapping_df = pd.DataFrame(records)
    aux = {
        "CRP0": crp_id,
        **plc_name_to_global_id,
        **device_name_to_global_id,
    }
    return mapping_df, aux, device_to_plc, plc_name_to_global_id


def build_node_features(
    mapping_df: pd.DataFrame,
    dev_feats: Dict[str, np.ndarray],
    device_to_plc: Dict[str, str],
    feature_names_override: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not dev_feats:
        raise ValueError("dev_feats is empty")

    sample_dim = int(next(iter(dev_feats.values())).shape[0])
    if feature_names_override is not None and len(feature_names_override) == sample_dim:
        feat_names = feature_names_override
    elif sample_dim == 7:
        feat_names = ["mean", "std", "min", "max", "last", "slope", "wmean"]
    elif sample_dim == 30:
        feat_names = [
            "general_device_type_sensor",
            "general_device_type_actuator",
            "general_device_type_controller",
            "general_device_type_unknown",
            "general_process_p1",
            "general_process_p2",
            "general_process_p3",
            "general_process_p4",
            "general_process_p5",
            "general_process_p6",
            "general_process_unk",
            "general_sub_process_sp1",
            "general_sub_process_sp2",
            "general_sub_process_sp3",
            "general_sub_process_sp4",
            "general_sub_process_unk",
            "control_plc1",
            "control_plc2",
            "control_plc3",
            "control_plc4",
            "control_plc5",
            "control_plc6",
            "measurement_periodicity_periodic",
            "measurement_periodicity_constant",
            "measurement_periodicity_non_periodic",
            "measurement_period_value",
            "measurement_period_max",
            "measurement_period_min",
            "measurement_period_entropy_mean",
            "measurement_period_entropy_var",
        ]
    elif sample_dim == 8:
        feat_names = [
            "periodicity_periodic",
            "periodicity_constant",
            "periodicity_non_periodic",
            "period_value",
            "period_max",
            "period_min",
            "period_entropy_mean",
            "period_entropy_var",
        ]
    else:
        feat_names = [f"wb_{i:03d}" for i in range(sample_dim)]

    plc_group = defaultdict(list)
    for dev, plc_name in device_to_plc.items():
        plc_group[plc_name].append(dev_feats[dev])

    plc_feats = {}
    for plc_name, vectors in plc_group.items():
        arr = np.vstack(vectors)
        plc_feats[plc_name] = np.mean(arr, axis=0).astype(np.float32)

    if plc_feats:
        crp_feat = np.mean(np.vstack(list(plc_feats.values())), axis=0).astype(np.float32)
    else:
        crp_feat = np.zeros(sample_dim, dtype=np.float32)

    rows = []
    for _, row in mapping_df.iterrows():
        node_name = row["node_name"]
        node_type = row["node_type"]

        if node_type == "device":
            vec = dev_feats[node_name]
        elif node_type == "plc":
            vec = plc_feats.get(node_name, np.zeros(sample_dim, dtype=np.float32))
        else:
            vec = crp_feat

        out = {
            "global_id": int(row["global_id"]),
            "node_name": node_name,
            "node_type": node_type,
            "plc_name": row["plc_name"],
        }
        for i, fn in enumerate(feat_names):
            out[f"feat_{fn}"] = float(vec[i])
        rows.append(out)

    feat_df = pd.DataFrame(rows)

    # Z-score normalization over all nodes for each feature column.
    feature_cols = [c for c in feat_df.columns if c.startswith("feat_")]
    for c in feature_cols:
        col = feat_df[c].to_numpy(dtype=np.float64)
        mu = col.mean()
        sd = col.std()
        if sd > 0:
            feat_df[c] = (col - mu) / sd
        else:
            feat_df[c] = 0.0

    return feat_df


def add_edge(
    out: List[EdgeRecord],
    src_global: int,
    dst_global: int,
    src_type: str,
    dst_type: str,
    edge_type: str,
    weight: float,
    directed: bool,
):
    out.append(
        EdgeRecord(
            src_global=src_global,
            dst_global=dst_global,
            src_type=src_type,
            dst_type=dst_type,
            edge_type=edge_type,
            weight=float(weight),
        )
    )
    if not directed and src_global != dst_global:
        out.append(
            EdgeRecord(
                src_global=dst_global,
                dst_global=src_global,
                src_type=dst_type,
                dst_type=src_type,
                edge_type=edge_type,
                weight=float(weight),
            )
        )


def deduplicate_edges(edges: List[EdgeRecord]) -> List[EdgeRecord]:
    best = {}
    for e in edges:
        key = (e.src_global, e.dst_global, e.edge_type)
        if key not in best:
            best[key] = e
        else:
            if abs(e.weight) > abs(best[key].weight):
                best[key] = e
    return list(best.values())


def build_hierarchy_edges(
    mapping_df: pd.DataFrame,
    id_map: Dict[str, int],
    device_to_plc: Dict[str, str],
    directed: bool,
) -> List[EdgeRecord]:
    out = []

    crp_id = id_map["CRP0"]
    for i in range(1, 7):
        plc_name = f"PLC{i}"
        plc_id = id_map[plc_name]
        add_edge(out, crp_id, plc_id, "crp", "plc", "hierarchy", 1.0, directed)

    for dev, plc_name in device_to_plc.items():
        dev_id = id_map[dev]
        plc_id = id_map[plc_name]
        add_edge(out, plc_id, dev_id, "plc", "device", "hierarchy", 1.0, directed)

    return deduplicate_edges(out)


def load_physical_matrix(path: Path) -> np.ndarray:
    mat = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("graph100.txt is not a square matrix")
    return mat


def build_physical_edges(
    sensors: List[str],
    id_map: Dict[str, int],
    dist_matrix: np.ndarray,
    threshold: float,
    directed: bool,
    with_self_loop: bool,
) -> List[EdgeRecord]:
    n = len(sensors)
    if dist_matrix.shape[0] < n:
        n = dist_matrix.shape[0]
        sensors = sensors[:n]
    elif dist_matrix.shape[0] > n:
        dist_matrix = dist_matrix[:n, :n]

    out = []
    for i in range(n):
        start_j = i if with_self_loop else i + 1
        for j in range(start_j, n):
            d = float(dist_matrix[i, j])
            if d <= threshold:
                # normalize to [0,1], closer distance -> larger weight
                w = max(0.0, 1.0 - d / max(threshold, 1e-12))
                src_name = sensors[i]
                dst_name = sensors[j]
                add_edge(
                    out,
                    id_map[src_name],
                    id_map[dst_name],
                    "device",
                    "device",
                    "physical",
                    w,
                    directed,
                )

    return deduplicate_edges(out)


def parse_device_tag(device_name: str) -> str:
    name = str(device_name).upper()
    for tag in ["FIT", "LIT", "LT", "AIT"]:
        if name.startswith(tag):
            return tag
    return "UNKNOWN"


def parse_device_stage(device_name: str) -> Optional[int]:
    m = re.search(r"(\d+)", str(device_name))
    if not m:
        return None
    digits = m.group(1)
    if not digits:
        return None
    stage = int(digits[0])
    if 1 <= stage <= 6:
        return stage
    return None


def parse_device_order(device_name: str) -> Optional[int]:
    m = re.search(r"(\d+)", str(device_name))
    if not m:
        return None
    return int(m.group(1))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_periodic_feature_map(
    source_df: pd.DataFrame,
    sensors: List[str],
    feature_v3_config: FeatureV3Config,
) -> Dict[str, np.ndarray]:
    periodic_map: Dict[str, np.ndarray] = {}
    for sensor in sensors:
        arr = pd.to_numeric(source_df[sensor], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        periodic_map[sensor] = extract_measurement_features(arr, feature_v3_config)
    return periodic_map


def build_periodic_feature_map_from_node_features(
    feature_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    periodic_cols_30 = [
        "feat_measurement_periodicity_periodic",
        "feat_measurement_periodicity_constant",
        "feat_measurement_periodicity_non_periodic",
        "feat_measurement_period_value",
        "feat_measurement_period_max",
        "feat_measurement_period_min",
        "feat_measurement_period_entropy_mean",
        "feat_measurement_period_entropy_var",
    ]
    periodic_cols_8 = [
        "feat_periodicity_periodic",
        "feat_periodicity_constant",
        "feat_periodicity_non_periodic",
        "feat_period_value",
        "feat_period_max",
        "feat_period_min",
        "feat_period_entropy_mean",
        "feat_period_entropy_var",
    ]

    if all(c in feature_df.columns for c in periodic_cols_30):
        use_cols = periodic_cols_30
    elif all(c in feature_df.columns for c in periodic_cols_8):
        use_cols = periodic_cols_8
    else:
        return {}

    part = feature_df[feature_df["node_type"] == "device"].copy()
    out: Dict[str, np.ndarray] = {}
    for _, row in part.iterrows():
        name = str(row["node_name"])
        vec = row[use_cols].to_numpy(dtype=np.float64)
        out[name] = vec
    return out


def build_flow_edges(
    sensors: List[str],
    id_map: Dict[str, int],
    directed: bool,
    with_self_loop: bool,
    periodic_feature_map: Dict[str, np.ndarray],
    period_similarity_threshold: float,
) -> Tuple[List[EdgeRecord], FlowRuleStats]:
    stats = FlowRuleStats()

    out = []
    n = len(sensors)
    for i in range(n):
        start_j = i if with_self_loop else i + 1
        for j in range(start_j, n):
            s1 = sensors[i]
            s2 = sensors[j]
            tag1 = parse_device_tag(s1)
            tag2 = parse_device_tag(s2)

            # Rule 4 has veto priority: AITs and other sensors are not correlated.
            if ((tag1 == "AIT") and (tag2 != "AIT")) or ((tag2 == "AIT") and (tag1 != "AIT")):
                stats.rule_ait_excluded += 1
                continue

            stage1 = parse_device_stage(s1)
            stage2 = parse_device_stage(s2)
            order1 = parse_device_order(s1)
            order2 = parse_device_order(s2)

            matched = False

            # Rule 1: FITs from adjacent stages.
            if tag1 == "FIT" and tag2 == "FIT" and stage1 is not None and stage2 is not None:
                if abs(stage1 - stage2) == 1:
                    matched = True
                    stats.rule_adjacent_stage_fit += 1

            # Rule 2: FIT with LIT/LT in same stage.
            if (not matched) and stage1 is not None and stage2 is not None and stage1 == stage2:
                tags = {tag1, tag2}
                if "FIT" in tags and ("LIT" in tags or "LT" in tags):
                    matched = True
                    stats.rule_same_stage_fit_lit += 1

            # Rule 3: Adjacent devices in piping diagram (name-based approximation).
            if (not matched) and stage1 is not None and stage2 is not None and order1 is not None and order2 is not None:
                if stage1 == stage2 and abs(order1 - order2) == 1:
                    matched = True
                    stats.rule_adjacent_device_name += 1

            # Rule 5: Same-stage devices with similar periodic features.
            if (not matched) and stage1 is not None and stage2 is not None and stage1 == stage2:
                feat1 = periodic_feature_map.get(s1)
                feat2 = periodic_feature_map.get(s2)
                if feat1 is not None and feat2 is not None:
                    sim = cosine_similarity(feat1, feat2)
                    if sim >= period_similarity_threshold:
                        matched = True
                        stats.rule_periodic_similarity += 1

            if matched:
                add_edge(
                    out,
                    id_map[s1],
                    id_map[s2],
                    "device",
                    "device",
                    "flow",
                    1.0,
                    directed,
                )

    return deduplicate_edges(out), stats


def edges_to_df(edges: List[EdgeRecord]) -> pd.DataFrame:
    rows = []
    for e in edges:
        rows.append(
            {
                "src_global": e.src_global,
                "dst_global": e.dst_global,
                "src_type": e.src_type,
                "dst_type": e.dst_type,
                "edge_type": e.edge_type,
                "weight": e.weight,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["src_global", "dst_global", "src_type", "dst_type", "edge_type", "weight"]
        )
    return pd.DataFrame(rows)


def save_numpy_hetero(
    out_dir: Path,
    feature_df: pd.DataFrame,
    all_edges: List[EdgeRecord],
):
    type_to_nodes = {
        "crp": feature_df[feature_df["node_type"] == "crp"].sort_values("global_id"),
        "plc": feature_df[feature_df["node_type"] == "plc"].sort_values("global_id"),
        "device": feature_df[feature_df["node_type"] == "device"].sort_values("global_id"),
    }

    feat_cols = [c for c in feature_df.columns if c.startswith("feat_")]

    global_to_local = {}
    for t, part in type_to_nodes.items():
        ids = part["global_id"].to_numpy(dtype=np.int64)
        for local_idx, gid in enumerate(ids.tolist()):
            global_to_local[(t, int(gid))] = local_idx

        x = part[feat_cols].to_numpy(dtype=np.float32)
        np.save(out_dir / f"x_{t}.npy", x)

    grouped = defaultdict(list)
    for e in all_edges:
        key = (e.src_type, e.edge_type, e.dst_type)
        src_local = global_to_local[(e.src_type, e.src_global)]
        dst_local = global_to_local[(e.dst_type, e.dst_global)]
        grouped[key].append((src_local, dst_local, e.weight))

    relation_files = []
    for (src_t, rel, dst_t), triples in grouped.items():
        edge_index = np.array([[a, b] for a, b, _ in triples], dtype=np.int64).T
        edge_weight = np.array([w for _, _, w in triples], dtype=np.float32)

        safe = f"{src_t}__{rel}__{dst_t}"
        eidx_name = f"edge_index__{safe}.npy"
        ew_name = f"edge_weight__{safe}.npy"

        np.save(out_dir / eidx_name, edge_index)
        np.save(out_dir / ew_name, edge_weight)

        relation_files.append(
            {
                "src_type": src_t,
                "relation": rel,
                "dst_type": dst_t,
                "edge_index_file": eidx_name,
                "edge_weight_file": ew_name,
                "num_edges": int(edge_index.shape[1]),
            }
        )

    meta = {
        "node_types": {
            t: int(part.shape[0]) for t, part in type_to_nodes.items()
        },
        "feature_columns": feat_cols,
        "relations": relation_files,
        "format": "numpy_for_pyg_heterodata",
    }

    with (out_dir / "hetero_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def make_snapshot_ranges(
    n_rows: int,
    window: int,
    stride: int,
    max_snapshots: int = 0,
) -> List[Tuple[int, int]]:
    if window <= 0 or stride <= 0:
        raise ValueError("temporal_window and temporal_stride must be positive")
    if n_rows < window:
        return []

    ranges = []
    start = 0
    while start + window <= n_rows:
        ranges.append((start, start + window))
        if max_snapshots > 0 and len(ranges) >= max_snapshots:
            break
        start += stride
    return ranges


def run_temporal_snapshots(
    out_dir: Path,
    temporal_subdir: str,
    train_df: pd.DataFrame,
    aligned_sensors: List[str],
    mapping_df: pd.DataFrame,
    id_map: Dict[str, int],
    device_to_plc: Dict[str, str],
    hierarchy_edges: List[EdgeRecord],
    physical_edges: List[EdgeRecord],
    feature_v3_config: FeatureV3Config,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    temporal_root = out_dir / temporal_subdir
    temporal_root.mkdir(parents=True, exist_ok=True)

    windows = make_snapshot_ranges(
        n_rows=len(train_df),
        window=args.temporal_window,
        stride=args.temporal_stride,
        max_snapshots=args.max_snapshots,
    )

    total_device_pos = 0
    total_device_count = 0
    agg_flow_stats = FlowRuleStats()
    summaries: List[Dict[str, Any]] = []
    for i, (start, end) in enumerate(windows):
        window_df = train_df.iloc[start:end].copy()
        snapshot_label = int((window_df["attack"] > 0).any())

        window_normal = window_df[window_df["attack"] == 0].copy()
        if window_normal.empty:
            # Keep a fallback window for feature extraction in all-attack windows.
            window_normal = window_df

        dev_feats = device_features(
            normal_df=window_normal,
            sensors=aligned_sensors,
            device_to_plc=device_to_plc,
            window_size=args.window_size,
            balance_factor=args.balance_factor,
            legacy_window_mode=args.legacy_window_mode,
            legacy_block_size=args.legacy_block_size,
            feature_version=args.feature_version,
            feature_v3_config=feature_v3_config,
        )
        if args.feature_version == "v3" and not args.legacy_window_mode:
            feat_df = build_node_features(
                mapping_df,
                dev_feats,
                device_to_plc,
                feature_names_override=feature_v3_config.full_feature_names(),
            )
        else:
            feat_df = build_node_features(mapping_df, dev_feats, device_to_plc)

        periodic_feature_map = build_periodic_feature_map_from_node_features(feat_df)
        if not periodic_feature_map:
            periodic_feature_map = build_periodic_feature_map(
                source_df=window_normal,
                sensors=aligned_sensors,
                feature_v3_config=feature_v3_config,
            )

        flow_edges, flow_stats = build_flow_edges(
            sensors=aligned_sensors,
            id_map=id_map,
            directed=args.directed,
            with_self_loop=args.with_self_loop,
            periodic_feature_map=periodic_feature_map,
            period_similarity_threshold=args.period_similarity_threshold,
        )

        agg_flow_stats.rule_adjacent_stage_fit += flow_stats.rule_adjacent_stage_fit
        agg_flow_stats.rule_same_stage_fit_lit += flow_stats.rule_same_stage_fit_lit
        agg_flow_stats.rule_adjacent_device_name += flow_stats.rule_adjacent_device_name
        agg_flow_stats.rule_periodic_similarity += flow_stats.rule_periodic_similarity
        agg_flow_stats.rule_ait_excluded += flow_stats.rule_ait_excluded

        all_edges = deduplicate_edges(hierarchy_edges + physical_edges + flow_edges)

        snap_dir = temporal_root / f"snapshot_{i:05d}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        mapping_df.to_csv(snap_dir / "node_mapping.csv", index=False)
        feat_df.to_csv(snap_dir / "hetero_node_features.csv", index=False)
        edges_to_df(flow_edges).to_csv(snap_dir / "edges_flow.csv", index=False)
        edges_to_df(all_edges).to_csv(snap_dir / "hetero_edges.csv", index=False)
        save_numpy_hetero(snap_dir, feat_df, all_edges)

        # Device-only ground truth: supervision is defined on现场设备 nodes only.
        device_map = mapping_df[mapping_df["node_type"] == "device"].copy().sort_values("global_id")
        device_label = np.full(shape=(device_map.shape[0],), fill_value=snapshot_label, dtype=np.int64)
        device_gt_df = pd.DataFrame(
            {
                "global_id": device_map["global_id"].astype(int).to_numpy(),
                "node_name": device_map["node_name"].astype(str).to_numpy(),
                "node_type": "device",
                "device_label": device_label,
            }
        )
        device_gt_df.to_csv(snap_dir / "device_ground_truth.csv", index=False)
        np.save(snap_dir / "device_ground_truth.npy", device_label)

        num_device_attack = int(device_label.sum())
        num_device_total = int(device_label.shape[0])
        total_device_pos += num_device_attack
        total_device_count += num_device_total

        snap_meta = {
            "snapshot_id": i,
            "row_start": int(start),
            "row_end_exclusive": int(end),
            "window_size": int(end - start),
            "snapshot_label": snapshot_label,
            "num_attack_rows": int((window_df["attack"] > 0).sum()),
            "num_normal_rows": int((window_df["attack"] == 0).sum()),
            "num_flow_edges": int(len(flow_edges)),
            "num_all_edges": int(len(all_edges)),
            "num_device_nodes": num_device_total,
            "num_device_attack_labels": num_device_attack,
            "device_attack_ratio": float(num_device_attack / max(num_device_total, 1)),
            "flow_rule_stats": {
                "adjacent_stage_fit": int(flow_stats.rule_adjacent_stage_fit),
                "same_stage_fit_lit": int(flow_stats.rule_same_stage_fit_lit),
                "adjacent_device_name": int(flow_stats.rule_adjacent_device_name),
                "periodic_similarity": int(flow_stats.rule_periodic_similarity),
                "ait_excluded": int(flow_stats.rule_ait_excluded),
            },
            "recommended_model": "HTGNN",
            "model_description": "Heterogeneous GNN encoder + GRU over snapshot sequence",
        }
        with (snap_dir / "snapshot_meta.json").open("w", encoding="utf-8") as f:
            json.dump(snap_meta, f, ensure_ascii=False, indent=2)
        summaries.append(snap_meta)

    summary = {
        "temporal_mode": True,
        "temporal_subdir": temporal_subdir,
        "recommended_model": "HTGNN",
        "model_description": "Heterogeneous GNN encoder + GRU over snapshot sequence",
        "num_snapshots": len(summaries),
        "window": int(args.temporal_window),
        "stride": int(args.temporal_stride),
        "snapshot_labels": {
            "normal": int(sum(1 for s in summaries if s["snapshot_label"] == 0)),
            "attack": int(sum(1 for s in summaries if s["snapshot_label"] == 1)),
        },
        "device_ground_truth": {
            "scope": "device_only",
            "total_device_labels": int(total_device_count),
            "attack_device_labels": int(total_device_pos),
            "normal_device_labels": int(total_device_count - total_device_pos),
            "attack_ratio": float(total_device_pos / max(total_device_count, 1)),
        },
        "flow_rule_stats": {
            "adjacent_stage_fit": int(agg_flow_stats.rule_adjacent_stage_fit),
            "same_stage_fit_lit": int(agg_flow_stats.rule_same_stage_fit_lit),
            "adjacent_device_name": int(agg_flow_stats.rule_adjacent_device_name),
            "periodic_similarity": int(agg_flow_stats.rule_periodic_similarity),
            "ait_excluded": int(agg_flow_stats.rule_ait_excluded),
        },
        "snapshots": summaries,
    }
    with (temporal_root / "snapshots_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(summaries).to_csv(temporal_root / "snapshots_metadata.csv", index=False)

    return summary


def build_heterodata(out_dir: str):
    """
    Optional helper to reconstruct HeteroData from saved numpy files.
    Requires torch and torch_geometric installed in your runtime.
    """
    import torch
    from torch_geometric.data import HeteroData

    out_path = Path(out_dir)
    meta = json.loads((out_path / "hetero_meta.json").read_text(encoding="utf-8"))

    data = HeteroData()
    for node_type in ["crp", "plc", "device"]:
        x = np.load(out_path / f"x_{node_type}.npy")
        data[node_type].x = torch.tensor(x, dtype=torch.float)

    for rel in meta["relations"]:
        eidx = np.load(out_path / rel["edge_index_file"])
        ew = np.load(out_path / rel["edge_weight_file"])
        data[(rel["src_type"], rel["relation"], rel["dst_type"])].edge_index = torch.tensor(
            eidx, dtype=torch.long
        )
        data[(rel["src_type"], rel["relation"], rel["dst_type"])].edge_attr = torch.tensor(
            ew.reshape(-1, 1), dtype=torch.float
        )

    return data


def write_report(
    out_dir: Path,
    cfg: Dict,
    mapping_df: pd.DataFrame,
    h_edges: List[EdgeRecord],
    p_edges: List[EdgeRecord],
    f_edges: List[EdgeRecord],
    all_edges: List[EdgeRecord],
    missing: List[str],
    extras: List[str],
    flow_rule_stats: Optional[FlowRuleStats] = None,
    temporal_summary: Optional[Dict[str, Any]] = None,
):
    node_counts = mapping_df["node_type"].value_counts().to_dict()

    edge_type_counts = defaultdict(int)
    out_deg = defaultdict(int)
    in_deg = defaultdict(int)
    for e in all_edges:
        edge_type_counts[e.edge_type] += 1
        out_deg[e.src_global] += 1
        in_deg[e.dst_global] += 1

    total_nodes = int(mapping_df.shape[0])
    isolated = 0
    for gid in mapping_df["global_id"].tolist():
        if out_deg[int(gid)] + in_deg[int(gid)] == 0:
            isolated += 1

    report = {
        "config": cfg,
        "node_counts": {k: int(v) for k, v in node_counts.items()},
        "edge_counts": {
            "hierarchy": int(len(h_edges)),
            "physical": int(len(p_edges)),
            "flow": int(len(f_edges)),
            "all": int(len(all_edges)),
        },
        "edge_type_counts": {k: int(v) for k, v in edge_type_counts.items()},
        "isolated_nodes": int(isolated),
        "isolated_ratio": float(isolated / max(total_nodes, 1)),
        "missing_in_train_vs_list": missing,
        "extra_in_train_vs_list": extras,
        "recommended_model": (
            "HTGNN (Heterogeneous GNN + GRU over snapshots)"
            if cfg.get("temporal_mode", False)
            else "Heterogeneous GNN (static graph baseline)"
        ),
    }

    if temporal_summary is not None:
        report["temporal_summary"] = temporal_summary
    if flow_rule_stats is not None:
        report["flow_rule_stats"] = {
            "adjacent_stage_fit": int(flow_rule_stats.rule_adjacent_stage_fit),
            "same_stage_fit_lit": int(flow_rule_stats.rule_same_stage_fit_lit),
            "adjacent_device_name": int(flow_rule_stats.rule_adjacent_device_name),
            "periodic_similarity": int(flow_rule_stats.rule_periodic_similarity),
            "ait_excluded": int(flow_rule_stats.rule_ait_excluded),
        }

    with (out_dir / "preprocess_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = []
    md.append("# Preprocess Report")
    md.append("")
    md.append("## Config")
    for k, v in cfg.items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Node Counts")
    for k, v in report["node_counts"].items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Edge Counts")
    for k, v in report["edge_counts"].items():
        md.append(f"- {k}: {v}")
    md.append("")
    if flow_rule_stats is not None:
        md.append("## Flow Rule Stats")
        md.append(f"- adjacent_stage_fit: {int(flow_rule_stats.rule_adjacent_stage_fit)}")
        md.append(f"- same_stage_fit_lit: {int(flow_rule_stats.rule_same_stage_fit_lit)}")
        md.append(f"- adjacent_device_name: {int(flow_rule_stats.rule_adjacent_device_name)}")
        md.append(f"- periodic_similarity: {int(flow_rule_stats.rule_periodic_similarity)}")
        md.append(f"- ait_excluded: {int(flow_rule_stats.rule_ait_excluded)}")
        md.append("")

    md.append("## Quality")
    md.append(f"- isolated_nodes: {report['isolated_nodes']}")
    md.append(f"- isolated_ratio: {report['isolated_ratio']:.6f}")
    md.append("")
    md.append("## Column Alignment")
    md.append(f"- missing_in_train_vs_list: {len(missing)}")
    md.append(f"- extra_in_train_vs_list: {len(extras)}")
    md.append("")
    md.append("## Model")
    if cfg.get("temporal_mode", False):
        md.append("- recommended_model: HTGNN")
        md.append("- architecture: Heterogeneous GNN encoder + GRU over snapshot sequence")
    else:
        md.append("- recommended_model: Heterogeneous GNN (static graph baseline)")

    if temporal_summary is not None:
        md.append("")
        md.append("## Temporal Summary")
        md.append(f"- num_snapshots: {temporal_summary.get('num_snapshots', 0)}")
        labels = temporal_summary.get("snapshot_labels", {})
        md.append(f"- snapshot_label_normal: {labels.get('normal', 0)}")
        md.append(f"- snapshot_label_attack: {labels.get('attack', 0)}")
        md.append(f"- temporal_window: {temporal_summary.get('window', 0)}")
        md.append(f"- temporal_stride: {temporal_summary.get('stride', 0)}")

    (out_dir / "preprocess_report.md").write_text("\n".join(md), encoding="utf-8")


def export_homo_view(out_dir: Path, mapping_df: pd.DataFrame, all_edges: List[EdgeRecord]):
    node_view = mapping_df[["global_id", "node_name", "node_type", "plc_name", "plc_index"]].copy()
    node_view.to_csv(out_dir / "homograph_nodes.csv", index=False)

    edges_df = edges_to_df(all_edges)
    edges_df.to_csv(out_dir / "homograph_edges.csv", index=False)


def export_v3_metadata_mapping(
    out_dir: Path,
    aligned_sensors: List[str],
    device_to_plc: Dict[str, str],
    feature_v3_config: FeatureV3Config,
):
    rows = []
    for dev in aligned_sensors:
        plc_name = device_to_plc.get(dev, "")
        if feature_v3_config.metadata_map and dev in feature_v3_config.metadata_map:
            device_type, process, sub_process = feature_v3_config.metadata_map[dev]
        else:
            device_type, process, sub_process = ("UNKNOWN", "PUNK", "SPUNK")
        rows.append(
            {
                "device_name": dev,
                "plc_name": plc_name,
                "device_type": device_type,
                "process": process,
                "sub_process": sub_process,
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "device_metadata_mapping.csv", index=False)


def load_cached_node_features(cached_path: Path, mapping_df: pd.DataFrame) -> pd.DataFrame:
    if not cached_path.exists():
        raise FileNotFoundError(f"Cached node feature file not found: {cached_path}")

    cached = pd.read_csv(cached_path)
    required = {"global_id", "node_name", "node_type"}
    missing_required = required - set(cached.columns)
    if missing_required:
        raise ValueError(
            f"Cached feature file missing required columns: {sorted(missing_required)}"
        )

    feat_cols = [c for c in cached.columns if c.startswith("feat_")]
    if not feat_cols:
        raise ValueError("Cached feature file has no feature columns starting with 'feat_'")

    cached_names = set(cached["node_name"].astype(str).tolist())
    target_names = set(mapping_df["node_name"].astype(str).tolist())
    if cached_names != target_names:
        missing_in_cached = sorted(list(target_names - cached_names))
        extra_in_cached = sorted(list(cached_names - target_names))
        raise ValueError(
            "Cached feature file node set mismatch. "
            f"missing_in_cached={missing_in_cached[:10]} "
            f"extra_in_cached={extra_in_cached[:10]}"
        )

    by_name = cached.set_index("node_name")
    rows = []
    for _, mrow in mapping_df.sort_values("global_id").iterrows():
        name = str(mrow["node_name"])
        crow = by_name.loc[name]

        out = {
            "global_id": int(mrow["global_id"]),
            "node_name": name,
            "node_type": str(mrow["node_type"]),
            "plc_name": str(mrow["plc_name"]),
        }
        for c in feat_cols:
            out[c] = float(crow[c])
        rows.append(out)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    static_out_dir = out_dir / args.static_subdir
    static_out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / args.train_file
    test_path = data_dir / args.test_file
    list_path = data_dir / args.list_file
    graph_path = data_dir / args.graph_file

    print("[1/6] Loading source files...")
    sensor_list = load_sensor_list(list_path)
    raw_train_df = clean_train_df(train_path)

    if args.legacy_window_mode:
        feature_cols = [c for c in raw_train_df.columns if c != "attack"]
        legacy_indexes = parse_legacy_device_indexes(args.legacy_device_indexes)
        aligned_sensors = select_columns_by_indexes(feature_cols, legacy_indexes)
        train_df = raw_train_df[aligned_sensors + ["attack"]].copy()
        missing = [s for s in sensor_list if s not in aligned_sensors]
        extras = [c for c in feature_cols if c not in sensor_list]
    else:
        train_df, aligned_sensors, missing, extras = align_sensor_columns(raw_train_df, sensor_list)

    print(f"  - sensors in list.txt: {len(sensor_list)}")
    print(f"  - aligned sensors used: {len(aligned_sensors)}")
    print(f"  - missing sensors: {len(missing)}")
    print(f"  - extra train columns: {len(extras)}")
    if args.legacy_window_mode:
        print(f"  - legacy selected sensors by index: {len(aligned_sensors)}")

    print("[2/6] Building 3-layer node schema (CRP/PLC/device)...")
    mapping_df, id_map, device_to_plc, plc_name_to_global = build_node_schema(aligned_sensors)
    mapping_df.to_csv(static_out_dir / "node_mapping.csv", index=False)

    feature_v3_config = FeatureV3Config(
        w0=args.period_w0,
        w_max=args.period_w_max,
        delta=args.period_delta,
        delta_w=args.period_delta_w,
        entropy_bins=args.entropy_bins,
        constant_threshold=args.constant_threshold,
        large_period_value=args.large_period_value,
        convexity_drop_ratio=args.period_convexity_drop_ratio,
        min_period_autocorr=args.period_min_autocorr,
        controller_count=6,
    )
    feature_v3_config = build_real_metadata_config(aligned_sensors, base=feature_v3_config)

    print("[3/6] Building node features from normal train samples...")
    normal_df = train_df[train_df["attack"] == 0].copy()
    if normal_df.empty:
        raise ValueError("No normal samples found (attack==0) in train.csv")

    if args.reuse_node_features:
        cached_path = (
            Path(args.reuse_node_features_file)
            if args.reuse_node_features_file
            else (static_out_dir / "hetero_node_features.csv")
        )
        print(f"  - Reusing node features from: {cached_path}")
        feat_df = load_cached_node_features(cached_path, mapping_df)
    else:
        dev_feats = device_features(
            normal_df=normal_df,
            sensors=aligned_sensors,
            device_to_plc=device_to_plc,
            window_size=args.window_size,
            balance_factor=args.balance_factor,
            legacy_window_mode=args.legacy_window_mode,
            legacy_block_size=args.legacy_block_size,
            feature_version=args.feature_version,
            feature_v3_config=feature_v3_config,
        )
        if args.feature_version == "v3" and not args.legacy_window_mode:
            feat_df = build_node_features(
                mapping_df,
                dev_feats,
                device_to_plc,
                feature_names_override=feature_v3_config.full_feature_names(),
            )
        else:
            feat_df = build_node_features(mapping_df, dev_feats, device_to_plc)
    feat_df.to_csv(static_out_dir / "hetero_node_features.csv", index=False)

    print("[4/6] Building hierarchy/physical/flow edges...")
    hierarchy_edges = build_hierarchy_edges(mapping_df, id_map, device_to_plc, directed=args.directed)

    dist_matrix = load_physical_matrix(graph_path)
    physical_edges = build_physical_edges(
        sensors=aligned_sensors,
        id_map=id_map,
        dist_matrix=dist_matrix,
        threshold=args.physical_threshold,
        directed=args.directed,
        with_self_loop=args.with_self_loop,
    )

    periodic_feature_map = build_periodic_feature_map_from_node_features(feat_df)
    if not periodic_feature_map:
        periodic_feature_map = build_periodic_feature_map(
            source_df=normal_df,
            sensors=aligned_sensors,
            feature_v3_config=feature_v3_config,
        )

    flow_edges, flow_rule_stats = build_flow_edges(
        sensors=aligned_sensors,
        id_map=id_map,
        directed=args.directed,
        with_self_loop=args.with_self_loop,
        periodic_feature_map=periodic_feature_map,
        period_similarity_threshold=args.period_similarity_threshold,
    )

    hierarchy_df = edges_to_df(hierarchy_edges)
    physical_df = edges_to_df(physical_edges)
    flow_df = edges_to_df(flow_edges)

    hierarchy_df.to_csv(static_out_dir / "edges_hierarchy.csv", index=False)
    physical_df.to_csv(static_out_dir / "edges_physical.csv", index=False)
    flow_df.to_csv(static_out_dir / "edges_flow.csv", index=False)

    all_edges = deduplicate_edges(hierarchy_edges + physical_edges + flow_edges)
    all_edges_df = edges_to_df(all_edges)
    all_edges_df.to_csv(static_out_dir / "hetero_edges.csv", index=False)

    print("[5/6] Exporting numpy tensors for PyG HeteroData...")
    save_numpy_hetero(static_out_dir, feat_df, all_edges)

    if args.export_homo_view:
        export_homo_view(static_out_dir, mapping_df, all_edges)

    if args.feature_version == "v3":
        export_v3_metadata_mapping(
            out_dir=static_out_dir,
            aligned_sensors=aligned_sensors,
            device_to_plc=device_to_plc,
            feature_v3_config=feature_v3_config,
        )

    temporal_summary = None
    if args.temporal_mode:
        print("[5b/6] Exporting temporal heterogeneous snapshots...")
        temporal_summary = run_temporal_snapshots(
            out_dir=out_dir,
            temporal_subdir="temporal_snapshots",
            train_df=train_df,
            aligned_sensors=aligned_sensors,
            mapping_df=mapping_df,
            id_map=id_map,
            device_to_plc=device_to_plc,
            hierarchy_edges=hierarchy_edges,
            physical_edges=physical_edges,
            feature_v3_config=feature_v3_config,
            args=args,
        )
        if temporal_summary["snapshot_labels"].get("attack", 0) == 0:
            msg = "No attack snapshots generated (snapshot_label=1 count is 0)."
            if args.fail_on_no_attack_snapshots:
                raise ValueError(msg)
            print(f"[WARN] {msg}")

    test_temporal_summary = None
    if args.temporal_mode and args.build_test_temporal:
        print("[5c/6] Exporting temporal snapshots from test file...")
        raw_test_df = clean_train_df(test_path)
        test_feature_cols = [c for c in raw_test_df.columns if c != "attack"]
        missing_in_test = [s for s in aligned_sensors if s not in test_feature_cols]
        if missing_in_test:
            raise ValueError(
                f"test_file is missing aligned train sensors: {missing_in_test[:10]}"
                + ("..." if len(missing_in_test) > 10 else "")
            )
        test_df = raw_test_df[aligned_sensors + ["attack"]].copy()

        test_temporal_summary = run_temporal_snapshots(
            out_dir=out_dir,
            temporal_subdir=args.test_temporal_subdir,
            train_df=test_df,
            aligned_sensors=aligned_sensors,
            mapping_df=mapping_df,
            id_map=id_map,
            device_to_plc=device_to_plc,
            hierarchy_edges=hierarchy_edges,
            physical_edges=physical_edges,
            feature_v3_config=feature_v3_config,
            args=args,
        )
        if test_temporal_summary["snapshot_labels"].get("attack", 0) == 0:
            msg = "No attack snapshots generated in test temporal set."
            if args.fail_on_no_attack_snapshots:
                raise ValueError(msg)
            print(f"[WARN] {msg}")

    cfg = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "static_out_dir": str(static_out_dir),
        "static_subdir": args.static_subdir,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "list_file": args.list_file,
        "graph_file": args.graph_file,
        "physical_threshold": args.physical_threshold,
        "corr_threshold": args.corr_threshold,
        "flow_rule_mode": "rule_based_no_pearson",
        "period_similarity_threshold": args.period_similarity_threshold,
        "window_size": args.window_size,
        "balance_factor": args.balance_factor,
        "feature_version": args.feature_version,
        "legacy_window_mode": args.legacy_window_mode,
        "legacy_block_size": args.legacy_block_size,
        "legacy_device_indexes": args.legacy_device_indexes,
        "period_w0": args.period_w0,
        "period_w_max": args.period_w_max,
        "period_delta": args.period_delta,
        "period_delta_w": args.period_delta_w,
        "entropy_bins": args.entropy_bins,
        "constant_threshold": args.constant_threshold,
        "large_period_value": args.large_period_value,
        "period_convexity_drop_ratio": args.period_convexity_drop_ratio,
        "period_min_autocorr": args.period_min_autocorr,
        "reuse_node_features": args.reuse_node_features,
        "reuse_node_features_file": args.reuse_node_features_file,
        "v3_device_types": list(feature_v3_config.device_types),
        "v3_processes": list(feature_v3_config.processes),
        "v3_sub_processes": list(feature_v3_config.sub_processes),
        "directed": args.directed,
        "with_self_loop": args.with_self_loop,
        "export_homo_view": args.export_homo_view,
        "temporal_mode": args.temporal_mode,
        "temporal_window": args.temporal_window,
        "temporal_stride": args.temporal_stride,
        "max_snapshots": args.max_snapshots,
        "build_test_temporal": args.build_test_temporal,
        "test_temporal_subdir": args.test_temporal_subdir,
        "seed": args.seed,
        "plc_rule": "Infer by first digit of numeric code in device name: 1xx->PLC1 ... 6xx->PLC6",
        "crp_rule": "Single virtual CRP0 connected to all virtual PLC1..PLC6",
        "flow_rules": [
            "FITs from adjacent stages => true",
            "FITs and LIT/LT within the same stage => true",
            "Adjacent devices in piping diagram (name-based) => true",
            "AITs and other sensors => false",
            "Devices in same stage with similar periodic features => true",
        ],
    }

    print("[6/6] Writing report...")
    write_report(
        out_dir=static_out_dir,
        cfg=cfg,
        mapping_df=mapping_df,
        h_edges=hierarchy_edges,
        p_edges=physical_edges,
        f_edges=flow_edges,
        all_edges=all_edges,
        missing=missing,
        extras=extras,
        flow_rule_stats=flow_rule_stats,
        temporal_summary=temporal_summary,
    )

    if test_temporal_summary is not None:
        with (static_out_dir / "preprocess_test_temporal_summary.json").open("w", encoding="utf-8") as f:
            json.dump(test_temporal_summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Output directory (root): {out_dir}")
    print(f"Static graph directory: {static_out_dir}")
    print("Generated files:")
    print(f"  - {args.static_subdir}/node_mapping.csv")
    print(f"  - {args.static_subdir}/hetero_node_features.csv")
    print(f"  - {args.static_subdir}/edges_hierarchy.csv")
    print(f"  - {args.static_subdir}/edges_physical.csv")
    print(f"  - {args.static_subdir}/edges_flow.csv")
    print(f"  - {args.static_subdir}/hetero_edges.csv")
    print(f"  - {args.static_subdir}/x_crp.npy, x_plc.npy, x_device.npy")
    print(f"  - {args.static_subdir}/edge_index__*.npy, edge_weight__*.npy")
    print(f"  - {args.static_subdir}/hetero_meta.json")
    if args.feature_version == "v3":
        print(f"  - {args.static_subdir}/device_metadata_mapping.csv")
    print(f"  - {args.static_subdir}/preprocess_report.json, preprocess_report.md")
    if args.temporal_mode:
        print("  - temporal_snapshots/snapshot_*/")
        print("  - temporal_snapshots/snapshot_*/device_ground_truth.csv")
        print("  - temporal_snapshots/snapshot_*/device_ground_truth.npy")
        print("  - temporal_snapshots/snapshots_metadata.json")
        if args.build_test_temporal:
            print(f"  - {args.test_temporal_subdir}/snapshot_*/")
            print(f"  - {args.test_temporal_subdir}/snapshot_*/device_ground_truth.csv")
            print(f"  - {args.test_temporal_subdir}/snapshot_*/device_ground_truth.npy")
            print(f"  - {args.test_temporal_subdir}/snapshots_metadata.json")
        print("Temporal recommended model: HTGNN (Heterogeneous GNN + GRU over snapshots)")


if __name__ == "__main__":
    main()
