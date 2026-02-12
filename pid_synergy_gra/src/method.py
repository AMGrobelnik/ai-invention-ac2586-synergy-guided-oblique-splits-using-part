#!/usr/bin/env python3
"""Pairwise PID Synergy Graph Construction Across 12 Benchmark Datasets.

Computes pairwise Partial Information Decomposition (PID) synergy scores for all
feature pairs across 12 tabular classification datasets using a manual MI-based
PID implementation (Williams-Beer I_min). Constructs synergy graphs at 3 threshold
levels and characterizes graph structure.

Baseline comparison: Mutual Information (MI) based feature interaction graphs
using I(Fi,Fj;Y) - I(Fi;Y) - I(Fj;Y) as an interaction score, compared against
the PID synergy decomposition which cleanly separates synergy from redundancy.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import time
import resource
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import KBinsDiscretizer
import networkx as nx

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
BLUE, GREEN, YELLOW, CYAN, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[0m"
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level: <7}}|{CYAN}{{name: >12.12}}{END}.{CYAN}{{function: <22.22}}{END}:{CYAN}{{line: <4}}{END}| {{message}}",
    colorize=False,
)
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(log_dir / "run.log", rotation="30 MB", level="DEBUG")

# ── Resource limits ──────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))  # 14 GB RAM
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))  # ~58 min CPU

# ── Constants ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DIR1 = WORKSPACE.parent.parent.parent / "iter_1" / "gen_art" / "data_id1_it1__opus"
DATA_DIR2 = WORKSPACE.parent.parent.parent / "iter_1" / "gen_art" / "data_id2_it1__opus"
DATA_DIR3 = WORKSPACE.parent.parent.parent / "iter_1" / "gen_art" / "data_id3_it1__opus"

BIN_LEVELS = [5, 10]
THRESHOLDS = {"top_10pct": 0.90, "top_25pct": 0.75, "above_median": 0.50}
PER_DATASET_TIMEOUT_S = 300  # 5 min per dataset

# ── Domain knowledge pairs for checking ──────────────────────────────────────
DOMAIN_PAIRS = {
    "diabetes": [("plas", "insu"), ("plas", "mass"), ("mass", "age")],
    "breast_cancer_wisconsin": [
        ("mean radius", "mean perimeter"),
        ("mean area", "mean concavity"),
    ],
    "heart-statlog": [
        ("chest", "maximum_heart_rate_achieved"),
        ("age", "oldpeak"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Information-theoretic helpers (manual MI-based PID)
# ═══════════════════════════════════════════════════════════════════════════════

def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy from an array of counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def _joint_counts_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build joint count matrix for two discrete arrays."""
    x_vals = np.unique(x)
    y_vals = np.unique(y)
    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}
    mat = np.zeros((len(x_vals), len(y_vals)), dtype=np.int64)
    for xi, yi in zip(x, y):
        mat[x_map[xi], y_map[yi]] += 1
    return mat


def _mi(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information I(X;Y) in bits."""
    joint = _joint_counts_1d(x, y)
    h_x = _entropy(joint.sum(axis=1))
    h_y = _entropy(joint.sum(axis=0))
    h_xy = _entropy(joint.ravel())
    return max(0.0, h_x + h_y - h_xy)


def _joint_counts_3d(fi: np.ndarray, fj: np.ndarray, y: np.ndarray):
    """Build 3D count array for (Fi, Fj, Y)."""
    fi_vals = np.unique(fi)
    fj_vals = np.unique(fj)
    y_vals = np.unique(y)
    fi_map = {v: i for i, v in enumerate(fi_vals)}
    fj_map = {v: i for i, v in enumerate(fj_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}
    cube = np.zeros((len(fi_vals), len(fj_vals), len(y_vals)), dtype=np.int64)
    for a, b, c in zip(fi, fj, y):
        cube[fi_map[a], fj_map[b], y_map[c]] += 1
    return cube


def _cond_mi(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Conditional mutual information I(X;Y|Z) in bits using chain rule."""
    # I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
    yz = np.array(list(zip(y, z)))
    # Use tuple hashing
    yz_keys = np.array([hash((a, b)) for a, b in zip(y, z)])
    return max(0.0, _mi(x, yz_keys) - _mi(x, z))


def compute_pid_manual(
    fi: np.ndarray, fj: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Compute PID (Williams-Beer I_min) for bivariate case.

    For two sources, I_min redundancy equals min(I(Fi;Y), I(Fj;Y)).
    This is exact for the bivariate case.

    Returns dict with synergy, unique_i, unique_j, redundancy, joint_mi.
    """
    mi_i = _mi(fi, y)
    mi_j = _mi(fj, y)

    # Joint MI: I(Fi,Fj ; Y) = H(Y) - H(Y|Fi,Fj)
    fij_keys = np.array([hash((a, b)) for a, b in zip(fi, fj)])
    mi_joint = _mi(fij_keys, y)

    # Williams-Beer I_min redundancy
    redundancy = min(mi_i, mi_j)

    # Unique information
    unique_i = mi_i - redundancy
    unique_j = mi_j - redundancy

    # Synergy: what's left after accounting for unique + redundancy
    synergy = mi_joint - unique_i - unique_j - redundancy

    return {
        "synergy": synergy,
        "unique_i": unique_i,
        "unique_j": unique_j,
        "redundancy": redundancy,
        "joint_mi": mi_joint,
        "mi_i": mi_i,
        "mi_j": mi_j,
    }


# ── Baseline: simple MI interaction score ────────────────────────────────────

def compute_mi_interaction(
    fi: np.ndarray, fj: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Baseline interaction score: I(Fi,Fj;Y) - I(Fi;Y) - I(Fj;Y).

    This is the 'interaction information' / 'co-information', a simpler
    measure that doesn't decompose into synergy vs redundancy.
    """
    mi_i = _mi(fi, y)
    mi_j = _mi(fj, y)
    fij_keys = np.array([hash((a, b)) for a, b in zip(fi, fj)])
    mi_joint = _mi(fij_keys, y)
    interaction = mi_joint - mi_i - mi_j
    return {
        "interaction_score": interaction,
        "mi_i": mi_i,
        "mi_j": mi_j,
        "joint_mi": mi_joint,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_datasets() -> dict:
    """Load all 12 datasets from the 3 dependency artifacts."""
    datasets = {}
    sources = [
        (DATA_DIR1, "data_id1"),
        (DATA_DIR2, "data_id2"),
        (DATA_DIR3, "data_id3"),
    ]

    for data_dir, source_id in sources:
        fpath = data_dir / "full_data_out.json"
        logger.info(f"{BLUE}Loading{END} {fpath.name} from {source_id}")
        raw = json.loads(fpath.read_text())

        for ds_block in raw["datasets"]:
            ds_name = ds_block["dataset"]
            examples = ds_block["examples"]
            if not examples:
                logger.warning(f"{YELLOW}Empty dataset: {ds_name}{END}")
                continue

            first = examples[0]
            feature_names = first.get("metadata_feature_names", [])
            n_features = first.get("metadata_n_features", len(feature_names))
            n_classes = first.get("metadata_n_classes", 2)
            feature_types = first.get("metadata_feature_types", None)

            # Parse all examples into arrays
            all_inputs = []
            all_labels = []
            # Pre-computed discretized values (data_id1 only)
            disc_5bin_avail = "metadata_discretized_5bin" in first
            disc_10bin_avail = "metadata_discretized_10bin" in first
            all_disc_5 = [] if disc_5bin_avail else None
            all_disc_10 = [] if disc_10bin_avail else None

            for ex in examples:
                inp = json.loads(ex["input"])
                vals = [inp[fn] for fn in feature_names]
                all_inputs.append(vals)
                all_labels.append(ex["output"])
                if disc_5bin_avail:
                    all_disc_5.append(ex["metadata_discretized_5bin"])
                if disc_10bin_avail:
                    all_disc_10.append(ex["metadata_discretized_10bin"])

            X = np.array(all_inputs, dtype=np.float64)
            # Encode labels as integers
            label_set = sorted(set(all_labels))
            label_map = {l: i for i, l in enumerate(label_set)}
            y = np.array([label_map[l] for l in all_labels], dtype=np.int64)

            ds_info = {
                "X": X,
                "y": y,
                "feature_names": feature_names,
                "feature_types": feature_types,
                "n_classes": n_classes,
                "n_features": len(feature_names),
                "n_samples": len(examples),
                "source_artifact": source_id,
                "label_names": label_set,
            }
            if all_disc_5 is not None:
                ds_info["precomputed_disc_5bin"] = np.array(all_disc_5, dtype=np.int64)
            if all_disc_10 is not None:
                ds_info["precomputed_disc_10bin"] = np.array(all_disc_10, dtype=np.int64)

            datasets[ds_name] = ds_info
            logger.info(
                f"  {GREEN}{ds_name}{END}: {X.shape[0]} samples, "
                f"{X.shape[1]} features, {n_classes} classes"
            )

    logger.info(f"{GREEN}Loaded {len(datasets)} datasets{END}")
    return datasets


# ═══════════════════════════════════════════════════════════════════════════════
#  Discretization
# ═══════════════════════════════════════════════════════════════════════════════

def discretize_dataset(
    ds_info: dict, n_bins: int
) -> np.ndarray:
    """Discretize continuous features. Categorical features kept as-is."""
    # Check for pre-computed discretization
    key = f"precomputed_disc_{n_bins}bin"
    if key in ds_info:
        logger.debug(f"  Using pre-computed {n_bins}-bin discretization")
        return ds_info[key]

    X = ds_info["X"]
    feature_types = ds_info["feature_types"]
    n_samples, n_features = X.shape

    X_disc = np.zeros_like(X, dtype=np.int64)
    for f_idx in range(n_features):
        col = X[:, f_idx]
        is_categorical = (
            feature_types is not None and feature_types[f_idx] == "categorical"
        )

        if is_categorical:
            # Already ordinal-encoded integers, use directly
            # If too many unique values, bin them
            unique_vals = np.unique(col)
            if len(unique_vals) > n_bins:
                # Bin by quantile
                kbd = KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy="quantile"
                )
                X_disc[:, f_idx] = kbd.fit_transform(col.reshape(-1, 1)).ravel().astype(np.int64)
            else:
                X_disc[:, f_idx] = col.astype(np.int64)
        else:
            # Continuous: quantile binning
            n_unique = len(np.unique(col))
            actual_bins = min(n_bins, n_unique)
            if actual_bins < 2:
                X_disc[:, f_idx] = 0
            else:
                kbd = KBinsDiscretizer(
                    n_bins=actual_bins, encode="ordinal", strategy="quantile"
                )
                X_disc[:, f_idx] = kbd.fit_transform(col.reshape(-1, 1)).ravel().astype(np.int64)

    return X_disc


# ═══════════════════════════════════════════════════════════════════════════════
#  Pairwise PID computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_pairwise_pid(
    X_disc: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    dataset_name: str,
    n_bins: int,
) -> list[dict]:
    """Compute PID for all feature pairs in a dataset."""
    n_features = X_disc.shape[1]
    pairs = list(combinations(range(n_features), 2))
    results = []
    t_start = time.time()

    for pair_idx, (i, j) in enumerate(pairs):
        elapsed = time.time() - t_start
        if elapsed > PER_DATASET_TIMEOUT_S:
            logger.warning(
                f"{YELLOW}Timeout after {elapsed:.0f}s on {dataset_name} "
                f"({pair_idx}/{len(pairs)} pairs computed){END}"
            )
            break

        fi = X_disc[:, i]
        fj = X_disc[:, j]

        # PID (our method)
        pid = compute_pid_manual(fi=fi, fj=fj, y=y)

        # Baseline (interaction information)
        baseline = compute_mi_interaction(fi=fi, fj=fj, y=y)

        results.append({
            "feature_i": feature_names[i],
            "feature_j": feature_names[j],
            "feature_idx_i": int(i),
            "feature_idx_j": int(j),
            "synergy": float(pid["synergy"]),
            "unique_i": float(pid["unique_i"]),
            "unique_j": float(pid["unique_j"]),
            "redundancy": float(pid["redundancy"]),
            "joint_mi": float(pid["joint_mi"]),
            "mi_i": float(pid["mi_i"]),
            "mi_j": float(pid["mi_j"]),
            "baseline_interaction": float(baseline["interaction_score"]),
            "n_bins": n_bins,
        })

    elapsed = time.time() - t_start
    logger.info(
        f"  {n_bins}-bin: {len(results)}/{len(pairs)} pairs in {elapsed:.1f}s"
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Graph construction & analysis
# ═══════════════════════════════════════════════════════════════════════════════

def build_synergy_graph(
    pid_results: list[dict],
    feature_names: list[str],
    threshold_quantile: float,
) -> tuple[nx.Graph, float]:
    """Build a synergy graph with edges above the given quantile threshold."""
    synergy_values = [r["synergy"] for r in pid_results]
    if not synergy_values:
        return nx.Graph(), 0.0

    threshold_value = float(np.quantile(synergy_values, threshold_quantile))

    G = nx.Graph()
    G.add_nodes_from(feature_names)
    for r in pid_results:
        if r["synergy"] >= threshold_value:
            G.add_edge(r["feature_i"], r["feature_j"], weight=r["synergy"])

    return G, threshold_value


def analyze_graph(G: nx.Graph) -> dict:
    """Compute graph statistics."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    nodes_with_edges = sum(1 for n in G.nodes() if G.degree(n) > 0)

    if n_nodes < 2:
        return {
            "n_edges": n_edges,
            "n_nodes_with_edges": nodes_with_edges,
            "density": 0.0,
            "n_connected_components": 0,
            "largest_component_size": 0,
            "largest_clique_size": 0,
            "mean_degree": 0.0,
            "max_degree": 0,
            "clustering_coefficient": 0.0,
        }

    max_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / max_edges if max_edges > 0 else 0.0

    components = list(nx.connected_components(G))
    component_sizes = [len(c) for c in components]

    cliques = list(nx.find_cliques(G)) if n_edges > 0 else []
    clique_sizes = [len(c) for c in cliques] if cliques else [0]

    degrees = [d for _, d in G.degree()]
    mean_degree = float(np.mean(degrees)) if degrees else 0.0
    max_degree = max(degrees) if degrees else 0

    cc = nx.average_clustering(G) if n_edges > 0 else 0.0

    return {
        "n_edges": n_edges,
        "n_nodes_with_edges": nodes_with_edges,
        "density": round(density, 4),
        "n_connected_components": len(components),
        "largest_component_size": max(component_sizes) if component_sizes else 0,
        "largest_clique_size": max(clique_sizes),
        "mean_degree": round(mean_degree, 2),
        "max_degree": max_degree,
        "clustering_coefficient": round(cc, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Discretization stability analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stability(
    results_5bin: list[dict],
    results_10bin: list[dict],
) -> dict:
    """Compare synergy rankings between 5-bin and 10-bin discretizations."""
    # Build pair -> synergy maps
    key_fn = lambda r: (r["feature_i"], r["feature_j"])
    syn_5 = {key_fn(r): r["synergy"] for r in results_5bin}
    syn_10 = {key_fn(r): r["synergy"] for r in results_10bin}

    # Common pairs
    common_keys = sorted(set(syn_5.keys()) & set(syn_10.keys()))
    if len(common_keys) < 3:
        return {
            "n_common_pairs": len(common_keys),
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
            "jaccard_top10pct": 0.0,
            "jaccard_top25pct": 0.0,
            "jaccard_median": 0.0,
        }

    vec_5 = np.array([syn_5[k] for k in common_keys])
    vec_10 = np.array([syn_10[k] for k in common_keys])

    # Handle constant arrays
    if np.std(vec_5) < 1e-12 or np.std(vec_10) < 1e-12:
        pearson_r = 0.0
        spearman_rho = 0.0
    else:
        pearson_r = float(pearsonr(vec_5, vec_10)[0])
        spearman_rho = float(spearmanr(vec_5, vec_10)[0])

    # Jaccard similarity of edge sets at different thresholds
    def jaccard_at_q(q: float) -> float:
        t5 = np.quantile(vec_5, q)
        t10 = np.quantile(vec_10, q)
        set_5 = {k for k, v in zip(common_keys, vec_5) if v >= t5}
        set_10 = {k for k, v in zip(common_keys, vec_10) if v >= t10}
        if not set_5 and not set_10:
            return 1.0
        inter = len(set_5 & set_10)
        union = len(set_5 | set_10)
        return inter / union if union > 0 else 0.0

    return {
        "n_common_pairs": len(common_keys),
        "pearson_r": round(pearson_r, 4),
        "spearman_rho": round(spearman_rho, 4),
        "jaccard_top10pct": round(jaccard_at_q(0.90), 4),
        "jaccard_top25pct": round(jaccard_at_q(0.75), 4),
        "jaccard_median": round(jaccard_at_q(0.50), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain-meaningful interaction check
# ═══════════════════════════════════════════════════════════════════════════════

def check_domain_pairs(
    pid_results: list[dict], dataset_name: str
) -> list[dict]:
    """Check how known-meaningful pairs rank in synergy ordering."""
    if dataset_name not in DOMAIN_PAIRS:
        return []

    # Sort by synergy descending
    sorted_results = sorted(pid_results, key=lambda r: r["synergy"], reverse=True)
    pair_to_rank = {}
    for rank, r in enumerate(sorted_results, 1):
        key = (r["feature_i"], r["feature_j"])
        pair_to_rank[key] = rank
        # Also store reverse
        pair_to_rank[(r["feature_j"], r["feature_i"])] = rank

    checks = []
    total_pairs = len(sorted_results)
    for fi, fj in DOMAIN_PAIRS[dataset_name]:
        rank = pair_to_rank.get((fi, fj), None)
        if rank is None:
            rank = pair_to_rank.get((fj, fi), None)
        checks.append({
            "feature_i": fi,
            "feature_j": fj,
            "synergy_rank": rank,
            "total_pairs": total_pairs,
            "in_top_10pct": rank is not None and rank <= max(1, int(total_pairs * 0.1)),
            "in_top_25pct": rank is not None and rank <= max(1, int(total_pairs * 0.25)),
        })

    return checks


# ═══════════════════════════════════════════════════════════════════════════════
#  Output generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_output(all_results: dict) -> dict:
    """Generate output in exp_gen_sol_out schema format.

    Schema: {"datasets": [{"dataset": str, "examples": [...]}]}
    Each example has: input, output, predict_*, metadata_*
    """
    output = {"datasets": []}

    for ds_name, ds_data in all_results.items():
        examples = []
        # Use 5-bin results as primary (more stable, fewer bins = less noise)
        pid_results_5 = ds_data.get("pid_5bin", [])
        pid_results_10 = ds_data.get("pid_10bin", [])
        graph_stats = ds_data.get("graph_stats", {})
        stability = ds_data.get("stability", {})
        domain_checks = ds_data.get("domain_checks", [])
        ds_info = ds_data["ds_info"]

        # Sort by synergy descending for ranking
        sorted_5 = sorted(pid_results_5, key=lambda r: r["synergy"], reverse=True)
        pair_to_rank = {}
        for rank, r in enumerate(sorted_5, 1):
            key = (r["feature_i"], r["feature_j"])
            pair_to_rank[key] = rank

        # Build edge sets for each threshold
        edge_sets = {}
        for thr_name, thr_q in THRESHOLDS.items():
            if sorted_5:
                thr_val = np.quantile([r["synergy"] for r in sorted_5], thr_q)
                edge_sets[thr_name] = {
                    (r["feature_i"], r["feature_j"])
                    for r in sorted_5 if r["synergy"] >= thr_val
                }
            else:
                edge_sets[thr_name] = set()

        for r in pid_results_5:
            pair_key = (r["feature_i"], r["feature_j"])
            rank = pair_to_rank.get(pair_key, len(sorted_5))

            input_dict = {
                "feature_i": r["feature_i"],
                "feature_j": r["feature_j"],
                "dataset": ds_name,
            }

            output_str = (
                f"synergy={r['synergy']:.4f},"
                f"redundancy={r['redundancy']:.4f},"
                f"unique_i={r['unique_i']:.4f},"
                f"unique_j={r['unique_j']:.4f},"
                f"joint_mi={r['joint_mi']:.4f},"
                f"baseline_interaction={r['baseline_interaction']:.4f}"
            )

            example = {
                "input": json.dumps(input_dict),
                "output": output_str,
                "predict_synergy_rank": str(rank),
                "predict_graph_edge_top_10pct": str(pair_key in edge_sets.get("top_10pct", set())).lower(),
                "predict_graph_edge_top_25pct": str(pair_key in edge_sets.get("top_25pct", set())).lower(),
                "predict_graph_edge_above_median": str(pair_key in edge_sets.get("above_median", set())).lower(),
                "predict_baseline_interaction": f"{r['baseline_interaction']:.4f}",
                "metadata_dataset": ds_name,
                "metadata_n_features": ds_info["n_features"],
                "metadata_n_samples": ds_info["n_samples"],
                "metadata_n_classes": ds_info["n_classes"],
                "metadata_bin_level": 5,
                "metadata_synergy": round(r["synergy"], 6),
                "metadata_redundancy": round(r["redundancy"], 6),
                "metadata_unique_i": round(r["unique_i"], 6),
                "metadata_unique_j": round(r["unique_j"], 6),
                "metadata_joint_mi": round(r["joint_mi"], 6),
                "metadata_mi_i": round(r["mi_i"], 6),
                "metadata_mi_j": round(r["mi_j"], 6),
                "metadata_baseline_interaction": round(r["baseline_interaction"], 6),
                "metadata_source_artifact": ds_info["source_artifact"],
            }

            # Add stability info if available
            if stability:
                example["metadata_stability_spearman"] = stability.get("spearman_rho", 0.0)
                example["metadata_stability_pearson"] = stability.get("pearson_r", 0.0)

            examples.append(example)

        if examples:
            output["datasets"].append({
                "dataset": ds_name,
                "examples": examples,
            })

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    global_start = time.time()
    logger.info(f"{BLUE}=== PID Synergy Graph Construction ==={END}")

    # ── Phase 1: Load datasets ───────────────────────────────────────────────
    logger.info(f"{BLUE}Phase 1: Loading datasets{END}")
    datasets = load_datasets()
    logger.info(f"Loaded {len(datasets)} datasets in {time.time()-global_start:.1f}s")

    # Sort datasets by number of feature pairs (ascending) for gradual scaling
    ds_order = sorted(
        datasets.keys(),
        key=lambda name: len(list(combinations(range(datasets[name]["n_features"]), 2)))
    )
    logger.info(f"Processing order (by pair count):")
    for name in ds_order:
        nf = datasets[name]["n_features"]
        n_pairs = nf * (nf - 1) // 2
        logger.info(f"  {name}: {nf} features -> {n_pairs} pairs")

    # ── Phase 2 & 3: Discretization + PID computation ────────────────────────
    all_results = {}

    for ds_idx, ds_name in enumerate(ds_order):
        ds_start = time.time()
        ds_info = datasets[ds_name]
        nf = ds_info["n_features"]
        n_pairs = nf * (nf - 1) // 2
        logger.info(
            f"\n{BLUE}[{ds_idx+1}/{len(ds_order)}] {ds_name}{END}: "
            f"{nf} features, {n_pairs} pairs, {ds_info['n_samples']} samples"
        )

        ds_results = {"ds_info": ds_info}

        for n_bins in BIN_LEVELS:
            # Discretize
            logger.info(f"  Discretizing ({n_bins} bins)...")
            X_disc = discretize_dataset(ds_info=ds_info, n_bins=n_bins)

            # Compute PID
            logger.info(f"  Computing PID ({n_bins} bins, {n_pairs} pairs)...")
            pid_results = compute_all_pairwise_pid(
                X_disc=X_disc,
                y=ds_info["y"],
                feature_names=ds_info["feature_names"],
                dataset_name=ds_name,
                n_bins=n_bins,
            )
            ds_results[f"pid_{n_bins}bin"] = pid_results

        # ── Phase 4: Graph construction ──────────────────────────────────────
        logger.info(f"  Building synergy graphs...")
        graph_stats = {}
        for thr_name, thr_q in THRESHOLDS.items():
            for n_bins in BIN_LEVELS:
                pid_key = f"pid_{n_bins}bin"
                if pid_key not in ds_results or not ds_results[pid_key]:
                    continue
                G, thr_val = build_synergy_graph(
                    pid_results=ds_results[pid_key],
                    feature_names=ds_info["feature_names"],
                    threshold_quantile=thr_q,
                )
                stats = analyze_graph(G)
                stats["threshold_value"] = round(thr_val, 6)
                graph_stats[f"{thr_name}_{n_bins}bin"] = stats
        ds_results["graph_stats"] = graph_stats

        # ── Phase 5: Stability analysis ──────────────────────────────────────
        if ds_results.get("pid_5bin") and ds_results.get("pid_10bin"):
            logger.info(f"  Computing discretization stability...")
            stability = compute_stability(
                results_5bin=ds_results["pid_5bin"],
                results_10bin=ds_results["pid_10bin"],
            )
            ds_results["stability"] = stability
            logger.info(
                f"  Stability: Spearman={stability['spearman_rho']:.3f}, "
                f"Jaccard(top25%)={stability['jaccard_top25pct']:.3f}"
            )

        # ── Phase 6: Domain checks ───────────────────────────────────────────
        if ds_name in DOMAIN_PAIRS and ds_results.get("pid_5bin"):
            logger.info(f"  Checking domain-meaningful interactions...")
            domain_checks = check_domain_pairs(
                pid_results=ds_results["pid_5bin"],
                dataset_name=ds_name,
            )
            ds_results["domain_checks"] = domain_checks
            for dc in domain_checks:
                logger.info(
                    f"    {dc['feature_i']}-{dc['feature_j']}: "
                    f"rank {dc['synergy_rank']}/{dc['total_pairs']} "
                    f"(top-10%: {dc['in_top_10pct']}, top-25%: {dc['in_top_25pct']})"
                )

        all_results[ds_name] = ds_results
        ds_elapsed = time.time() - ds_start
        logger.info(
            f"  {GREEN}Completed {ds_name} in {ds_elapsed:.1f}s{END}"
        )

        # Check total runtime
        total_elapsed = time.time() - global_start
        if total_elapsed > 3000:  # 50 min safety cutoff
            logger.warning(
                f"{YELLOW}Approaching time limit ({total_elapsed:.0f}s), "
                f"stopping after {ds_idx+1}/{len(ds_order)} datasets{END}"
            )
            break

    # ── Phase 7: Generate output ─────────────────────────────────────────────
    logger.info(f"\n{BLUE}Phase 7: Generating output{END}")
    output = generate_output(all_results)

    # Count total examples
    total_examples = sum(len(d["examples"]) for d in output["datasets"])
    logger.info(f"Output: {len(output['datasets'])} datasets, {total_examples} examples")

    # Save output
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path}")

    # ── Summary CSV ──────────────────────────────────────────────────────────
    summary_lines = [
        "dataset,n_features,n_samples,n_pairs_5bin,n_pairs_10bin,"
        "mean_synergy_5bin,max_synergy_5bin,"
        "mean_baseline_5bin,max_baseline_5bin,"
        "spearman_stability,jaccard25_stability,"
        "graph_edges_top10pct_5bin,graph_edges_top25pct_5bin,"
        "graph_density_top25pct_5bin,largest_clique_top25pct_5bin"
    ]
    for ds_name in ds_order:
        if ds_name not in all_results:
            continue
        dr = all_results[ds_name]
        di = dr["ds_info"]
        p5 = dr.get("pid_5bin", [])
        p10 = dr.get("pid_10bin", [])
        stab = dr.get("stability", {})
        gs = dr.get("graph_stats", {})

        syn_5 = [r["synergy"] for r in p5] if p5 else [0]
        bl_5 = [r["baseline_interaction"] for r in p5] if p5 else [0]

        gs_t10 = gs.get("top_10pct_5bin", {})
        gs_t25 = gs.get("top_25pct_5bin", {})

        summary_lines.append(
            f"{ds_name},{di['n_features']},{di['n_samples']},"
            f"{len(p5)},{len(p10)},"
            f"{np.mean(syn_5):.4f},{np.max(syn_5):.4f},"
            f"{np.mean(bl_5):.4f},{np.max(bl_5):.4f},"
            f"{stab.get('spearman_rho', 'N/A')},{stab.get('jaccard_top25pct', 'N/A')},"
            f"{gs_t10.get('n_edges', 'N/A')},{gs_t25.get('n_edges', 'N/A')},"
            f"{gs_t25.get('density', 'N/A')},{gs_t25.get('largest_clique_size', 'N/A')}"
        )

    csv_path = WORKSPACE / "summary.csv"
    csv_path.write_text("\n".join(summary_lines))
    logger.info(f"Saved summary CSV to {csv_path}")

    total_time = time.time() - global_start
    logger.info(f"\n{GREEN}=== DONE in {total_time:.1f}s ==={END}")


if __name__ == "__main__":
    main()
