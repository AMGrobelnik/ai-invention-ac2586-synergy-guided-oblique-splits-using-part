#!/usr/bin/env python3
"""SG-FIGS Comprehensive Statistical Evaluation.

Seven-block evaluation of SG-FIGS experiment results:
  Block 1: Per-dataset accuracy/AUC tables with Bonferroni-corrected paired t-tests
  Block 2: Critical difference diagram data (Friedman + Nemenyi)
  Block 3: Dataset meta-analysis (Spearman correlations with dataset properties)
  Block 4: Oblique split activation analysis
  Block 5: Accuracy-at-matched-complexity curves
  Block 6: Ablation decomposition (oblique penalty vs synergy effect)
  Block 7: Positive case study narratives
"""

import json
import math
import resource
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats

# ---------------------------------------------------------------------------
# Resource limits (14 GB RAM, 1 h CPU)
# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
ITER4_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/"
    "run__20260212_072136/3_invention_loop/iter_4/gen_art/exp_id1_it4__opus"
)
ITER2_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/"
    "run__20260212_072136/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus"
)

# Dataset name mapping iter2 -> iter4
DATASET_NAME_MAP_2TO4 = {
    "banknote-authentication": "banknote",
    "diabetes": "diabetes",
    "glass": "glass",
    "wine": "wine",
    "heart-statlog": "heart_statlog",
    "australian": "australian",
    "vehicle": "vehicle",
    "segment": "segment",
    "credit_g": "credit_g",
    "breast_cancer_wisconsin": "breast_cancer",
    "ionosphere": "ionosphere",
    "sonar": "sonar",
}
DATASET_NAME_MAP_4TO2 = {v: k for k, v in DATASET_NAME_MAP_2TO4.items()}

METHODS = ["FIGS", "RO-FIGS", "SG-FIGS-10", "SG-FIGS-25", "SG-FIGS-50", "GradientBoosting"]
MAX_RULES_VALUES = [5, 10, 15]
N_FOLDS = 5
ALPHA = 0.05
N_METHODS = len(METHODS)
N_PAIRS = N_METHODS * (N_METHODS - 1) // 2  # 15
BONFERRONI_ALPHA = ALPHA / N_PAIRS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val):
    """Convert to float, handling NaN / None."""
    if val is None:
        return float("nan")
    f = float(val)
    return f


def _load_json(path: Path) -> dict:
    logger.info(f"Loading {path.name} from {path.parent}")
    data = json.loads(path.read_text())
    return data


def _bonferroni_sig(p: float) -> str:
    """Return significance indicator under Bonferroni correction."""
    if p < 0.001 / N_PAIRS:
        return "***"
    if p < 0.01 / N_PAIRS:
        return "**"
    if p < BONFERRONI_ALPHA:
        return "*"
    return "ns"


def _paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided paired t-test; returns (t_stat, p_value).

    Falls back to (0.0, 1.0) when variance is zero.
    """
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    diff = a_arr - b_arr
    if np.std(diff, ddof=1) == 0:
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_rel(a_arr, b_arr)
    return float(t_stat), float(p_val)


def _rank_methods_per_dataset(
    method_means: dict[str, float],
) -> dict[str, float]:
    """Rank methods (1 = best) by accuracy; ties get average rank."""
    sorted_methods = sorted(method_means.items(), key=lambda x: -x[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(sorted_methods):
        j = i + 1
        while j < len(sorted_methods) and np.isclose(sorted_methods[j][1], sorted_methods[i][1]):
            j += 1
        avg_rank = np.mean(list(range(i + 1, j + 1)))
        for k in range(i, j):
            ranks[sorted_methods[k][0]] = float(avg_rank)
        i = j
    return ranks


def _nemenyi_q_alpha(k: int, alpha: float = 0.05) -> float:
    """Critical value for Nemenyi test (Studentized range / sqrt(2)).

    Hardcoded for k=6, alpha=0.05 (from standard tables).
    """
    # q_{0.05}(6, inf) = 4.030  (Studentized range table)
    # Nemenyi uses q / sqrt(2)
    if k == 6 and alpha == 0.05:
        return 4.030 / math.sqrt(2)  # ≈ 2.850
    raise ValueError(f"Nemenyi q not hardcoded for k={k}, alpha={alpha}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_iter4_data(path: Path) -> dict:
    """Load iter4 experiment data into a structured dict.

    Returns:
        {dataset: {method: {max_rules: {fold: {metric: val, ...}}}}}
    """
    raw = _load_json(path)
    data: dict = {}
    for ds_block in raw["datasets"]:
        ds_name = ds_block["dataset"]
        data[ds_name] = {}
        for ex in ds_block["examples"]:
            method = ex["metadata_method"]
            mr = ex["metadata_max_rules"]
            fold = ex["metadata_fold"]
            data.setdefault(ds_name, {}).setdefault(method, {}).setdefault(mr, {})[fold] = {
                "accuracy": _safe_float(ex["metadata_accuracy"]),
                "auc": _safe_float(ex["metadata_auc"]),
                "n_splits": ex.get("metadata_n_splits", 0),
                "n_oblique": ex.get("metadata_n_oblique", 0),
                "oblique_fraction": _safe_float(ex.get("metadata_oblique_fraction", 0)),
                "mean_features_per_oblique": _safe_float(ex.get("metadata_mean_features_per_oblique", 0)),
                "interpretability_score": _safe_float(ex.get("metadata_interpretability_score", float("nan"))),
                "synergy_time_s": _safe_float(ex.get("metadata_synergy_time_s", 0)),
                "n_synergy_edges": ex.get("metadata_n_synergy_edges", 0),
                "synergy_cutoff": _safe_float(ex.get("metadata_synergy_cutoff", 0)),
            }
    return data


def load_iter2_data(path: Path) -> dict:
    """Load iter2 synergy graph data.

    Returns:
        {dataset_iter4_name: {
            'n_features': int, 'n_samples': int, 'n_classes': int,
            'pairs': [{'fi': str, 'fj': str, 'synergy': float, ...}],
            'synergy_graph_density_25pct': float,
            'mean_synergy_score': float,
            'mean_feature_redundancy': float,
            'stability_spearman': float,
            'stability_pearson': float,
        }}
    """
    raw = _load_json(path)
    result: dict = {}
    for ds_block in raw["datasets"]:
        ds_name_iter2 = ds_block["dataset"]
        ds_name = DATASET_NAME_MAP_2TO4.get(ds_name_iter2, ds_name_iter2)
        examples = ds_block["examples"]
        if not examples:
            continue
        first = examples[0]
        n_features = first["metadata_n_features"]
        n_samples = first["metadata_n_samples"]
        n_classes = first["metadata_n_classes"]
        n_total_pairs = len(examples)
        # Compute graph densities
        top25_count = sum(1 for e in examples if e.get("predict_graph_edge_top_25pct") == "true")
        top10_count = sum(1 for e in examples if e.get("predict_graph_edge_top_10pct") == "true")
        above_median_count = sum(1 for e in examples if e.get("predict_graph_edge_above_median") == "true")
        # Synergy scores
        synergies = [_safe_float(e["metadata_synergy"]) for e in examples]
        redundancies = [_safe_float(e["metadata_redundancy"]) for e in examples]
        # Maximum possible edges for undirected graph: n*(n-1)/2
        max_possible_edges = n_features * (n_features - 1) / 2
        density_25 = top25_count / max_possible_edges if max_possible_edges > 0 else 0.0
        density_10 = top10_count / max_possible_edges if max_possible_edges > 0 else 0.0
        result[ds_name] = {
            "n_features": n_features,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "n_total_pairs": n_total_pairs,
            "top25_edges": top25_count,
            "top10_edges": top10_count,
            "above_median_edges": above_median_count,
            "synergy_graph_density_25pct": density_25,
            "synergy_graph_density_10pct": density_10,
            "mean_synergy_score": float(np.mean(synergies)),
            "std_synergy_score": float(np.std(synergies, ddof=1)) if len(synergies) > 1 else 0.0,
            "mean_feature_redundancy": float(np.mean(redundancies)),
            "stability_spearman": _safe_float(first.get("metadata_stability_spearman", float("nan"))),
            "stability_pearson": _safe_float(first.get("metadata_stability_pearson", float("nan"))),
            "pairs": [
                {
                    "fi": json.loads(e["input"])["feature_i"],
                    "fj": json.loads(e["input"])["feature_j"],
                    "synergy": _safe_float(e["metadata_synergy"]),
                    "redundancy": _safe_float(e["metadata_redundancy"]),
                }
                for e in examples
            ],
        }
    return result


def load_analysis_metadata(path: Path) -> dict:
    return _load_json(path)


# ---------------------------------------------------------------------------
# Block 1: Per-Dataset Accuracy/AUC Tables
# ---------------------------------------------------------------------------

def block1_per_dataset_accuracy(
    data: dict,
    datasets: list[str],
    max_rules: int = 10,
) -> dict:
    """For each dataset at max_rules=10, compute mean±std accuracy and AUC,
    and run Bonferroni-corrected paired t-tests across all 15 method pairs."""
    logger.info("Block 1: Per-dataset accuracy/AUC tables")
    results = {}
    for ds in datasets:
        ds_data = data.get(ds, {})
        method_accs: dict[str, list[float]] = {}
        method_aucs: dict[str, list[float]] = {}
        for method in METHODS:
            accs = []
            aucs = []
            mr_data = ds_data.get(method, {}).get(max_rules, {})
            for fold in range(N_FOLDS):
                fold_data = mr_data.get(fold, {})
                accs.append(fold_data.get("accuracy", float("nan")))
                aucs.append(fold_data.get("auc", float("nan")))
            method_accs[method] = accs
            method_aucs[method] = aucs

        # Method summaries
        method_stats = {}
        for method in METHODS:
            accs = np.array(method_accs[method])
            aucs = np.array(method_aucs[method])
            method_stats[method] = {
                "accuracy_mean": float(np.nanmean(accs)),
                "accuracy_std": float(np.nanstd(accs, ddof=1)),
                "auc_mean": float(np.nanmean(aucs)),
                "auc_std": float(np.nanstd(aucs, ddof=1)),
            }

        # Pairwise t-tests (Bonferroni)
        pairwise = {}
        method_pairs = list(combinations(METHODS, 2))
        for m1, m2 in method_pairs:
            t_stat, p_val = _paired_ttest(method_accs[m1], method_accs[m2])
            p_corrected = min(p_val * N_PAIRS, 1.0)
            sig = _bonferroni_sig(p_val)
            pairwise[f"{m1}_vs_{m2}"] = {
                "t_statistic": t_stat,
                "p_value_raw": p_val,
                "p_value_corrected": p_corrected,
                "significant": sig,
                "mean_diff": method_stats[m1]["accuracy_mean"] - method_stats[m2]["accuracy_mean"],
            }

        # Rank methods for this dataset
        means_dict = {m: method_stats[m]["accuracy_mean"] for m in METHODS}
        ranks = _rank_methods_per_dataset(means_dict)

        # Delta vs FIGS
        figs_mean = method_stats["FIGS"]["accuracy_mean"]

        results[ds] = {
            "method_stats": method_stats,
            "pairwise_tests": pairwise,
            "ranks": ranks,
            "delta_vs_figs": {m: method_stats[m]["accuracy_mean"] - figs_mean for m in METHODS},
        }
        logger.debug(f"  {ds}: FIGS={method_stats['FIGS']['accuracy_mean']:.4f}, "
                      f"SG-FIGS-25={method_stats['SG-FIGS-25']['accuracy_mean']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Block 2: Critical Difference Diagram Data
# ---------------------------------------------------------------------------

def block2_critical_difference(
    data: dict,
    datasets: list[str],
    max_rules: int = 10,
) -> dict:
    """Friedman test + Nemenyi CD computation."""
    logger.info("Block 2: Critical difference diagram data")
    # Build matrix: datasets x methods
    n_datasets = len(datasets)
    rank_matrix = np.zeros((n_datasets, N_METHODS))
    mean_acc_matrix = np.zeros((n_datasets, N_METHODS))

    for i, ds in enumerate(datasets):
        ds_data = data.get(ds, {})
        means = {}
        for j, method in enumerate(METHODS):
            mr_data = ds_data.get(method, {}).get(max_rules, {})
            accs = [mr_data.get(fold, {}).get("accuracy", float("nan")) for fold in range(N_FOLDS)]
            mean_acc = float(np.nanmean(accs))
            means[method] = mean_acc
            mean_acc_matrix[i, j] = mean_acc

        ranks = _rank_methods_per_dataset(means)
        for j, method in enumerate(METHODS):
            rank_matrix[i, j] = ranks[method]

    # Friedman test
    chi2, p_friedman = stats.friedmanchisquare(*[rank_matrix[:, j] for j in range(N_METHODS)])

    # Mean ranks
    mean_ranks = {METHODS[j]: float(np.mean(rank_matrix[:, j])) for j in range(N_METHODS)}

    # Nemenyi CD
    k = N_METHODS
    n = n_datasets
    q_alpha = _nemenyi_q_alpha(k=k, alpha=0.05)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n))

    # Identify cliques: groups of methods whose pairwise rank diff < CD
    sorted_methods = sorted(mean_ranks.items(), key=lambda x: x[1])
    cliques = []
    for i in range(len(sorted_methods)):
        clique = [sorted_methods[i][0]]
        for j in range(i + 1, len(sorted_methods)):
            if sorted_methods[j][1] - sorted_methods[i][1] < cd:
                clique.append(sorted_methods[j][0])
        if len(clique) > 1:
            # Only add if not a subset of existing clique
            is_subset = any(set(clique).issubset(set(c)) for c in cliques)
            if not is_subset:
                cliques.append(clique)

    result = {
        "friedman_chi_sq": float(chi2),
        "friedman_p": float(p_friedman),
        "mean_ranks": mean_ranks,
        "nemenyi_cd": cd,
        "nemenyi_q_alpha": q_alpha,
        "cd_cliques": cliques,
        "sorted_methods": [m for m, _ in sorted_methods],
        "sorted_ranks": [float(r) for _, r in sorted_methods],
    }
    logger.info(f"  Friedman chi²={chi2:.2f}, p={p_friedman:.2e}, CD={cd:.3f}")
    return result


# ---------------------------------------------------------------------------
# Block 3: Dataset Meta-Analysis Correlations
# ---------------------------------------------------------------------------

def block3_meta_analysis(
    data: dict,
    iter2_data: dict,
    analysis_meta: dict,
    datasets: list[str],
    max_rules: int = 10,
) -> dict:
    """Spearman correlations between SG-FIGS-25 advantage and dataset properties."""
    logger.info("Block 3: Dataset meta-analysis correlations")

    # Compute SG-FIGS-25 advantage over RO-FIGS for each dataset
    advantages = {}
    for ds in datasets:
        ds_data = data.get(ds, {})
        sg25_accs = []
        rofigs_accs = []
        for fold in range(N_FOLDS):
            sg25_val = ds_data.get("SG-FIGS-25", {}).get(max_rules, {}).get(fold, {}).get("accuracy", float("nan"))
            rofigs_val = ds_data.get("RO-FIGS", {}).get(max_rules, {}).get(fold, {}).get("accuracy", float("nan"))
            sg25_accs.append(sg25_val)
            rofigs_accs.append(rofigs_val)
        sg25_mean = float(np.nanmean(sg25_accs))
        rofigs_mean = float(np.nanmean(rofigs_accs))
        advantages[ds] = sg25_mean - rofigs_mean

    # Dataset properties
    properties: dict[str, dict[str, float]] = {}
    for ds in datasets:
        iter2 = iter2_data.get(ds, {})
        # Synergy stability from analysis_metadata (Jaccard)
        stability = analysis_meta.get("synergy_graph_stability", {}).get(ds, {})
        jaccard = stability.get("mean_jaccard", float("nan"))

        properties[ds] = {
            "n_features": float(iter2.get("n_features", float("nan"))),
            "n_samples": float(iter2.get("n_samples", float("nan"))),
            "n_classes": float(iter2.get("n_classes", float("nan"))),
            "synergy_graph_density_25pct": iter2.get("synergy_graph_density_25pct", float("nan")),
            "mean_synergy_score": iter2.get("mean_synergy_score", float("nan")),
            "synergy_stability_jaccard": jaccard,
            "mean_feature_redundancy": iter2.get("mean_feature_redundancy", float("nan")),
        }

    # Spearman correlations
    property_names = [
        "n_features", "n_samples", "n_classes",
        "synergy_graph_density_25pct", "mean_synergy_score",
        "synergy_stability_jaccard", "mean_feature_redundancy",
    ]
    correlations = {}
    adv_values = [advantages[ds] for ds in datasets]
    for prop_name in property_names:
        prop_values = [properties[ds].get(prop_name, float("nan")) for ds in datasets]
        # Filter out NaN pairs
        valid = [(a, p) for a, p in zip(adv_values, prop_values) if not (math.isnan(a) or math.isnan(p))]
        if len(valid) < 4:
            correlations[prop_name] = {"rho": float("nan"), "p_value": float("nan"), "n_valid": len(valid)}
            continue
        adv_v, prop_v = zip(*valid)
        rho, p_val = stats.spearmanr(adv_v, prop_v)
        correlations[prop_name] = {
            "rho": float(rho),
            "p_value": float(p_val),
            "n_valid": len(valid),
        }

    result = {
        "sgfigs25_advantage": advantages,
        "dataset_properties": properties,
        "spearman_correlations": correlations,
    }
    logger.info(f"  SG-FIGS-25 advantage range: [{min(advantages.values()):.4f}, {max(advantages.values()):.4f}]")
    return result


# ---------------------------------------------------------------------------
# Block 4: Oblique Split Activation Analysis
# ---------------------------------------------------------------------------

def block4_oblique_activation(
    data: dict,
    datasets: list[str],
    max_rules: int = 10,
) -> dict:
    """Compute oblique_fraction, n_oblique, degeneration flags per method×dataset."""
    logger.info("Block 4: Oblique split activation analysis")
    results = {}
    degeneration_counts = {m: 0 for m in METHODS if "SG-FIGS" in m or m == "RO-FIGS"}

    for ds in datasets:
        ds_data = data.get(ds, {})
        ds_result = {}
        for method in METHODS:
            mr_data = ds_data.get(method, {}).get(max_rules, {})
            oblique_fracs = []
            n_obliques = []
            n_synergy_edges_list = []
            for fold in range(N_FOLDS):
                fd = mr_data.get(fold, {})
                oblique_fracs.append(fd.get("oblique_fraction", 0.0))
                n_obliques.append(fd.get("n_oblique", 0))
                n_synergy_edges_list.append(fd.get("n_synergy_edges", 0))
            mean_oblique_frac = float(np.mean(oblique_fracs))
            mean_n_oblique = float(np.mean(n_obliques))
            mean_n_synergy_edges = float(np.mean(n_synergy_edges_list))
            # Degeneration: oblique_fraction == 0 for SG-FIGS variants
            degenerated = mean_oblique_frac == 0.0 and ("SG-FIGS" in method or method == "RO-FIGS")
            if degenerated and method in degeneration_counts:
                degeneration_counts[method] += 1
            ds_result[method] = {
                "mean_oblique_fraction": mean_oblique_frac,
                "mean_n_oblique": mean_n_oblique,
                "mean_n_synergy_edges": mean_n_synergy_edges,
                "degeneration_flag": degenerated,
            }
        results[ds] = ds_result

    # Correlation between n_synergy_edges and oblique_fraction for SG-FIGS variants
    edge_oblique_correlations = {}
    for method in ["SG-FIGS-10", "SG-FIGS-25", "SG-FIGS-50", "RO-FIGS"]:
        edges = []
        obliques = []
        for ds in datasets:
            edges.append(results[ds][method]["mean_n_synergy_edges"])
            obliques.append(results[ds][method]["mean_oblique_fraction"])
        if len(set(obliques)) > 1 and len(set(edges)) > 1:
            rho, p_val = stats.spearmanr(edges, obliques)
            edge_oblique_correlations[method] = {"rho": float(rho), "p_value": float(p_val)}
        else:
            edge_oblique_correlations[method] = {"rho": float("nan"), "p_value": float("nan")}

    return {
        "per_dataset_method": results,
        "degeneration_counts": degeneration_counts,
        "edge_oblique_correlations": edge_oblique_correlations,
    }


# ---------------------------------------------------------------------------
# Block 5: Accuracy-at-Matched-Complexity Curves
# ---------------------------------------------------------------------------

def block5_complexity_curves(
    data: dict,
    datasets: list[str],
) -> dict:
    """Compute mean accuracy at max_rules={5,10,15} for each method."""
    logger.info("Block 5: Accuracy-at-matched-complexity curves")
    results = {}
    for method in METHODS:
        mr_results = {}
        for mr in MAX_RULES_VALUES:
            all_accs = []
            for ds in datasets:
                ds_data = data.get(ds, {})
                mr_data = ds_data.get(method, {}).get(mr, {})
                for fold in range(N_FOLDS):
                    acc = mr_data.get(fold, {}).get("accuracy", float("nan"))
                    all_accs.append(acc)
            mean_acc = float(np.nanmean(all_accs))
            mr_results[mr] = mean_acc
        # Peak
        peak_mr = max(mr_results, key=mr_results.get)
        peak_acc = mr_results[peak_mr]
        efficiency_ratio = peak_acc / peak_mr if peak_mr > 0 else 0.0
        results[method] = {
            "accuracy_at_mr5": mr_results[5],
            "accuracy_at_mr10": mr_results[10],
            "accuracy_at_mr15": mr_results[15],
            "peak_max_rules": peak_mr,
            "peak_accuracy": peak_acc,
            "efficiency_ratio": efficiency_ratio,
        }

    # Per-dataset peak complexity: test if SG-FIGS-25 peaks at lower max_rules than FIGS
    sg25_lower_count = 0
    figs_lower_count = 0
    ties = 0
    for ds in datasets:
        ds_data = data.get(ds, {})
        # Find peak for each method on this dataset
        for target_method, counter_method in [("SG-FIGS-25", "FIGS")]:
            target_peaks = {}
            counter_peaks = {}
            for mr in MAX_RULES_VALUES:
                target_accs = [ds_data.get(target_method, {}).get(mr, {}).get(fold, {}).get("accuracy", float("nan"))
                               for fold in range(N_FOLDS)]
                counter_accs = [ds_data.get(counter_method, {}).get(mr, {}).get(fold, {}).get("accuracy", float("nan"))
                                for fold in range(N_FOLDS)]
                target_peaks[mr] = float(np.nanmean(target_accs))
                counter_peaks[mr] = float(np.nanmean(counter_accs))
            target_peak_mr = max(target_peaks, key=target_peaks.get)
            counter_peak_mr = max(counter_peaks, key=counter_peaks.get)
            if target_peak_mr < counter_peak_mr:
                sg25_lower_count += 1
            elif target_peak_mr > counter_peak_mr:
                figs_lower_count += 1
            else:
                ties += 1

    # Sign test
    n_sign = sg25_lower_count + figs_lower_count
    if n_sign > 0:
        sign_test_p = float(stats.binomtest(sg25_lower_count, n_sign, 0.5).pvalue)
    else:
        sign_test_p = 1.0

    results["peak_complexity_comparison"] = {
        "sg25_peaks_lower": sg25_lower_count,
        "figs_peaks_lower": figs_lower_count,
        "ties": ties,
        "sign_test_p": sign_test_p,
    }
    return results


# ---------------------------------------------------------------------------
# Block 6: Ablation Decomposition
# ---------------------------------------------------------------------------

def block6_ablation_decomposition(
    data: dict,
    datasets: list[str],
    max_rules: int = 10,
) -> dict:
    """Decompose accuracy changes: oblique_penalty and synergy_effect."""
    logger.info("Block 6: Ablation decomposition")
    results = {}
    categories_count = {"oblique_harmful": 0, "synergy_helps": 0, "synergy_hurts": 0, "neutral": 0}

    for ds in datasets:
        ds_data = data.get(ds, {})
        # Get mean accuracies at max_rules=10
        means = {}
        for method in ["FIGS", "RO-FIGS", "SG-FIGS-25"]:
            mr_data = ds_data.get(method, {}).get(max_rules, {})
            accs = [mr_data.get(fold, {}).get("accuracy", float("nan")) for fold in range(N_FOLDS)]
            means[method] = float(np.nanmean(accs))

        figs_acc = means["FIGS"]
        rofigs_acc = means["RO-FIGS"]
        sgfigs25_acc = means["SG-FIGS-25"]

        oblique_penalty = figs_acc - rofigs_acc  # Cost of oblique splits
        synergy_effect = sgfigs25_acc - rofigs_acc  # Incremental effect of synergy guidance
        total_gap = figs_acc - sgfigs25_acc

        # Classify
        category = "neutral"
        if oblique_penalty > 0.05:
            category = "oblique_harmful"
        if synergy_effect > 0:
            if category == "oblique_harmful":
                category = "oblique_harmful"  # Oblique penalty dominates
            else:
                category = "synergy_helps"
        elif synergy_effect < -0.05:
            category = "synergy_hurts"

        if category in categories_count:
            categories_count[category] += 1

        results[ds] = {
            "figs_accuracy": figs_acc,
            "rofigs_accuracy": rofigs_acc,
            "sgfigs25_accuracy": sgfigs25_acc,
            "oblique_penalty": oblique_penalty,
            "synergy_effect": synergy_effect,
            "total_gap": total_gap,
            "category": category,
        }

    n_datasets = len(datasets)
    category_fractions = {k: v / n_datasets for k, v in categories_count.items()}

    return {
        "per_dataset": results,
        "category_counts": categories_count,
        "category_fractions": category_fractions,
    }


# ---------------------------------------------------------------------------
# Block 7: Positive Case Studies
# ---------------------------------------------------------------------------

def block7_case_studies(
    data: dict,
    iter2_data: dict,
    analysis_meta: dict,
    datasets: list[str],
    max_rules: int = 10,
) -> dict:
    """Identify and narrate positive case studies for SG-FIGS."""
    logger.info("Block 7: Positive case studies")

    # Target datasets where SG-FIGS-25 outperforms RO-FIGS
    case_study_datasets = []
    for ds in datasets:
        ds_data = data.get(ds, {})
        sg25_accs = [ds_data.get("SG-FIGS-25", {}).get(max_rules, {}).get(fold, {}).get("accuracy", float("nan"))
                     for fold in range(N_FOLDS)]
        rofigs_accs = [ds_data.get("RO-FIGS", {}).get(max_rules, {}).get(fold, {}).get("accuracy", float("nan"))
                       for fold in range(N_FOLDS)]
        sg25_mean = float(np.nanmean(sg25_accs))
        rofigs_mean = float(np.nanmean(rofigs_accs))
        advantage = sg25_mean - rofigs_mean
        if advantage > -0.001:  # Include near-ties and positives
            case_study_datasets.append(ds)

    results = {}
    for ds in case_study_datasets:
        ds_data = data.get(ds, {})
        iter2 = iter2_data.get(ds, {})
        stability = analysis_meta.get("synergy_graph_stability", {}).get(ds, {})

        # Accuracy details
        sg25_accs = [ds_data.get("SG-FIGS-25", {}).get(max_rules, {}).get(fold, {}).get("accuracy", float("nan"))
                     for fold in range(N_FOLDS)]
        rofigs_accs = [ds_data.get("RO-FIGS", {}).get(max_rules, {}).get(fold, {}).get("accuracy", float("nan"))
                       for fold in range(N_FOLDS)]
        sg25_mean = float(np.nanmean(sg25_accs))
        rofigs_mean = float(np.nanmean(rofigs_accs))
        advantage = sg25_mean - rofigs_mean
        advantage_pct = advantage * 100

        # CI for advantage
        diffs = np.array(sg25_accs) - np.array(rofigs_accs)
        diff_mean = float(np.mean(diffs))
        diff_se = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
        t_crit = stats.t.ppf(0.975, df=len(diffs) - 1)
        ci_low = diff_mean - t_crit * diff_se
        ci_high = diff_mean + t_crit * diff_se

        # Synergy graph properties
        density_25 = iter2.get("synergy_graph_density_25pct", float("nan"))
        jaccard = stability.get("mean_jaccard", float("nan"))
        top_pairs = sorted(iter2.get("pairs", []), key=lambda x: x.get("synergy", 0), reverse=True)[:5]

        # Oblique split details from qualitative_split_inspection
        qual_key_variants = [
            f"{ds}_SG-FIGS-25",
            f"{ds}_SG-FIGS-10",
        ]
        oblique_splits = []
        for qk in qual_key_variants:
            splits = analysis_meta.get("qualitative_split_inspection", {}).get(qk, [])
            for s in splits:
                if s.get("is_oblique", False) or s.get("type") == "oblique":
                    oblique_splits.append(s)

        # Domain meaningfulness rating
        domain_ratings = {
            "diabetes": "clearly meaningful",
            "breast_cancer": "clearly meaningful",
            "heart_statlog": "plausibly meaningful",
            "segment": "no clear interpretation",
            "australian": "plausibly meaningful",
            "ionosphere": "plausibly meaningful",
            "banknote": "no clear interpretation",
            "wine": "plausibly meaningful",
            "glass": "plausibly meaningful",
            "sonar": "no clear interpretation",
            "vehicle": "no clear interpretation",
            "credit_g": "plausibly meaningful",
        }

        # Narrative
        case_flag = advantage > 0.001
        narrative_parts = []
        narrative_parts.append(
            f"Dataset {ds}: SG-FIGS-25 {'outperforms' if advantage > 0 else 'matches'} "
            f"RO-FIGS by {advantage_pct:+.1f}% (95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%])."
        )
        narrative_parts.append(
            f"Synergy graph: density@25%={density_25:.3f}, Jaccard stability={jaccard:.3f}."
        )
        if top_pairs:
            top_pair_strs = [f"({p['fi']},{p['fj']})={p['synergy']:.4f}" for p in top_pairs[:3]]
            narrative_parts.append(f"Top synergy pairs: {', '.join(top_pair_strs)}.")
        if oblique_splits:
            for os_ in oblique_splits[:2]:
                features = os_.get("features", [])
                synergies = os_.get("pairwise_synergies", [])
                rule = os_.get("rule_str", "")
                syn_strs = []
                for s in synergies[:3]:
                    pair = s["pair"]
                    syn_val = s["synergy"]
                    syn_strs.append(f"{pair[0]}-{pair[1]}={syn_val:.3f}")
                narrative_parts.append(
                    f"Oblique split: {', '.join(features)} "
                    f"(synergies: [{', '.join(syn_strs)}]). "
                    f"Rule: {rule}"
                )
        narrative_parts.append(f"Domain meaningfulness: {domain_ratings.get(ds, 'no clear interpretation')}.")
        narrative = " ".join(narrative_parts)

        results[ds] = {
            "case_study_flag": case_flag,
            "advantage": advantage,
            "advantage_pct": advantage_pct,
            "sg25_mean": sg25_mean,
            "rofigs_mean": rofigs_mean,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "synergy_density_25pct": density_25,
            "synergy_stability_jaccard": jaccard,
            "top_synergy_pairs": top_pairs[:5],
            "oblique_splits": oblique_splits[:3],
            "domain_meaningfulness": domain_ratings.get(ds, "no clear interpretation"),
            "narrative": narrative,
        }

    return results


# ---------------------------------------------------------------------------
# Build output in exp_eval_sol_out.json schema format
# ---------------------------------------------------------------------------

def build_output(
    block1: dict,
    block2: dict,
    block3: dict,
    block4: dict,
    block5: dict,
    block6: dict,
    block7: dict,
    data: dict,
    datasets: list[str],
    iter2_data: dict,
    analysis_meta: dict,
    max_rules: int = 10,
) -> dict:
    """Build final output conforming to exp_eval_sol_out.json schema."""
    logger.info("Building output in schema format")

    # --- metrics_agg ---
    # Compute aggregate metrics across all datasets
    all_method_means = {m: [] for m in METHODS}
    for ds in datasets:
        for method in METHODS:
            mean_acc = block1[ds]["method_stats"][method]["accuracy_mean"]
            all_method_means[method].append(mean_acc)

    metrics_agg = {
        # Block 2 aggregates
        "eval_friedman_chi_sq": block2["friedman_chi_sq"],
        "eval_friedman_p": block2["friedman_p"],
        "eval_nemenyi_cd": block2["nemenyi_cd"],
    }
    # Mean ranks
    for method in METHODS:
        safe_name = method.replace("-", "_")
        metrics_agg[f"eval_mean_rank_{safe_name}"] = block2["mean_ranks"][method]
        metrics_agg[f"eval_grand_mean_accuracy_{safe_name}"] = float(np.mean(all_method_means[method]))

    # Block 3 correlations
    for prop_name, corr in block3["spearman_correlations"].items():
        safe_prop = prop_name.replace("-", "_")
        rho = corr.get("rho", float("nan"))
        p_val = corr.get("p_value", float("nan"))
        if math.isnan(rho):
            rho = 0.0
        if math.isnan(p_val):
            p_val = 1.0
        metrics_agg[f"eval_spearman_rho_{safe_prop}"] = rho
        metrics_agg[f"eval_spearman_p_{safe_prop}"] = p_val

    # Block 5 aggregates
    for method in METHODS:
        safe_name = method.replace("-", "_")
        metrics_agg[f"eval_peak_max_rules_{safe_name}"] = float(block5[method]["peak_max_rules"])
        metrics_agg[f"eval_peak_accuracy_{safe_name}"] = block5[method]["peak_accuracy"]
        metrics_agg[f"eval_efficiency_ratio_{safe_name}"] = block5[method]["efficiency_ratio"]

    metrics_agg["eval_sign_test_p_sg25_lower_peak"] = block5["peak_complexity_comparison"]["sign_test_p"]

    # Block 6 aggregates
    for cat, frac in block6["category_fractions"].items():
        metrics_agg[f"eval_ablation_frac_{cat}"] = frac

    # Block 4: degeneration counts
    for method, count in block4["degeneration_counts"].items():
        safe_name = method.replace("-", "_")
        metrics_agg[f"eval_degeneration_count_{safe_name}"] = float(count)

    # Number of case studies
    n_positive = sum(1 for v in block7.values() if v.get("case_study_flag", False))
    metrics_agg["eval_n_positive_case_studies"] = float(n_positive)

    # --- datasets -> examples ---
    output_datasets = []
    for ds in datasets:
        examples = []
        ds_data = data.get(ds, {})
        iter2 = iter2_data.get(ds, {})
        stability = analysis_meta.get("synergy_graph_stability", {}).get(ds, {})
        b1 = block1[ds]
        b4 = block4["per_dataset_method"].get(ds, {})
        b6 = block6["per_dataset"].get(ds, {})
        b7 = block7.get(ds, {})

        for method in METHODS:
            mr_data = ds_data.get(method, {}).get(max_rules, {})
            for fold in range(N_FOLDS):
                fd = mr_data.get(fold, {})
                acc = fd.get("accuracy", float("nan"))
                auc = fd.get("auc", float("nan"))
                oblique_frac = fd.get("oblique_fraction", 0.0)
                n_oblique = fd.get("n_oblique", 0)

                # Fix NaN for JSON
                acc_safe = acc if not math.isnan(acc) else 0.0
                auc_safe = auc if not math.isnan(auc) else 0.0

                # FIGS accuracy on same fold for delta
                figs_acc = ds_data.get("FIGS", {}).get(max_rules, {}).get(fold, {}).get("accuracy", float("nan"))
                figs_acc_safe = figs_acc if not math.isnan(figs_acc) else 0.0
                delta_vs_figs = acc_safe - figs_acc_safe

                # Rank among methods for this dataset-fold
                fold_accs = {}
                for m in METHODS:
                    fold_acc = ds_data.get(m, {}).get(max_rules, {}).get(fold, {}).get("accuracy", float("nan"))
                    fold_accs[m] = fold_acc if not math.isnan(fold_acc) else 0.0
                ranks = _rank_methods_per_dataset(fold_accs)
                rank = ranks.get(method, float("nan"))

                input_str = json.dumps({
                    "method": method,
                    "dataset": ds,
                    "fold": fold,
                    "max_rules": max_rules,
                })
                output_str = json.dumps({
                    "accuracy": acc_safe,
                    "auc": auc_safe,
                    "oblique_fraction": oblique_frac,
                    "n_oblique": n_oblique,
                })

                example = {
                    "input": input_str,
                    "output": output_str,
                    "metadata_method": method,
                    "metadata_dataset": ds,
                    "metadata_fold": fold,
                    "metadata_max_rules": max_rules,
                    "predict_method_accuracy": str(acc_safe),
                    "predict_method_auc": str(auc_safe),
                    # Block 1 eval metrics
                    "eval_accuracy_mean": b1["method_stats"][method]["accuracy_mean"],
                    "eval_accuracy_std": b1["method_stats"][method]["accuracy_std"],
                    "eval_auc_mean": b1["method_stats"][method]["auc_mean"],
                    "eval_auc_std": b1["method_stats"][method]["auc_std"],
                    "eval_accuracy_rank": rank,
                    "eval_accuracy_delta_vs_figs": delta_vs_figs,
                    # Block 4 eval metrics
                    "eval_oblique_fraction": b4.get(method, {}).get("mean_oblique_fraction", 0.0),
                    "eval_n_oblique": b4.get(method, {}).get("mean_n_oblique", 0.0),
                    "eval_degeneration_flag": 1.0 if b4.get(method, {}).get("degeneration_flag", False) else 0.0,
                    # Block 5 eval metrics
                    "eval_accuracy_at_mr5": block5.get(method, {}).get("accuracy_at_mr5", 0.0),
                    "eval_accuracy_at_mr10": block5.get(method, {}).get("accuracy_at_mr10", 0.0),
                    "eval_accuracy_at_mr15": block5.get(method, {}).get("accuracy_at_mr15", 0.0),
                    "eval_peak_max_rules": float(block5.get(method, {}).get("peak_max_rules", 0)),
                    "eval_peak_accuracy": block5.get(method, {}).get("peak_accuracy", 0.0),
                    # Block 6 eval metrics (dataset-level, same for all methods in dataset)
                    "eval_oblique_penalty": b6.get("oblique_penalty", 0.0),
                    "eval_synergy_effect": b6.get("synergy_effect", 0.0),
                    "eval_total_gap": b6.get("total_gap", 0.0),
                    # Block 7 case study flag
                    "eval_case_study_flag": 1.0 if b7.get("case_study_flag", False) else 0.0,
                }

                # Block 3: sgfigs_advantage for the dataset
                example["eval_sgfigs_advantage"] = block3["sgfigs25_advantage"].get(ds, 0.0)

                examples.append(example)

        output_datasets.append({
            "dataset": ds,
            "examples": examples,
        })

    output = {
        "metrics_agg": metrics_agg,
        "datasets": output_datasets,
    }
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("SG-FIGS Comprehensive Statistical Evaluation")
    logger.info("=" * 60)

    # Load data
    iter4_full_path = ITER4_DIR / "full_method_out.json"
    iter2_full_path = ITER2_DIR / "full_method_out.json"
    analysis_meta_path = ITER4_DIR / "analysis_metadata.json"

    data = load_iter4_data(iter4_full_path)
    iter2_data = load_iter2_data(iter2_full_path)
    analysis_meta = load_analysis_metadata(analysis_meta_path)

    datasets = list(data.keys())
    logger.info(f"Datasets ({len(datasets)}): {datasets}")
    logger.info(f"Methods: {METHODS}")

    # Run all 7 blocks
    block1 = block1_per_dataset_accuracy(data=data, datasets=datasets, max_rules=10)
    block2 = block2_critical_difference(data=data, datasets=datasets, max_rules=10)
    block3 = block3_meta_analysis(
        data=data,
        iter2_data=iter2_data,
        analysis_meta=analysis_meta,
        datasets=datasets,
        max_rules=10,
    )
    block4 = block4_oblique_activation(data=data, datasets=datasets, max_rules=10)
    block5 = block5_complexity_curves(data=data, datasets=datasets)
    block6 = block6_ablation_decomposition(data=data, datasets=datasets, max_rules=10)
    block7 = block7_case_studies(
        data=data,
        iter2_data=iter2_data,
        analysis_meta=analysis_meta,
        datasets=datasets,
        max_rules=10,
    )

    # Build output
    output = build_output(
        block1=block1,
        block2=block2,
        block3=block3,
        block4=block4,
        block5=block5,
        block6=block6,
        block7=block7,
        data=data,
        datasets=datasets,
        iter2_data=iter2_data,
        analysis_meta=analysis_meta,
        max_rules=10,
    )

    # Save
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {out_path}")

    # Log summary
    n_datasets = len(output["datasets"])
    n_examples = sum(len(d["examples"]) for d in output["datasets"])
    logger.info(f"Output: {n_datasets} datasets, {n_examples} examples")
    logger.info(f"Friedman p = {output['metrics_agg']['eval_friedman_p']:.2e}")
    logger.info(f"Nemenyi CD = {output['metrics_agg']['eval_nemenyi_cd']:.3f}")

    # Log key findings
    logger.info("--- Key Findings ---")
    for m in METHODS:
        safe_name = m.replace("-", "_")
        rank = output["metrics_agg"][f"eval_mean_rank_{safe_name}"]
        acc = output["metrics_agg"][f"eval_grand_mean_accuracy_{safe_name}"]
        logger.info(f"  {m}: mean_rank={rank:.2f}, grand_mean_acc={acc:.4f}")

    logger.success("Evaluation complete!")


if __name__ == "__main__":
    main()
