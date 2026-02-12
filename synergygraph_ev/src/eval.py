#!/usr/bin/env python3
"""Synergy Graph Sufficiency Analysis.

Evaluates whether synergy graph structural properties (density, clique coverage,
threshold sensitivity) explain the SG-FIGS accuracy gap across datasets.

Combines data from:
- iter4: SG-FIGS experiment results (accuracy, oblique fractions, synergy edges)
- iter2: PID synergy graph construction (stability metrics)
- iter4 analysis_metadata: synergy graph stability (Jaccard), statistical tests

Metrics computed:
1. Graph Density vs. Accuracy Gap Correlation
2. Threshold Sensitivity Profile
3. Oblique Activation Rate vs. Accuracy
4. Sparse Graph Correction Counterfactual
5. Stability-Performance Relationship
"""

import json
import math
import resource
import sys
from pathlib import Path

from loguru import logger

# --- Resource limits ---
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))  # 14GB RAM
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time

# --- Logging setup ---
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# --- Paths ---
WORKSPACE = Path(__file__).parent
ITER4_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260212_072136"
    "/3_invention_loop/iter_4/gen_art/exp_id1_it4__opus"
)
ITER2_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260212_072136"
    "/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus"
)

# Dataset name mapping between iter2 and iter4
ITER2_TO_ITER4_NAME = {
    "banknote-authentication": "banknote",
    "heart-statlog": "heart_statlog",
    "breast_cancer_wisconsin": "breast_cancer",
    "diabetes": "diabetes",
    "glass": "glass",
    "wine": "wine",
    "sonar": "sonar",
    "ionosphere": "ionosphere",
    "vehicle": "vehicle",
    "segment": "segment",
    "credit_g": "credit_g",
    "australian": "australian",
}

ITER4_TO_ITER2_NAME = {v: k for k, v in ITER2_TO_ITER4_NAME.items()}

# Methods we care about
FIGS_METHOD = "FIGS"
SG_METHODS = ["SG-FIGS-10", "SG-FIGS-25", "SG-FIGS-50"]
ALL_METHODS = [FIGS_METHOD, "RO-FIGS"] + SG_METHODS + ["GradientBoosting"]

# We focus on max_rules=10 for graph density analysis as specified in proposal
TARGET_MAX_RULES = 10


def safe_float(val: float) -> float:
    """Convert NaN/Inf to None-safe float for JSON serialization."""
    if val is None or math.isnan(val) or math.isinf(val):
        return 0.0
    return float(val)


def spearman_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Spearman rank correlation and approximate p-value.

    Uses scipy-free implementation with tied-rank handling.
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    def rank_data(data: list[float]) -> list[float]:
        """Assign ranks with average tie-breaking."""
        indexed = sorted(enumerate(data), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = rank_data(x)
    ry = rank_data(y)

    # Pearson on ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        return 0.0, 1.0

    rho = num / (den_x * den_y)
    rho = max(-1.0, min(1.0, rho))

    # Approximate two-sided p-value using t-distribution approximation
    if abs(rho) >= 1.0:
        p_val = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1 - rho**2))
        # Approximate p-value using normal distribution for large-ish n
        # For small n this is rough but acceptable
        p_val = 2 * (1 - _normal_cdf(abs(t_stat)))

    return rho, p_val


def pearson_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Pearson correlation and approximate p-value."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        return 0.0, 1.0

    r = num / (den_x * den_y)
    r = max(-1.0, min(1.0, r))

    if abs(r) >= 1.0:
        p_val = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r**2))
        p_val = 2 * (1 - _normal_cdf(abs(t_stat)))

    return r, p_val


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using error function approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def load_iter4_data(data_path: Path) -> dict:
    """Load iter4 experiment results."""
    logger.info(f"Loading iter4 data from {data_path}")
    raw = json.loads(data_path.read_text())
    logger.info(f"Loaded {len(raw['datasets'])} datasets from iter4")
    return raw


def load_iter2_data(data_path: Path) -> dict:
    """Load iter2 synergy graph data."""
    logger.info(f"Loading iter2 data from {data_path}")
    raw = json.loads(data_path.read_text())
    logger.info(f"Loaded {len(raw['datasets'])} datasets from iter2")
    return raw


def load_analysis_metadata(meta_path: Path) -> dict:
    """Load analysis metadata from iter4."""
    logger.info(f"Loading analysis metadata from {meta_path}")
    raw = json.loads(meta_path.read_text())
    return raw


def build_iter4_index(
    iter4_data: dict,
) -> dict[str, dict[str, dict[int, dict[int, dict]]]]:
    """Build index: dataset -> method -> max_rules -> fold -> example."""
    index: dict = {}
    for ds_block in iter4_data["datasets"]:
        ds_name = ds_block["dataset"]
        index[ds_name] = {}
        for ex in ds_block["examples"]:
            method = ex["metadata_method"]
            max_rules = ex["metadata_max_rules"]
            fold = ex["metadata_fold"]
            if method not in index[ds_name]:
                index[ds_name][method] = {}
            if max_rules not in index[ds_name][method]:
                index[ds_name][method][max_rules] = {}
            index[ds_name][method][max_rules][fold] = ex
    return index


def build_iter2_index(iter2_data: dict) -> dict[str, list[dict]]:
    """Build index: iter4_dataset_name -> list of synergy pair examples."""
    index: dict = {}
    for ds_block in iter2_data["datasets"]:
        ds_name_iter2 = ds_block["dataset"]
        ds_name_iter4 = ITER2_TO_ITER4_NAME.get(ds_name_iter2, ds_name_iter2)
        index[ds_name_iter4] = ds_block["examples"]
    return index


def compute_graph_density(
    iter4_index: dict,
    iter2_index: dict,
    datasets: list[str],
) -> dict:
    """Metric 1: Graph Density vs. Accuracy Gap Correlation.

    For each dataset at max_rules=10:
    - Compute synergy graph density = n_synergy_edges / total_possible_pairs at 25% threshold
    - Compute accuracy gap = FIGS_acc - SG_FIGS_25_acc
    - Compute Spearman correlation across datasets
    """
    logger.info("Computing Metric 1: Graph Density vs. Accuracy Gap Correlation")

    densities = []
    accuracy_gaps = []
    per_dataset = {}

    for ds in datasets:
        # Get n_features from iter2 data for total possible pairs
        if ds in iter2_index and iter2_index[ds]:
            n_features = iter2_index[ds][0].get("metadata_n_features", 0)
        else:
            logger.warning(f"No iter2 data for {ds}, skipping density calc")
            continue

        total_possible_pairs = n_features * (n_features - 1) // 2
        if total_possible_pairs == 0:
            logger.warning(f"Zero possible pairs for {ds}")
            continue

        # Get SG-FIGS-25 synergy edges at max_rules=10, average across folds
        sg25_data = iter4_index.get(ds, {}).get("SG-FIGS-25", {}).get(TARGET_MAX_RULES, {})
        figs_data = iter4_index.get(ds, {}).get("FIGS", {}).get(TARGET_MAX_RULES, {})

        if not sg25_data or not figs_data:
            logger.warning(f"Missing SG-FIGS-25 or FIGS data for {ds} at max_rules={TARGET_MAX_RULES}")
            continue

        # Average synergy edges across folds for SG-FIGS-25
        sg25_edges_list = []
        sg25_acc_list = []
        figs_acc_list = []

        for fold in sorted(sg25_data.keys()):
            ex = sg25_data[fold]
            n_edges = ex.get("metadata_n_synergy_edges", 0)
            sg25_edges_list.append(n_edges)
            sg25_acc = ex.get("metadata_accuracy", 0.0)
            if isinstance(sg25_acc, float) and math.isnan(sg25_acc):
                sg25_acc = 0.0
            sg25_acc_list.append(sg25_acc)

        for fold in sorted(figs_data.keys()):
            ex = figs_data[fold]
            figs_acc = ex.get("metadata_accuracy", 0.0)
            if isinstance(figs_acc, float) and math.isnan(figs_acc):
                figs_acc = 0.0
            figs_acc_list.append(figs_acc)

        avg_synergy_edges = sum(sg25_edges_list) / len(sg25_edges_list)
        density = avg_synergy_edges / total_possible_pairs
        avg_figs_acc = sum(figs_acc_list) / len(figs_acc_list)
        avg_sg25_acc = sum(sg25_acc_list) / len(sg25_acc_list)
        acc_gap = avg_figs_acc - avg_sg25_acc

        densities.append(density)
        accuracy_gaps.append(acc_gap)

        per_dataset[ds] = {
            "n_features": n_features,
            "total_possible_pairs": total_possible_pairs,
            "avg_synergy_edges_25pct": round(avg_synergy_edges, 2),
            "graph_density_25pct": round(density, 4),
            "avg_figs_accuracy": round(avg_figs_acc, 4),
            "avg_sg25_accuracy": round(avg_sg25_acc, 4),
            "accuracy_gap_figs_minus_sg25": round(acc_gap, 4),
        }

        logger.info(
            f"  {ds}: density={density:.4f}, acc_gap={acc_gap:.4f}, "
            f"edges={avg_synergy_edges:.1f}/{total_possible_pairs}"
        )

    # Compute Spearman correlation
    rho, p_val = spearman_correlation(densities, accuracy_gaps)
    logger.info(f"  Spearman rho={rho:.4f}, p={p_val:.6f}")

    return {
        "spearman_rho": round(rho, 4),
        "spearman_p_value": round(p_val, 6),
        "n_datasets": len(densities),
        "interpretation": (
            "Significant negative correlation: denser graphs -> smaller accuracy loss"
            if rho < -0.3 and p_val < 0.05
            else (
                "Significant positive correlation: denser graphs -> larger accuracy loss"
                if rho > 0.3 and p_val < 0.05
                else "No significant correlation between graph density and accuracy gap"
            )
        ),
        "per_dataset": per_dataset,
    }


def compute_threshold_sensitivity(
    iter4_index: dict,
    datasets: list[str],
) -> dict:
    """Metric 2: Threshold Sensitivity Profile.

    For each dataset, compute accuracy at all 3 thresholds relative to FIGS baseline.
    Classify as monotonic_improving, monotonic_worsening, non_monotonic, or flat.
    """
    logger.info("Computing Metric 2: Threshold Sensitivity Profile")

    per_dataset = {}
    profile_counts = {
        "monotonic_improving": 0,
        "monotonic_worsening": 0,
        "non_monotonic": 0,
        "flat": 0,
    }

    for ds in datasets:
        figs_data = iter4_index.get(ds, {}).get("FIGS", {}).get(TARGET_MAX_RULES, {})
        sg10_data = iter4_index.get(ds, {}).get("SG-FIGS-10", {}).get(TARGET_MAX_RULES, {})
        sg25_data = iter4_index.get(ds, {}).get("SG-FIGS-25", {}).get(TARGET_MAX_RULES, {})
        sg50_data = iter4_index.get(ds, {}).get("SG-FIGS-50", {}).get(TARGET_MAX_RULES, {})

        if not all([figs_data, sg10_data, sg25_data, sg50_data]):
            logger.warning(f"Missing data for threshold sensitivity on {ds}")
            continue

        def avg_acc(data: dict) -> float:
            accs = []
            for fold_ex in data.values():
                acc = fold_ex.get("metadata_accuracy", 0.0)
                if isinstance(acc, float) and math.isnan(acc):
                    acc = 0.0
                accs.append(acc)
            return sum(accs) / len(accs) if accs else 0.0

        figs_acc = avg_acc(figs_data)
        sg10_acc = avg_acc(sg10_data)
        sg25_acc = avg_acc(sg25_data)
        sg50_acc = avg_acc(sg50_data)

        # Relative accuracy (gap from FIGS)
        rel_10 = sg10_acc - figs_acc
        rel_25 = sg25_acc - figs_acc
        rel_50 = sg50_acc - figs_acc

        # Classify pattern: does accuracy change as threshold becomes more permissive?
        # SG-FIGS-10 (most restrictive) -> SG-FIGS-25 -> SG-FIGS-50 (most permissive)
        eps = 0.005  # tolerance for "flat"

        if abs(sg50_acc - sg10_acc) < eps and abs(sg25_acc - sg10_acc) < eps:
            profile = "flat"
        elif sg50_acc > sg25_acc - eps and sg25_acc > sg10_acc - eps and sg50_acc > sg10_acc + eps:
            profile = "monotonic_improving"
        elif sg50_acc < sg25_acc + eps and sg25_acc < sg10_acc + eps and sg50_acc < sg10_acc - eps:
            profile = "monotonic_worsening"
        else:
            profile = "non_monotonic"

        profile_counts[profile] += 1

        per_dataset[ds] = {
            "figs_accuracy": round(figs_acc, 4),
            "sg10_accuracy": round(sg10_acc, 4),
            "sg25_accuracy": round(sg25_acc, 4),
            "sg50_accuracy": round(sg50_acc, 4),
            "relative_sg10": round(rel_10, 4),
            "relative_sg25": round(rel_25, 4),
            "relative_sg50": round(rel_50, 4),
            "profile": profile,
        }

        logger.info(
            f"  {ds}: SG10={sg10_acc:.4f}, SG25={sg25_acc:.4f}, SG50={sg50_acc:.4f}, "
            f"FIGS={figs_acc:.4f} -> {profile}"
        )

    logger.info(f"  Profile distribution: {profile_counts}")

    return {
        "profile_distribution": profile_counts,
        "per_dataset": per_dataset,
        "n_datasets": len(per_dataset),
        "interpretation": (
            f"Monotonic improving: {profile_counts['monotonic_improving']}, "
            f"Monotonic worsening: {profile_counts['monotonic_worsening']}, "
            f"Non-monotonic: {profile_counts['non_monotonic']}, "
            f"Flat: {profile_counts['flat']}"
        ),
    }


def compute_oblique_activation_rate(
    iter4_index: dict,
    datasets: list[str],
) -> dict:
    """Metric 3: Oblique Activation Rate vs. Accuracy.

    For each method x dataset combo, compute Pearson correlation between
    oblique_fraction and accuracy. Identifies datasets where SG-FIGS actually
    produces oblique splits.
    """
    logger.info("Computing Metric 3: Oblique Activation Rate vs. Accuracy")

    # Collect oblique_fraction and accuracy for SG-FIGS methods across datasets
    per_dataset = {}
    zero_oblique_datasets = []
    nonzero_oblique_datasets = []

    # For overall correlation across all methods x datasets
    all_oblique_fractions = []
    all_accuracies = []

    for ds in datasets:
        ds_info = {}
        for method in SG_METHODS:
            method_data = iter4_index.get(ds, {}).get(method, {}).get(TARGET_MAX_RULES, {})
            if not method_data:
                continue

            oblique_fracs = []
            accs = []
            for fold_ex in method_data.values():
                of = fold_ex.get("metadata_oblique_fraction", 0.0)
                acc = fold_ex.get("metadata_accuracy", 0.0)
                if isinstance(of, float) and math.isnan(of):
                    of = 0.0
                if isinstance(acc, float) and math.isnan(acc):
                    acc = 0.0
                oblique_fracs.append(of)
                accs.append(acc)

            avg_of = sum(oblique_fracs) / len(oblique_fracs)
            avg_acc = sum(accs) / len(accs)

            ds_info[method] = {
                "avg_oblique_fraction": round(avg_of, 4),
                "avg_accuracy": round(avg_acc, 4),
            }

            all_oblique_fractions.append(avg_of)
            all_accuracies.append(avg_acc)

        # Check if SG-FIGS-25 has zero oblique fraction
        sg25_of = ds_info.get("SG-FIGS-25", {}).get("avg_oblique_fraction", 0.0)
        if sg25_of == 0.0:
            zero_oblique_datasets.append(ds)
        else:
            nonzero_oblique_datasets.append(ds)

        per_dataset[ds] = ds_info

    # Compute Pearson correlation across all method x dataset combos
    r, p_val = pearson_correlation(all_oblique_fractions, all_accuracies)

    # Also compute mean accuracy for zero vs nonzero oblique datasets (SG-FIGS-25 only)
    zero_acc = []
    nonzero_acc = []
    for ds in zero_oblique_datasets:
        info = per_dataset.get(ds, {}).get("SG-FIGS-25", {})
        if info:
            zero_acc.append(info["avg_accuracy"])
    for ds in nonzero_oblique_datasets:
        info = per_dataset.get(ds, {}).get("SG-FIGS-25", {})
        if info:
            nonzero_acc.append(info["avg_accuracy"])

    mean_zero_acc = sum(zero_acc) / len(zero_acc) if zero_acc else 0.0
    mean_nonzero_acc = sum(nonzero_acc) / len(nonzero_acc) if nonzero_acc else 0.0

    logger.info(f"  Pearson r={r:.4f}, p={p_val:.6f}")
    logger.info(f"  Zero-oblique datasets: {zero_oblique_datasets}")
    logger.info(f"  Nonzero-oblique datasets: {nonzero_oblique_datasets}")
    logger.info(f"  Mean acc (zero oblique): {mean_zero_acc:.4f}")
    logger.info(f"  Mean acc (nonzero oblique): {mean_nonzero_acc:.4f}")

    return {
        "pearson_r": round(r, 4),
        "pearson_p_value": round(p_val, 6),
        "n_observations": len(all_oblique_fractions),
        "zero_oblique_datasets_sg25": zero_oblique_datasets,
        "nonzero_oblique_datasets_sg25": nonzero_oblique_datasets,
        "mean_accuracy_zero_oblique": round(mean_zero_acc, 4),
        "mean_accuracy_nonzero_oblique": round(mean_nonzero_acc, 4),
        "per_dataset": per_dataset,
        "interpretation": (
            f"Datasets with zero oblique splits (SG-FIGS-25): {len(zero_oblique_datasets)}/12. "
            f"Mean accuracy: zero-oblique={mean_zero_acc:.4f}, "
            f"nonzero-oblique={mean_nonzero_acc:.4f}. "
            f"Pearson r={r:.4f} (p={p_val:.4f})"
        ),
    }


def compute_sparse_graph_correction(
    iter4_index: dict,
    datasets: list[str],
) -> dict:
    """Metric 4: Sparse Graph Correction Counterfactual.

    For datasets where SG-FIGS-25 has oblique_fraction=0, SG-FIGS-25 should
    behave identically to FIGS. Verify by comparing accuracies. Any discrepancy
    indicates a bug or implementation artifact.
    """
    logger.info("Computing Metric 4: Sparse Graph Correction Counterfactual")

    per_dataset = {}
    discrepancies = []

    for ds in datasets:
        sg25_data = iter4_index.get(ds, {}).get("SG-FIGS-25", {}).get(TARGET_MAX_RULES, {})
        figs_data = iter4_index.get(ds, {}).get("FIGS", {}).get(TARGET_MAX_RULES, {})

        if not sg25_data or not figs_data:
            continue

        # Check if SG-FIGS-25 has zero oblique fraction across all folds
        all_zero = True
        for fold_ex in sg25_data.values():
            of = fold_ex.get("metadata_oblique_fraction", 0.0)
            if isinstance(of, float) and math.isnan(of):
                of = 0.0
            if of > 0.0:
                all_zero = False
                break

        if not all_zero:
            continue

        # Compare fold-by-fold accuracy
        fold_comparisons = []
        max_discrepancy = 0.0
        for fold in sorted(sg25_data.keys()):
            if fold not in figs_data:
                continue
            sg25_acc = sg25_data[fold].get("metadata_accuracy", 0.0)
            figs_acc = figs_data[fold].get("metadata_accuracy", 0.0)
            if isinstance(sg25_acc, float) and math.isnan(sg25_acc):
                sg25_acc = 0.0
            if isinstance(figs_acc, float) and math.isnan(figs_acc):
                figs_acc = 0.0
            diff = abs(sg25_acc - figs_acc)
            max_discrepancy = max(max_discrepancy, diff)
            fold_comparisons.append({
                "fold": fold,
                "figs_accuracy": round(figs_acc, 6),
                "sg25_accuracy": round(sg25_acc, 6),
                "absolute_difference": round(diff, 6),
            })

        is_identical = max_discrepancy < 1e-10
        if not is_identical:
            discrepancies.append(ds)

        per_dataset[ds] = {
            "all_oblique_zero": True,
            "max_fold_discrepancy": round(max_discrepancy, 8),
            "is_identical_to_figs": is_identical,
            "fold_comparisons": fold_comparisons,
        }

        logger.info(
            f"  {ds}: zero-oblique, max_discrepancy={max_discrepancy:.8f}, "
            f"identical={is_identical}"
        )

    return {
        "n_zero_oblique_datasets": len(per_dataset),
        "datasets_with_discrepancy": discrepancies,
        "n_discrepancies": len(discrepancies),
        "per_dataset": per_dataset,
        "interpretation": (
            f"Found {len(per_dataset)} datasets with zero oblique splits for SG-FIGS-25. "
            + (
                f"All match FIGS exactly — no implementation artifacts."
                if len(discrepancies) == 0
                else (
                    f"{len(discrepancies)} dataset(s) show discrepancy from FIGS despite "
                    f"zero oblique splits: {discrepancies}. This indicates potential "
                    f"implementation artifacts (e.g., Ridge preprocessing side effects)."
                )
            )
        ),
    }


def compute_stability_performance(
    iter4_index: dict,
    analysis_metadata: dict,
    datasets: list[str],
) -> dict:
    """Metric 5: Stability-Performance Relationship.

    Compute whether datasets with more stable synergy graphs (higher Jaccard)
    have smaller accuracy gaps. Spearman correlation across 12 datasets.
    """
    logger.info("Computing Metric 5: Stability-Performance Relationship")

    stability_data = analysis_metadata.get("synergy_graph_stability", {})

    jaccards = []
    accuracy_gaps = []
    per_dataset = {}

    for ds in datasets:
        if ds not in stability_data:
            logger.warning(f"No stability data for {ds}")
            continue

        mean_jaccard = stability_data[ds].get("mean_jaccard", 0.0)

        # Compute accuracy gap (FIGS - SG-FIGS-25) at max_rules=10
        figs_data = iter4_index.get(ds, {}).get("FIGS", {}).get(TARGET_MAX_RULES, {})
        sg25_data = iter4_index.get(ds, {}).get("SG-FIGS-25", {}).get(TARGET_MAX_RULES, {})

        if not figs_data or not sg25_data:
            continue

        def avg_acc(data: dict) -> float:
            accs = []
            for fold_ex in data.values():
                acc = fold_ex.get("metadata_accuracy", 0.0)
                if isinstance(acc, float) and math.isnan(acc):
                    acc = 0.0
                accs.append(acc)
            return sum(accs) / len(accs) if accs else 0.0

        figs_acc = avg_acc(figs_data)
        sg25_acc = avg_acc(sg25_data)
        acc_gap = figs_acc - sg25_acc

        jaccards.append(mean_jaccard)
        accuracy_gaps.append(acc_gap)

        per_dataset[ds] = {
            "mean_jaccard": round(mean_jaccard, 4),
            "std_jaccard": round(stability_data[ds].get("std_jaccard", 0.0), 4),
            "figs_accuracy": round(figs_acc, 4),
            "sg25_accuracy": round(sg25_acc, 4),
            "accuracy_gap": round(acc_gap, 4),
        }

        logger.info(
            f"  {ds}: jaccard={mean_jaccard:.4f}, acc_gap={acc_gap:.4f}"
        )

    rho, p_val = spearman_correlation(jaccards, accuracy_gaps)
    logger.info(f"  Spearman rho={rho:.4f}, p={p_val:.6f}")

    return {
        "spearman_rho": round(rho, 4),
        "spearman_p_value": round(p_val, 6),
        "n_datasets": len(jaccards),
        "interpretation": (
            "Stable graphs (high Jaccard) correlate with smaller accuracy gaps"
            if rho < -0.3 and p_val < 0.1
            else (
                "Stable graphs correlate with LARGER accuracy gaps (unexpected)"
                if rho > 0.3 and p_val < 0.1
                else "No significant relationship between graph stability and accuracy gap"
            )
        ),
        "per_dataset": per_dataset,
    }


def compute_additional_metrics(
    iter4_index: dict,
    iter2_index: dict,
    analysis_metadata: dict,
    datasets: list[str],
) -> dict:
    """Additional analysis beyond the 5 core metrics.

    - Method ranking consistency across max_rules values
    - AUC gap analysis (complementary to accuracy)
    - Edge count distribution across thresholds
    """
    logger.info("Computing additional metrics")

    # --- AUC Gap Analysis ---
    auc_gaps = {}
    for ds in datasets:
        figs_data = iter4_index.get(ds, {}).get("FIGS", {}).get(TARGET_MAX_RULES, {})
        sg25_data = iter4_index.get(ds, {}).get("SG-FIGS-25", {}).get(TARGET_MAX_RULES, {})
        if not figs_data or not sg25_data:
            continue

        def avg_metric(data: dict, key: str) -> float:
            vals = []
            for fold_ex in data.values():
                v = fold_ex.get(key, 0.0)
                if isinstance(v, float) and math.isnan(v):
                    v = 0.0
                vals.append(v)
            return sum(vals) / len(vals) if vals else 0.0

        figs_auc = avg_metric(figs_data, "metadata_auc")
        sg25_auc = avg_metric(sg25_data, "metadata_auc")
        auc_gaps[ds] = {
            "figs_auc": round(figs_auc, 4),
            "sg25_auc": round(sg25_auc, 4),
            "auc_gap": round(figs_auc - sg25_auc, 4),
        }

    # --- Edge count progression across thresholds ---
    edge_progression = {}
    for ds in datasets:
        ds_edges = {}
        for method in SG_METHODS:
            method_data = iter4_index.get(ds, {}).get(method, {}).get(TARGET_MAX_RULES, {})
            if not method_data:
                continue
            edges = []
            for fold_ex in method_data.values():
                e = fold_ex.get("metadata_n_synergy_edges", 0)
                edges.append(e)
            ds_edges[method] = round(sum(edges) / len(edges), 2) if edges else 0.0
        edge_progression[ds] = ds_edges

    # --- Max rules sensitivity (do results change across max_rules?) ---
    max_rules_sensitivity = {}
    for ds in datasets:
        for mr in [5, 10, 15]:
            figs_data = iter4_index.get(ds, {}).get("FIGS", {}).get(mr, {})
            sg25_data = iter4_index.get(ds, {}).get("SG-FIGS-25", {}).get(mr, {})
            if not figs_data or not sg25_data:
                continue

            def avg_acc(data: dict) -> float:
                accs = []
                for fold_ex in data.values():
                    acc = fold_ex.get("metadata_accuracy", 0.0)
                    if isinstance(acc, float) and math.isnan(acc):
                        acc = 0.0
                    accs.append(acc)
                return sum(accs) / len(accs) if accs else 0.0

            figs_acc = avg_acc(figs_data)
            sg25_acc = avg_acc(sg25_data)
            if ds not in max_rules_sensitivity:
                max_rules_sensitivity[ds] = {}
            max_rules_sensitivity[ds][str(mr)] = {
                "figs_acc": round(figs_acc, 4),
                "sg25_acc": round(sg25_acc, 4),
                "gap": round(figs_acc - sg25_acc, 4),
            }

    return {
        "auc_gap_analysis": auc_gaps,
        "edge_count_progression": edge_progression,
        "max_rules_sensitivity": max_rules_sensitivity,
    }


def build_eval_output(
    iter4_data: dict,
    iter2_data: dict,
    analysis_metadata: dict,
    metric1: dict,
    metric2: dict,
    metric3: dict,
    metric4: dict,
    metric5: dict,
    additional: dict,
    datasets: list[str],
    iter4_index: dict,
) -> dict:
    """Build output conforming to exp_eval_sol_out.json schema.

    Schema requires:
    - metrics_agg: aggregate metrics (numbers only)
    - datasets: array of {dataset, examples: [{input, output, eval_*, metadata_*, predict_*}]}
    """
    logger.info("Building evaluation output in exp_eval_sol_out schema format")

    # --- metrics_agg ---
    metrics_agg = {
        # Metric 1: Graph Density vs Accuracy Gap
        "m1_density_gap_spearman_rho": safe_float(metric1["spearman_rho"]),
        "m1_density_gap_spearman_p": safe_float(metric1["spearman_p_value"]),
        "m1_n_datasets": metric1["n_datasets"],
        # Metric 2: Threshold Sensitivity
        "m2_n_monotonic_improving": metric2["profile_distribution"]["monotonic_improving"],
        "m2_n_monotonic_worsening": metric2["profile_distribution"]["monotonic_worsening"],
        "m2_n_non_monotonic": metric2["profile_distribution"]["non_monotonic"],
        "m2_n_flat": metric2["profile_distribution"]["flat"],
        # Metric 3: Oblique Activation Rate
        "m3_oblique_acc_pearson_r": safe_float(metric3["pearson_r"]),
        "m3_oblique_acc_pearson_p": safe_float(metric3["pearson_p_value"]),
        "m3_n_zero_oblique_datasets": len(metric3["zero_oblique_datasets_sg25"]),
        "m3_mean_acc_zero_oblique": safe_float(metric3["mean_accuracy_zero_oblique"]),
        "m3_mean_acc_nonzero_oblique": safe_float(metric3["mean_accuracy_nonzero_oblique"]),
        # Metric 4: Sparse Graph Correction
        "m4_n_zero_oblique_datasets": metric4["n_zero_oblique_datasets"],
        "m4_n_discrepancies": metric4["n_discrepancies"],
        # Metric 5: Stability-Performance
        "m5_stability_gap_spearman_rho": safe_float(metric5["spearman_rho"]),
        "m5_stability_gap_spearman_p": safe_float(metric5["spearman_p_value"]),
    }

    # --- datasets array ---
    eval_datasets = []

    for ds in datasets:
        examples = []

        # Each dataset gets one example per analysis aspect
        # Example 1: Dataset-level summary with all metrics
        ds_density = metric1.get("per_dataset", {}).get(ds, {})
        ds_threshold = metric2.get("per_dataset", {}).get(ds, {})
        ds_oblique = metric3.get("per_dataset", {}).get(ds, {})
        ds_sparse = metric4.get("per_dataset", {}).get(ds, {})
        ds_stability = metric5.get("per_dataset", {}).get(ds, {})
        ds_auc = additional.get("auc_gap_analysis", {}).get(ds, {})
        ds_edges = additional.get("edge_count_progression", {}).get(ds, {})
        ds_mr = additional.get("max_rules_sensitivity", {}).get(ds, {})

        input_str = json.dumps({
            "dataset": ds,
            "analysis": "synergy_graph_sufficiency",
            "max_rules": TARGET_MAX_RULES,
        })

        output_parts = []
        if ds_density:
            output_parts.append(
                f"density={ds_density.get('graph_density_25pct', 0):.4f}"
            )
            output_parts.append(
                f"acc_gap={ds_density.get('accuracy_gap_figs_minus_sg25', 0):.4f}"
            )
        if ds_threshold:
            output_parts.append(f"profile={ds_threshold.get('profile', 'unknown')}")
        if ds_oblique:
            sg25_of = ds_oblique.get("SG-FIGS-25", {}).get("avg_oblique_fraction", 0)
            output_parts.append(f"oblique_frac_sg25={sg25_of:.4f}")
        if ds_sparse:
            output_parts.append(
                f"identical_to_figs={ds_sparse.get('is_identical_to_figs', 'N/A')}"
            )
        if ds_stability:
            output_parts.append(
                f"jaccard={ds_stability.get('mean_jaccard', 0):.4f}"
            )
        output_str = ", ".join(output_parts) if output_parts else "no_data"

        example = {
            "input": input_str,
            "output": output_str,
        }

        # eval_ fields (per-example metrics, must be numbers)
        if ds_density:
            example["eval_graph_density_25pct"] = safe_float(
                ds_density.get("graph_density_25pct", 0)
            )
            example["eval_accuracy_gap"] = safe_float(
                ds_density.get("accuracy_gap_figs_minus_sg25", 0)
            )
            example["eval_avg_synergy_edges"] = safe_float(
                ds_density.get("avg_synergy_edges_25pct", 0)
            )

        if ds_threshold:
            example["eval_figs_accuracy"] = safe_float(
                ds_threshold.get("figs_accuracy", 0)
            )
            example["eval_sg10_accuracy"] = safe_float(
                ds_threshold.get("sg10_accuracy", 0)
            )
            example["eval_sg25_accuracy"] = safe_float(
                ds_threshold.get("sg25_accuracy", 0)
            )
            example["eval_sg50_accuracy"] = safe_float(
                ds_threshold.get("sg50_accuracy", 0)
            )
            example["eval_relative_sg10"] = safe_float(
                ds_threshold.get("relative_sg10", 0)
            )
            example["eval_relative_sg25"] = safe_float(
                ds_threshold.get("relative_sg25", 0)
            )
            example["eval_relative_sg50"] = safe_float(
                ds_threshold.get("relative_sg50", 0)
            )

        if ds_oblique:
            sg25_info = ds_oblique.get("SG-FIGS-25", {})
            example["eval_oblique_fraction_sg25"] = safe_float(
                sg25_info.get("avg_oblique_fraction", 0)
            )
            example["eval_accuracy_sg25"] = safe_float(
                sg25_info.get("avg_accuracy", 0)
            )

        if ds_sparse:
            example["eval_max_fold_discrepancy"] = safe_float(
                ds_sparse.get("max_fold_discrepancy", 0)
            )
            example["eval_is_identical_to_figs"] = (
                1.0 if ds_sparse.get("is_identical_to_figs", False) else 0.0
            )

        if ds_stability:
            example["eval_mean_jaccard"] = safe_float(
                ds_stability.get("mean_jaccard", 0)
            )
            example["eval_stability_accuracy_gap"] = safe_float(
                ds_stability.get("accuracy_gap", 0)
            )

        if ds_auc:
            example["eval_auc_gap"] = safe_float(ds_auc.get("auc_gap", 0))

        # metadata_ fields
        example["metadata_dataset"] = ds
        example["metadata_max_rules"] = TARGET_MAX_RULES
        if ds_density:
            example["metadata_n_features"] = ds_density.get("n_features", 0)
            example["metadata_total_possible_pairs"] = ds_density.get(
                "total_possible_pairs", 0
            )
        if ds_threshold:
            example["metadata_threshold_profile"] = ds_threshold.get("profile", "unknown")

        # predict_ fields
        if ds_density:
            example["predict_graph_density"] = str(
                round(ds_density.get("graph_density_25pct", 0), 4)
            )
        if ds_threshold:
            example["predict_threshold_profile"] = ds_threshold.get("profile", "unknown")

        examples.append(example)

        # Add per-method examples for this dataset (one per SG method)
        for method in SG_METHODS:
            method_oblique = ds_oblique.get(method, {})
            method_edges_avg = ds_edges.get(method, 0)

            method_input = json.dumps({
                "dataset": ds,
                "method": method,
                "analysis": "method_performance",
                "max_rules": TARGET_MAX_RULES,
            })

            method_output_parts = [
                f"method={method}",
                f"oblique_frac={method_oblique.get('avg_oblique_fraction', 0):.4f}",
                f"accuracy={method_oblique.get('avg_accuracy', 0):.4f}",
                f"avg_edges={method_edges_avg}",
            ]
            method_output = ", ".join(method_output_parts)

            method_example = {
                "input": method_input,
                "output": method_output,
                "eval_oblique_fraction": safe_float(
                    method_oblique.get("avg_oblique_fraction", 0)
                ),
                "eval_method_accuracy": safe_float(
                    method_oblique.get("avg_accuracy", 0)
                ),
                "eval_avg_synergy_edges": safe_float(method_edges_avg),
                "metadata_dataset": ds,
                "metadata_method": method,
                "metadata_max_rules": TARGET_MAX_RULES,
                "predict_oblique_fraction": str(
                    round(method_oblique.get("avg_oblique_fraction", 0), 4)
                ),
                "predict_accuracy": str(
                    round(method_oblique.get("avg_accuracy", 0), 4)
                ),
            }
            examples.append(method_example)

        # Add per-max_rules examples
        for mr_key, mr_data in sorted(ds_mr.items()):
            mr_input = json.dumps({
                "dataset": ds,
                "max_rules": int(mr_key),
                "analysis": "max_rules_sensitivity",
            })
            mr_output = (
                f"max_rules={mr_key}, "
                f"figs_acc={mr_data['figs_acc']:.4f}, "
                f"sg25_acc={mr_data['sg25_acc']:.4f}, "
                f"gap={mr_data['gap']:.4f}"
            )
            mr_example = {
                "input": mr_input,
                "output": mr_output,
                "eval_figs_accuracy": safe_float(mr_data["figs_acc"]),
                "eval_sg25_accuracy": safe_float(mr_data["sg25_acc"]),
                "eval_accuracy_gap": safe_float(mr_data["gap"]),
                "metadata_dataset": ds,
                "metadata_max_rules": int(mr_key),
                "predict_accuracy_gap": str(round(mr_data["gap"], 4)),
            }
            examples.append(mr_example)

        eval_datasets.append({
            "dataset": ds,
            "examples": examples,
        })

    return {
        "metrics_agg": metrics_agg,
        "datasets": eval_datasets,
    }


@logger.catch
def main():
    """Main evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Synergy Graph Sufficiency Analysis")
    logger.info("=" * 60)

    # --- Load data ---
    try:
        iter4_data = load_iter4_data(ITER4_DIR / "full_method_out.json")
    except FileNotFoundError:
        logger.exception("iter4 data file not found")
        raise
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in iter4 data")
        raise

    try:
        iter2_data = load_iter2_data(ITER2_DIR / "full_method_out.json")
    except FileNotFoundError:
        logger.exception("iter2 data file not found")
        raise
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in iter2 data")
        raise

    try:
        analysis_metadata = load_analysis_metadata(ITER4_DIR / "analysis_metadata.json")
    except FileNotFoundError:
        logger.exception("analysis_metadata.json not found")
        raise
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in analysis_metadata")
        raise

    # --- Build indices ---
    iter4_index = build_iter4_index(iter4_data)
    iter2_index = build_iter2_index(iter2_data)

    datasets = sorted(iter4_index.keys())
    logger.info(f"Found {len(datasets)} datasets: {datasets}")

    # --- Compute metrics ---
    metric1 = compute_graph_density(
        iter4_index=iter4_index,
        iter2_index=iter2_index,
        datasets=datasets,
    )

    metric2 = compute_threshold_sensitivity(
        iter4_index=iter4_index,
        datasets=datasets,
    )

    metric3 = compute_oblique_activation_rate(
        iter4_index=iter4_index,
        datasets=datasets,
    )

    metric4 = compute_sparse_graph_correction(
        iter4_index=iter4_index,
        datasets=datasets,
    )

    metric5 = compute_stability_performance(
        iter4_index=iter4_index,
        analysis_metadata=analysis_metadata,
        datasets=datasets,
    )

    additional = compute_additional_metrics(
        iter4_index=iter4_index,
        iter2_index=iter2_index,
        analysis_metadata=analysis_metadata,
        datasets=datasets,
    )

    # --- Build output ---
    output = build_eval_output(
        iter4_data=iter4_data,
        iter2_data=iter2_data,
        analysis_metadata=analysis_metadata,
        metric1=metric1,
        metric2=metric2,
        metric3=metric3,
        metric4=metric4,
        metric5=metric5,
        additional=additional,
        datasets=datasets,
        iter4_index=iter4_index,
    )

    # --- Save output ---
    output_path = WORKSPACE / "eval_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved evaluation output to {output_path}")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Metric 1 (Density-Gap Correlation): rho={metric1['spearman_rho']}, p={metric1['spearman_p_value']}")
    logger.info(f"  {metric1['interpretation']}")
    logger.info(f"Metric 2 (Threshold Sensitivity): {metric2['interpretation']}")
    logger.info(f"Metric 3 (Oblique Activation): r={metric3['pearson_r']}, p={metric3['pearson_p_value']}")
    logger.info(f"  {metric3['interpretation']}")
    logger.info(f"Metric 4 (Sparse Correction): {metric4['interpretation']}")
    logger.info(f"Metric 5 (Stability-Performance): rho={metric5['spearman_rho']}, p={metric5['spearman_p_value']}")
    logger.info(f"  {metric5['interpretation']}")
    logger.info(f"Total examples in output: {sum(len(d['examples']) for d in output['datasets'])}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
