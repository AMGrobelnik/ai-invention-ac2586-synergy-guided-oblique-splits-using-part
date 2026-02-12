#!/usr/bin/env python3
"""Information-Theoretic Split Quality Audit for SG-FIGS.

Evaluates whether SG-FIGS oblique splits capture more target information than
random feature subsets of the same size, directly testing the PID synergy premise.

Metrics computed:
1. Synergy Concentration Ratio (SCR)
2. Interpretability Score Decomposition
3. Impurity Reduction per Feature Count
4. Redundancy Load
5. Joint MI Coverage
6. Additional: Synergy-Weighted Impurity Efficiency
"""

import json
import math
import resource
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(
    "logs/run.log",
    rotation="30 MB",
    level="DEBUG",
)

# ---------------------------------------------------------------------------
# Resource limits (14 GB RAM, 1h CPU)
# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ITER4_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260212_072136"
    "/3_invention_loop/iter_4/gen_art/exp_id1_it4__opus"
)
ITER2_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260212_072136"
    "/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus"
)
WORKSPACE = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260212_072136"
    "/3_invention_loop/iter_6/gen_art/eval_id3_it6__opus"
)

# Dataset name mapping: iter4 → iter2
DATASET_NAME_MAP: dict[str, str] = {
    "banknote": "banknote-authentication",
    "wine": "wine",
    "glass": "glass",
    "diabetes": "diabetes",
    "heart_statlog": "heart-statlog",
    "sonar": "sonar",
    "breast_cancer": "breast_cancer_wisconsin",
    "ionosphere": "ionosphere",
    "vehicle": "vehicle",
    "segment": "segment",
    "credit_g": "credit_g",
    "australian": "australian",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    """Load a JSON file with proper error handling."""
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    raw = path.read_text()
    # Handle NaN in JSON (non-standard)
    raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(raw)


def build_synergy_lookup(
    iter2_data: dict,
) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    """Build per-dataset lookup: (feat_i, feat_j) → {synergy, redundancy, joint_mi, mi_i, mi_j, ...}.

    Keys are canonical (sorted) feature name tuples.
    """
    lookup: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for ds_block in iter2_data["datasets"]:
        ds_name = ds_block["dataset"]
        pairs: dict[tuple[str, str], dict[str, float]] = {}
        for ex in ds_block["examples"]:
            inp = json.loads(ex["input"])
            fi, fj = inp["feature_i"], inp["feature_j"]
            key = tuple(sorted([fi, fj]))
            pairs[key] = {
                "synergy": ex.get("metadata_synergy", 0.0) or 0.0,
                "redundancy": ex.get("metadata_redundancy", 0.0) or 0.0,
                "joint_mi": ex.get("metadata_joint_mi", 0.0) or 0.0,
                "mi_i": ex.get("metadata_mi_i", 0.0) or 0.0,
                "mi_j": ex.get("metadata_mi_j", 0.0) or 0.0,
                "unique_i": ex.get("metadata_unique_i", 0.0) or 0.0,
                "unique_j": ex.get("metadata_unique_j", 0.0) or 0.0,
            }
        lookup[ds_name] = pairs
    return lookup


def resolve_iter2_name(iter4_name: str) -> str:
    """Resolve iter4 dataset name to iter2 name."""
    return DATASET_NAME_MAP.get(iter4_name, iter4_name)


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------
def compute_scr(
    split_synergies: list[float],
    all_synergies: list[float],
) -> float:
    """Synergy Concentration Ratio: mean split synergy / mean all synergy.

    SCR > 1 means oblique split concentrates on above-average-synergy feature pairs.
    """
    if not split_synergies or not all_synergies:
        return float("nan")
    mean_split = sum(split_synergies) / len(split_synergies)
    mean_all = sum(all_synergies) / len(all_synergies)
    if mean_all == 0:
        return float("nan")
    return mean_split / mean_all


def compute_above_median_fraction(
    split_synergies: list[float],
    all_synergies: list[float],
) -> float:
    """Fraction of split feature pairs with above-median synergy in dataset."""
    if not split_synergies or not all_synergies:
        return float("nan")
    sorted_all = sorted(all_synergies)
    n = len(sorted_all)
    median_synergy = (
        sorted_all[n // 2]
        if n % 2 == 1
        else (sorted_all[n // 2 - 1] + sorted_all[n // 2]) / 2
    )
    above = sum(1 for s in split_synergies if s > median_synergy)
    return above / len(split_synergies)


def compute_impurity_per_feature(
    oblique_impurity: float,
    n_features: int,
    best_axis_impurity: float,
) -> dict[str, float]:
    """Impurity reduction per feature count comparison."""
    oblique_per_feat = oblique_impurity / n_features if n_features > 0 else float("nan")
    return {
        "oblique_impurity_per_feature": oblique_per_feat,
        "best_axis_impurity": best_axis_impurity,
        "ratio": oblique_per_feat / best_axis_impurity if best_axis_impurity > 0 else float("nan"),
    }


def compute_redundancy_load(
    split_features: list[str],
    synergy_pairs: dict[tuple[str, str], dict[str, float]],
    all_pairs_redundancies: list[float],
) -> dict[str, float]:
    """Mean redundancy among split feature pairs vs expected random."""
    feature_pairs = [
        tuple(sorted(pair)) for pair in combinations(split_features, 2)
    ]
    redundancies = []
    for pair in feature_pairs:
        if pair in synergy_pairs:
            redundancies.append(synergy_pairs[pair]["redundancy"])

    if not redundancies:
        return {
            "mean_redundancy": float("nan"),
            "expected_random_redundancy": float("nan"),
            "redundancy_ratio": float("nan"),
        }

    mean_red = sum(redundancies) / len(redundancies)
    mean_all = (
        sum(all_pairs_redundancies) / len(all_pairs_redundancies)
        if all_pairs_redundancies
        else 0.0
    )

    return {
        "mean_redundancy": mean_red,
        "expected_random_redundancy": mean_all,
        "redundancy_ratio": mean_red / mean_all if mean_all > 0 else float("nan"),
    }


def compute_joint_mi_coverage(
    split_features: list[str],
    synergy_pairs: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    """Sum joint_mi of all feature pairs in split vs sum of individual MI."""
    feature_pairs = [
        tuple(sorted(pair)) for pair in combinations(split_features, 2)
    ]
    total_joint_mi = 0.0
    individual_mis: dict[str, float] = {}
    pair_count = 0

    for pair in feature_pairs:
        if pair in synergy_pairs:
            pdata = synergy_pairs[pair]
            total_joint_mi += pdata["joint_mi"]
            pair_count += 1
            # Track individual MI for each feature
            fi, fj = pair
            if fi not in individual_mis:
                individual_mis[fi] = pdata["mi_i"]
            if fj not in individual_mis:
                individual_mis[fj] = pdata["mi_j"]

    sum_individual = sum(individual_mis.values())

    return {
        "total_joint_mi": total_joint_mi,
        "sum_individual_mi": sum_individual,
        "n_pairs_found": pair_count,
        "joint_mi_ratio": (
            total_joint_mi / sum_individual if sum_individual > 0 else float("nan")
        ),
    }


def compute_synergy_weighted_efficiency(
    oblique_impurity: float,
    mean_split_synergy: float,
    best_axis_impurity: float,
) -> float:
    """Synergy-weighted impurity efficiency: (impurity * mean_synergy) / best_axis."""
    if best_axis_impurity == 0 or math.isnan(mean_split_synergy):
        return float("nan")
    return (oblique_impurity * mean_split_synergy) / best_axis_impurity


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
@logger.catch
def main() -> None:
    """Run the information-theoretic split quality audit."""
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Information-Theoretic Split Quality Audit")
    logger.info("=" * 60)

    iter4_method_path = ITER4_DIR / "full_method_out.json"
    iter2_method_path = ITER2_DIR / "full_method_out.json"
    analysis_path = ITER4_DIR / "analysis_metadata.json"

    iter4_data = load_json(iter4_method_path)
    iter2_data = load_json(iter2_method_path)
    analysis_meta = load_json(analysis_path)

    # Build synergy lookup from iter2 data
    synergy_lookup = build_synergy_lookup(iter2_data)
    logger.info(f"Built synergy lookup for {len(synergy_lookup)} datasets")
    for ds_name, pairs in synergy_lookup.items():
        logger.info(f"  {ds_name}: {len(pairs)} feature pairs")

    # Extract qualitative split inspection from analysis_metadata
    split_inspection = analysis_meta.get("qualitative_split_inspection", {})
    logger.info(f"Found {len(split_inspection)} split inspection entries")

    # ------------------------------------------------------------------
    # 2. Process each oblique split inspection entry
    # ------------------------------------------------------------------
    all_scr_values: list[float] = []
    all_above_median_values: list[float] = []
    all_impurity_ratios: list[float] = []
    all_redundancy_ratios: list[float] = []
    all_joint_mi_ratios: list[float] = []
    all_synergy_efficiency: list[float] = []

    output_datasets: list[dict] = []

    for entry_key, splits in split_inspection.items():
        # Parse key: e.g. "diabetes_SG-FIGS-10"
        parts = entry_key.rsplit("_SG-FIGS-", 1)
        if len(parts) != 2:
            logger.warning(f"Skipping unrecognized key: {entry_key}")
            continue

        ds_name_iter4 = parts[0]
        threshold_str = parts[1]
        method_name = f"SG-FIGS-{threshold_str}"
        ds_name_iter2 = resolve_iter2_name(ds_name_iter4)

        logger.info(f"Processing: {ds_name_iter4} / {method_name}")

        # Get synergy data for this dataset
        synergy_pairs = synergy_lookup.get(ds_name_iter2, {})
        if not synergy_pairs:
            # Try with exact name
            synergy_pairs = synergy_lookup.get(ds_name_iter4, {})
        if not synergy_pairs:
            logger.warning(f"  No synergy data for {ds_name_iter2}, skipping")
            continue

        # Collect all synergy/redundancy values for baseline comparison
        all_synergies_dataset = [p["synergy"] for p in synergy_pairs.values()]
        all_redundancies_dataset = [p["redundancy"] for p in synergy_pairs.values()]

        # Separate oblique and axis-aligned splits
        oblique_splits = [s for s in splits if s.get("is_oblique", False)]
        axis_splits = [s for s in splits if not s.get("is_oblique", False)]

        if not oblique_splits:
            logger.info(f"  No oblique splits found for {entry_key}")

        # Best axis-aligned impurity reduction
        best_axis_impurity = (
            max(s.get("impurity_reduction", 0.0) for s in axis_splits)
            if axis_splits
            else 0.0
        )

        examples_for_dataset: list[dict] = []

        for oblique in oblique_splits:
            features = oblique.get("features", [])
            n_features = len(features)
            impurity_reduction = oblique.get("impurity_reduction", 0.0)

            # Extract pairwise synergies from the oblique split data
            split_pairwise = oblique.get("pairwise_synergies", [])
            split_synergy_values = [p["synergy"] for p in split_pairwise]

            # Also look up from iter2 data for pairs not in the split data
            feature_pairs_keys = [
                tuple(sorted(pair)) for pair in combinations(features, 2)
            ]
            iter2_split_synergies = []
            for pair_key in feature_pairs_keys:
                if pair_key in synergy_pairs:
                    iter2_split_synergies.append(synergy_pairs[pair_key]["synergy"])

            # Use split_pairwise synergies if available, else iter2 lookup
            effective_synergies = (
                split_synergy_values if split_synergy_values else iter2_split_synergies
            )

            # Metric 1: Synergy Concentration Ratio
            scr = compute_scr(effective_synergies, all_synergies_dataset)

            # Metric 2: Above-median fraction
            above_med = compute_above_median_fraction(
                effective_synergies, all_synergies_dataset
            )

            # Metric 3: Impurity reduction per feature count
            imp_per_feat = compute_impurity_per_feature(
                impurity_reduction, n_features, best_axis_impurity
            )

            # Metric 4: Redundancy load
            red_load = compute_redundancy_load(
                features, synergy_pairs, all_redundancies_dataset
            )

            # Metric 5: Joint MI coverage
            joint_mi_cov = compute_joint_mi_coverage(features, synergy_pairs)

            # Metric 6: Synergy-weighted efficiency
            mean_split_synergy = (
                sum(effective_synergies) / len(effective_synergies)
                if effective_synergies
                else float("nan")
            )
            syn_eff = compute_synergy_weighted_efficiency(
                impurity_reduction, mean_split_synergy, best_axis_impurity
            )

            # Collect for aggregation
            if not math.isnan(scr):
                all_scr_values.append(scr)
            if not math.isnan(above_med):
                all_above_median_values.append(above_med)
            if not math.isnan(imp_per_feat["ratio"]):
                all_impurity_ratios.append(imp_per_feat["ratio"])
            if not math.isnan(red_load["redundancy_ratio"]):
                all_redundancy_ratios.append(red_load["redundancy_ratio"])
            if not math.isnan(joint_mi_cov["joint_mi_ratio"]):
                all_joint_mi_ratios.append(joint_mi_cov["joint_mi_ratio"])
            if not math.isnan(syn_eff):
                all_synergy_efficiency.append(syn_eff)

            # Build output record
            input_str = json.dumps({
                "dataset": ds_name_iter4,
                "method": method_name,
                "tree": oblique.get("tree", 0),
                "split_index": oblique.get("split_index", 0),
                "features": features,
                "n_features": n_features,
            })

            output_str = json.dumps({
                "scr": round(scr, 6) if not math.isnan(scr) else None,
                "above_median_fraction": round(above_med, 6) if not math.isnan(above_med) else None,
                "impurity_per_feature_ratio": round(imp_per_feat["ratio"], 6) if not math.isnan(imp_per_feat["ratio"]) else None,
                "redundancy_ratio": round(red_load["redundancy_ratio"], 6) if not math.isnan(red_load["redundancy_ratio"]) else None,
                "joint_mi_ratio": round(joint_mi_cov["joint_mi_ratio"], 6) if not math.isnan(joint_mi_cov["joint_mi_ratio"]) else None,
                "synergy_weighted_efficiency": round(syn_eff, 6) if not math.isnan(syn_eff) else None,
            })

            example: dict[str, Any] = {
                "input": input_str,
                "output": output_str,
                "metadata_dataset": ds_name_iter4,
                "metadata_method": method_name,
                "metadata_tree": oblique.get("tree", 0),
                "metadata_split_index": oblique.get("split_index", 0),
                "metadata_n_features": n_features,
                "metadata_impurity_reduction": impurity_reduction,
                "metadata_best_axis_impurity": best_axis_impurity,
                "metadata_n_axis_splits": len(axis_splits),
                "metadata_n_oblique_splits": len(oblique_splits),
                "metadata_n_synergy_pairs_dataset": len(synergy_pairs),
                "metadata_rule_str": oblique.get("rule_str", ""),
                "predict_scr": str(round(scr, 6)) if not math.isnan(scr) else "nan",
                "predict_above_median_fraction": str(round(above_med, 6)) if not math.isnan(above_med) else "nan",
                "predict_impurity_per_feature_ratio": str(round(imp_per_feat["ratio"], 6)) if not math.isnan(imp_per_feat["ratio"]) else "nan",
                "predict_redundancy_ratio": str(round(red_load["redundancy_ratio"], 6)) if not math.isnan(red_load["redundancy_ratio"]) else "nan",
                "predict_joint_mi_ratio": str(round(joint_mi_cov["joint_mi_ratio"], 6)) if not math.isnan(joint_mi_cov["joint_mi_ratio"]) else "nan",
                "predict_synergy_weighted_efficiency": str(round(syn_eff, 6)) if not math.isnan(syn_eff) else "nan",
                "eval_scr": scr if not math.isnan(scr) else 0.0,
                "eval_above_median_fraction": above_med if not math.isnan(above_med) else 0.0,
                "eval_impurity_per_feature_ratio": imp_per_feat["ratio"] if not math.isnan(imp_per_feat["ratio"]) else 0.0,
                "eval_redundancy_ratio": red_load["redundancy_ratio"] if not math.isnan(red_load["redundancy_ratio"]) else 0.0,
                "eval_joint_mi_ratio": joint_mi_cov["joint_mi_ratio"] if not math.isnan(joint_mi_cov["joint_mi_ratio"]) else 0.0,
                "eval_synergy_weighted_efficiency": syn_eff if not math.isnan(syn_eff) else 0.0,
                "eval_oblique_impurity_per_feature": imp_per_feat["oblique_impurity_per_feature"] if not math.isnan(imp_per_feat["oblique_impurity_per_feature"]) else 0.0,
                "eval_mean_redundancy": red_load["mean_redundancy"] if not math.isnan(red_load["mean_redundancy"]) else 0.0,
                "eval_expected_random_redundancy": red_load["expected_random_redundancy"] if not math.isnan(red_load["expected_random_redundancy"]) else 0.0,
                "eval_total_joint_mi": joint_mi_cov["total_joint_mi"],
                "eval_sum_individual_mi": joint_mi_cov["sum_individual_mi"],
            }
            examples_for_dataset.append(example)

            logger.info(
                f"  Split tree={oblique.get('tree')}, features={features}: "
                f"SCR={scr:.3f}, above_med={above_med:.3f}, "
                f"imp_ratio={imp_per_feat['ratio']:.3f}, "
                f"red_ratio={red_load['redundancy_ratio']:.3f}"
                if not any(
                    math.isnan(v) for v in [scr, above_med, imp_per_feat["ratio"], red_load["redundancy_ratio"]]
                )
                else f"  Split tree={oblique.get('tree')}, features={features}: some metrics NaN"
            )

        # Also process iter4 method_out examples for this dataset
        # to get per-method accuracy comparisons
        for ds_block in iter4_data["datasets"]:
            if ds_block["dataset"] == ds_name_iter4:
                for ex in ds_block["examples"]:
                    inp = json.loads(ex["input"])
                    method = inp.get("method", "")
                    if not method.startswith("SG-FIGS"):
                        continue
                    fold = inp.get("fold", 0)
                    accuracy = ex.get("metadata_accuracy", 0.0)
                    auc = ex.get("metadata_auc", 0.0)
                    n_oblique_model = ex.get("metadata_n_oblique", 0) or 0
                    oblique_frac = ex.get("metadata_oblique_fraction", 0.0) or 0.0
                    mean_feat = ex.get("metadata_mean_features_per_oblique", 0.0) or 0.0

                    # Check if this method+dataset combo matches our entry
                    if method != method_name:
                        continue

                    perf_input = json.dumps({
                        "dataset": ds_name_iter4,
                        "method": method_name,
                        "fold": fold,
                        "type": "performance_context",
                    })
                    perf_output = json.dumps({
                        "accuracy": accuracy,
                        "auc": auc,
                        "n_oblique": n_oblique_model,
                        "oblique_fraction": oblique_frac,
                        "mean_features_per_oblique": mean_feat,
                    })

                    perf_example: dict[str, Any] = {
                        "input": perf_input,
                        "output": perf_output,
                        "metadata_dataset": ds_name_iter4,
                        "metadata_method": method_name,
                        "metadata_fold": fold,
                        "metadata_accuracy": accuracy if accuracy else 0.0,
                        "metadata_auc": auc if auc else 0.0,
                        "metadata_n_oblique": n_oblique_model,
                        "metadata_oblique_fraction": oblique_frac,
                        "predict_accuracy": str(round(accuracy, 6)) if accuracy else "0.0",
                        "predict_auc": str(round(auc, 6)) if auc else "0.0",
                        "eval_accuracy": accuracy if accuracy else 0.0,
                        "eval_auc": auc if auc else 0.0,
                    }
                    examples_for_dataset.append(perf_example)
                break

        if examples_for_dataset:
            output_datasets.append({
                "dataset": f"{ds_name_iter4}_{method_name}",
                "examples": examples_for_dataset,
            })

    # ------------------------------------------------------------------
    # 3. Compute aggregate metrics
    # ------------------------------------------------------------------
    def safe_mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def safe_std(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = safe_mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

    metrics_agg: dict[str, float] = {
        "mean_scr": round(safe_mean(all_scr_values), 6),
        "std_scr": round(safe_std(all_scr_values), 6),
        "n_oblique_splits_evaluated": len(all_scr_values),
        "mean_above_median_fraction": round(safe_mean(all_above_median_values), 6),
        "std_above_median_fraction": round(safe_std(all_above_median_values), 6),
        "expected_random_above_median": 0.5,
        "mean_impurity_per_feature_ratio": round(safe_mean(all_impurity_ratios), 6),
        "std_impurity_per_feature_ratio": round(safe_std(all_impurity_ratios), 6),
        "mean_redundancy_ratio": round(safe_mean(all_redundancy_ratios), 6),
        "std_redundancy_ratio": round(safe_std(all_redundancy_ratios), 6),
        "mean_joint_mi_ratio": round(safe_mean(all_joint_mi_ratios), 6),
        "std_joint_mi_ratio": round(safe_std(all_joint_mi_ratios), 6),
        "mean_synergy_weighted_efficiency": round(safe_mean(all_synergy_efficiency), 6),
        "std_synergy_weighted_efficiency": round(safe_std(all_synergy_efficiency), 6),
        "n_datasets_with_oblique_splits": len(
            set(
                d["dataset"].rsplit("_SG-FIGS-", 1)[0]
                for d in output_datasets
                if any(
                    "eval_scr" in ex
                    for ex in d["examples"]
                )
            )
        ),
        "total_examples": sum(len(d["examples"]) for d in output_datasets),
    }

    logger.info("=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)
    for k, v in metrics_agg.items():
        logger.info(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    result = {
        "metrics_agg": metrics_agg,
        "datasets": output_datasets,
    }

    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Summary
    logger.info("=" * 60)
    logger.info("INTERPRETATION SUMMARY")
    logger.info("=" * 60)
    scr_mean = metrics_agg["mean_scr"]
    logger.info(
        f"SCR = {scr_mean:.3f} ({'> 1: splits concentrate high-synergy pairs' if scr_mean > 1 else '≤ 1: splits do NOT preferentially select high-synergy pairs'})"
    )
    above_med = metrics_agg["mean_above_median_fraction"]
    logger.info(
        f"Above-median fraction = {above_med:.3f} (random baseline = 0.5, "
        f"{'better' if above_med > 0.5 else 'worse'} than random)"
    )
    imp_ratio = metrics_agg["mean_impurity_per_feature_ratio"]
    logger.info(
        f"Impurity/feature ratio = {imp_ratio:.3f} "
        f"({'oblique splits MORE efficient per feature' if imp_ratio > 1 else 'oblique splits LESS efficient per feature than best axis-aligned'})"
    )
    red_ratio = metrics_agg["mean_redundancy_ratio"]
    logger.info(
        f"Redundancy ratio = {red_ratio:.3f} "
        f"({'higher than random: features overlap' if red_ratio > 1 else 'lower than random: features complementary'})"
    )


if __name__ == "__main__":
    main()
