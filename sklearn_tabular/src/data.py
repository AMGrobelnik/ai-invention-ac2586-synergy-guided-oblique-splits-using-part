# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scikit-learn>=1.3.0",
#     "numpy>=1.26.0,<2.3.0",
# ]
# ///
"""
Data collection script for Sklearn Built-in Tabular Classification Datasets.

Loads breast_cancer and wine datasets from sklearn.datasets (the 2 best-matching
datasets for the criteria: 10-30 features, 100-600 samples, all-numeric, no missing values).
For each dataset:
  1. Extract X (features), y (labels), feature_names, target_names
  2. Standardize continuous features (StandardScaler)
  3. Discretize via KBinsDiscretizer (5 and 10 bins, quantile strategy)
  4. Generate 5-fold StratifiedKFold indices (random_state=42)
  5. Output each row as a separate example in exp_sel_data_out.json schema

Output schema: { "datasets": [ { "dataset": "...", "examples": [...] } ] }
Each example: { "input": "<JSON string of feature values>", "output": "<label>",
                "metadata_fold": int, "metadata_feature_names": [...], ... }
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

# Resource limits: 14GB RAM, 3600s CPU
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

WORKSPACE = Path(__file__).parent
OUTPUT_FILE = WORKSPACE / "full_data_out.json"


def process_dataset(
    name: str,
    loader_func,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Process a single sklearn dataset into the standardized schema."""
    data = loader_func()
    X_raw = data.data
    y = data.target
    feature_names = [str(f) for f in data.feature_names]
    target_names = [str(t) for t in data.target_names]
    n_samples, n_features = X_raw.shape
    n_classes = len(target_names)

    print(f"  Processing {name}: {n_samples} samples, {n_features} features, {n_classes} classes")

    # 1. Standardize continuous features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 2. Discretize with KBinsDiscretizer (5 bins and 10 bins)
    discretizer_5 = KBinsDiscretizer(
        n_bins=5,
        encode="ordinal",
        strategy="quantile",
        subsample=None,
    )
    X_disc_5 = discretizer_5.fit_transform(X_raw)

    discretizer_10 = KBinsDiscretizer(
        n_bins=10,
        encode="ordinal",
        strategy="quantile",
        subsample=None,
    )
    X_disc_10 = discretizer_10.fit_transform(X_raw)

    # 3. Generate 5-fold StratifiedKFold indices
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )
    fold_assignments = np.zeros(n_samples, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(X_raw, y)):
        fold_assignments[test_idx] = fold_idx

    # 4. Build examples (one per row)
    examples = []
    for i in range(n_samples):
        # Build input as JSON string of feature values (using original continuous values)
        feature_dict = {}
        for j, fname in enumerate(feature_names):
            feature_dict[fname] = round(float(X_raw[i, j]), 6)

        # Build the input string: JSON representation of feature values
        input_str = json.dumps(feature_dict)

        # Output: target class name as string
        output_str = str(target_names[y[i]])

        example = {
            "input": input_str,
            "output": output_str,
            "metadata_fold": int(fold_assignments[i]),
            "metadata_feature_names": feature_names,
            "metadata_task_type": "classification",
            "metadata_n_classes": n_classes,
            "metadata_row_index": i,
            "metadata_n_features": n_features,
            "metadata_n_samples": n_samples,
            "metadata_target_names": target_names,
            "metadata_standardized_values": [round(float(v), 6) for v in X_scaled[i]],
            "metadata_discretized_5bin": [int(v) for v in X_disc_5[i]],
            "metadata_discretized_10bin": [int(v) for v in X_disc_10[i]],
        }
        examples.append(example)

    print(f"  -> Generated {len(examples)} examples for {name}")
    return {"dataset": name, "examples": examples}


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Loading sklearn built-in tabular classification datasets")
    print("=" * 60)

    datasets_config = [
        ("breast_cancer_wisconsin", load_breast_cancer),
        ("wine", load_wine),
    ]

    all_datasets = []
    for name, loader in datasets_config:
        dataset_entry = process_dataset(name=name, loader_func=loader)
        all_datasets.append(dataset_entry)

    output = {"datasets": all_datasets}

    # Write output
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    file_size = OUTPUT_FILE.stat().st_size
    print(f"\nOutput written to: {OUTPUT_FILE}")
    print(f"File size: {file_size / 1024:.1f} KB")

    # Summary
    total_examples = sum(len(d["examples"]) for d in all_datasets)
    print(f"Total datasets: {len(all_datasets)}")
    print(f"Total examples: {total_examples}")
    for d in all_datasets:
        print(f"  {d['dataset']}: {len(d['examples'])} examples")

    print("\nDone!")


if __name__ == "__main__":
    main()
