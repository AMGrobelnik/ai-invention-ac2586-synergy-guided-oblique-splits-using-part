# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.26.0,<2.3.0",
#     "pandas>=2.0.0",
#     "scikit-learn>=1.3.0",
# ]
# ///
"""
data.py — Load 8 selected OpenML numeric-only classification datasets, standardize
features, assign 5-fold CV indices, and output in exp_sel_data_out.json schema format.

Selected datasets span diverse sizes (208-2310 samples), feature counts (4-60),
binary and multi-class problems (2-7 classes), and domains (medical, physics,
signal processing, computer vision, forensics, image processing).

Each row = one example. Output grouped by dataset.
"""
import json
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# --- Resource limits ---
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))  # 14GB RAM
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time

# --- Configuration ---
WORKSPACE = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260212_072136/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
DATASET_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_FILE = WORKSPACE / "full_data_out.json"

# Final 8 selected datasets (all-numeric, no missing, 2-7 classes, 200-2500 samples)
DATASETS = {
    "diabetes": {"data_id": 37, "domain": "medical"},
    "heart-statlog": {"data_id": 53, "domain": "medical"},
    "ionosphere": {"data_id": 59, "domain": "physics"},
    "sonar": {"data_id": 40, "domain": "signal_processing"},
    "vehicle": {"data_id": 54, "domain": "computer_vision"},
    "segment": {"data_id": 36, "domain": "computer_vision"},
    "glass": {"data_id": 41, "domain": "forensics"},
    "banknote-authentication": {"data_id": 1462, "domain": "image_processing"},
}

N_FOLDS = 5
RANDOM_STATE = 42


def load_and_process_dataset(name: str, info: dict) -> dict:
    """Load a pickled dataset and convert to schema-compliant examples."""
    pkl_path = DATASET_DIR / f"{name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    X: pd.DataFrame = data['X']
    y: pd.Series = data['y']
    feature_names: list = data['feature_names']
    n_classes: int = data['n_classes']

    # Ensure all numeric, drop any NaN rows (should be none)
    X = X.apply(pd.to_numeric, errors='coerce')
    mask = ~X.isnull().any(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # Standardize features (z-score)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_names,
    )

    # Assign stratified k-fold indices
    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_assignments = np.zeros(len(X_scaled), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(X_scaled, y)):
        fold_assignments[test_idx] = fold_idx

    # Build examples (one per row)
    examples = []
    for row_idx in range(len(X_scaled)):
        # Input: JSON string of feature name-value pairs
        row_dict = {}
        for feat_name in feature_names:
            val = X_scaled.iloc[row_idx][feat_name]
            row_dict[feat_name] = round(float(val), 6)

        input_str = json.dumps(row_dict)
        output_str = str(y.iloc[row_idx])

        example = {
            "input": input_str,
            "output": output_str,
            "metadata_fold": int(fold_assignments[row_idx]),
            "metadata_feature_names": feature_names,
            "metadata_task_type": "classification",
            "metadata_n_classes": n_classes,
            "metadata_row_index": row_idx,
            "metadata_domain": info["domain"],
            "metadata_openml_id": info["data_id"],
        }
        examples.append(example)

    return {
        "dataset": name,
        "examples": examples,
    }


def main():
    start_time = time.time()
    print(f"Processing {len(DATASETS)} datasets...")
    print(f"Output: {OUTPUT_FILE}")

    all_datasets = []
    for name, info in DATASETS.items():
        t0 = time.time()
        try:
            dataset_entry = load_and_process_dataset(name=name, info=info)
            n_examples = len(dataset_entry["examples"])
            elapsed = time.time() - t0
            print(f"  {name:<25s} | {n_examples:>5d} examples | {elapsed:.1f}s")
            all_datasets.append(dataset_entry)
        except Exception as e:
            print(f"  {name:<25s} | ERROR: {e}")

    # Build final output
    output = {"datasets": all_datasets}

    # Write JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    total_examples = sum(len(d["examples"]) for d in all_datasets)
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    total_time = time.time() - start_time

    print(f"\nDone!")
    print(f"  Datasets: {len(all_datasets)}")
    print(f"  Total examples: {total_examples}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
