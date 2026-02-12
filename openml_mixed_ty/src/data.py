# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scikit-learn>=1.3.0",
#     "pandas>=2.0.0",
#     "numpy>=1.26.0,<2.3.0",
# ]
# ///
"""
data.py — OpenML Mixed-Type Classification Benchmarks

Loads 2 datasets (credit-g and Australian) from OpenML via sklearn,
processes them with ordinal encoding for categoricals + standardization
for numerics, generates 5-fold CV indices, and saves to full_data_out.json.

Datasets:
  - credit_g (OpenML ID=31): 1000x20, binary, 7 numeric + 13 categorical
    German credit risk dataset (Dr. Hans Hofmann, UCI 1994)
  - australian (OpenML ID=40981): 690x14, binary, 6 numeric + 8 categorical
    Australian credit approval (Statlog project, Quinlan 1987)

Each row becomes one example with:
  - input: JSON string of processed feature values
  - output: target label as string
  - metadata_fold: fold index (0-4)
  - metadata_feature_names: list of feature names
  - metadata_feature_types: list of 'categorical' or 'continuous'
  - metadata_task_type: 'classification'
  - metadata_n_classes: number of classes
  - metadata_row_index: original row index
"""
import json
import resource
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Resource limits: 14GB RAM, 1 hour CPU
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# Paths
WORKSPACE = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260212_072136/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
OUTPUT_PATH = WORKSPACE / "full_data_out.json"

# Dataset configurations: (openml_id, name)
# Selected 2 best datasets matching proposal criteria:
# - Mixed numeric + categorical features
# - 500-1000 samples, 14-20 features, binary classification
# - Both ordinal and nominal categoricals with 2-10 unique values
# - No missing values
DATASETS = [
    (31, "credit_g"),
    (40981, "australian"),
]

N_FOLDS = 5
RANDOM_STATE = 42


def process_dataset(data_id: int, dataset_name: str) -> dict:
    """Load, encode, and process a single dataset into the schema format."""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name} (OpenML ID={data_id})")

    # Fetch from OpenML
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = bunch.data.copy()
    target = bunch.target.copy()

    n_rows, n_cols = df.shape
    n_classes = target.nunique()
    print(f"  Shape: {n_rows}x{n_cols}, Classes: {n_classes}")

    # Identify categorical vs numeric columns
    cat_cols = [c for c in df.columns if df[c].dtype.name == "category" or df[c].dtype == object]
    num_cols = [c for c in df.columns if c not in cat_cols]
    print(f"  Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}")

    # Build feature_types list (preserving column order)
    feature_types = []
    for c in df.columns:
        if c in cat_cols:
            feature_types.append("categorical")
        else:
            feature_types.append("continuous")

    feature_names = list(df.columns)

    # Ordinal encode categorical features
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = oe.fit_transform(df[cat_cols])
        cat_categories = {col: list(cats) for col, cats in zip(cat_cols, oe.categories_)}
        print(f"  Ordinal encoded {len(cat_cols)} categorical features")
        for col in cat_cols[:3]:
            print(f"    {col}: {len(cat_categories[col])} categories")

    # Standardize numeric features
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        print(f"  Standardized {len(num_cols)} numeric features")

    # Convert to float numpy array
    X = df.values.astype(np.float64)
    y = target.values

    # Generate 5-fold stratified CV indices
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_assignments = np.zeros(len(y), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(X, y)):
        fold_assignments[test_idx] = fold_idx

    # Build examples list — one example per row
    examples = []
    for i in range(len(X)):
        # Create input as JSON string of feature values
        feature_dict = {}
        for j, fname in enumerate(feature_names):
            val = X[i, j]
            # Round to avoid floating point noise
            if feature_types[j] == "categorical":
                feature_dict[fname] = int(val)
            else:
                feature_dict[fname] = round(float(val), 6)

        example = {
            "input": json.dumps(feature_dict),
            "output": str(y[i]),
            "metadata_fold": int(fold_assignments[i]),
            "metadata_feature_names": feature_names,
            "metadata_feature_types": feature_types,
            "metadata_task_type": "classification",
            "metadata_n_classes": int(n_classes),
            "metadata_row_index": i,
        }
        examples.append(example)

    print(f"  Generated {len(examples)} examples with {N_FOLDS}-fold CV")

    return {
        "dataset": dataset_name,
        "examples": examples,
    }


def main():
    print("=" * 60)
    print("OpenML Mixed-Type Classification Benchmarks — data.py")
    print("=" * 60)

    all_datasets = []
    for data_id, name in DATASETS:
        dataset_result = process_dataset(data_id=data_id, dataset_name=name)
        all_datasets.append(dataset_result)

    # Assemble final output
    output = {"datasets": all_datasets}

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"File size: {file_size_mb:.2f} MB")

    # Summary
    total_examples = sum(len(ds["examples"]) for ds in all_datasets)
    print(f"\nTotal datasets: {len(all_datasets)}")
    print(f"Total examples: {total_examples}")
    for ds in all_datasets:
        print(f"  {ds['dataset']}: {len(ds['examples'])} examples")


if __name__ == "__main__":
    main()
