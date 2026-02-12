#!/usr/bin/env python3
"""SG-FIGS vs Baselines: Synergy-Guided Oblique Tree Ensembles.

Implements SG-FIGS (synergy-constrained oblique FIGS), RO-FIGS (random oblique FIGS),
standard FIGS, and GradientBoosting baselines. Runs 5-fold CV on 12 tabular datasets.
Compares accuracy, AUC, model complexity, and split interpretability score.
"""

import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from imodels import FIGSClassifier
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer, label_binarize
from sklearn.tree import DecisionTreeRegressor

# ============================================================
# SETUP
# ============================================================

# Resource limits: 14GB RAM, 3500s CPU
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))

WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# Data artifact paths
DATA_PATHS = {
    "data_id1": WORKSPACE / "dependencies" / "data_id1_it1__opus" / "full_data_out.json",
    "data_id2": WORKSPACE / "dependencies" / "data_id2_it1__opus" / "full_data_out.json",
    "data_id3": WORKSPACE / "dependencies" / "data_id3_it1__opus" / "full_data_out.json",
}

# Global config
MAX_EXAMPLES = None  # Set to int to limit examples per dataset (for scaling)
METHODS = ["FIGS", "RO-FIGS", "SG-FIGS", "GradientBoosting"]
MAX_RULES_CANDIDATES = [5, 10, 15]


# ============================================================
# PHASE 0: DATA LOADING
# ============================================================


def load_all_datasets(
    data_paths: dict[str, Path],
    max_examples: int | None = None,
) -> dict[str, dict]:
    """Load all datasets from JSON artifacts.

    Each artifact has structure:
      {"datasets": [{"dataset": str, "examples": [...]}]}

    Returns dict: dataset_name -> {X, y, folds, feature_names, class_names, n_classes}
    """
    all_datasets: dict[str, dict] = {}

    for artifact_id, path in data_paths.items():
        logger.info(f"Loading artifact {artifact_id} from {path.name}")
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            logger.exception(f"Data file not found: {path}")
            raise
        except json.JSONDecodeError:
            logger.exception(f"Invalid JSON in: {path}")
            raise

        for ds_entry in data["datasets"]:
            name = ds_entry["dataset"]
            examples = ds_entry["examples"]

            if max_examples is not None:
                examples = examples[:max_examples]

            # Get full feature names from first example
            first_input = json.loads(examples[0]["input"])
            feature_names = list(first_input.keys())
            n_features = len(feature_names)
            n_samples = len(examples)

            # Reconstruct X, y, folds
            X = np.zeros((n_samples, n_features))
            y_labels: list[str] = []
            folds = np.zeros(n_samples, dtype=int)

            for idx, ex in enumerate(examples):
                features = json.loads(ex["input"])
                X[idx] = [features[fname] for fname in feature_names]
                y_labels.append(ex["output"])
                folds[idx] = ex["metadata_fold"]

            # Encode y as integers
            unique_classes = sorted(set(y_labels))
            class_to_int = {c: i for i, c in enumerate(unique_classes)}
            y = np.array([class_to_int[c] for c in y_labels])

            all_datasets[name] = {
                "X": X,
                "y": y,
                "folds": folds,
                "feature_names": feature_names,
                "class_names": unique_classes,
                "n_classes": len(unique_classes),
            }
            logger.info(
                f"  {name}: {X.shape}, {len(unique_classes)} classes, "
                f"{len(np.unique(folds))} folds"
            )

    return all_datasets


# ============================================================
# PHASE 1: PID SYNERGY COMPUTATION
# ============================================================


def compute_mi(x_disc: np.ndarray, y_disc: np.ndarray) -> float:
    """Mutual information between two discrete arrays using counting."""
    n = len(x_disc)
    if n == 0:
        return 0.0
    joint = Counter(zip(x_disc, y_disc))
    px = Counter(x_disc)
    py = Counter(y_disc)
    mi = 0.0
    for (xi, yi), count in joint.items():
        pxy = count / n
        pxi = px[xi] / n
        pyi = py[yi] / n
        if pxy > 0 and pxi > 0 and pyi > 0:
            mi += pxy * np.log2(pxy / (pxi * pyi))
    return max(float(mi), 0.0)


def compute_pid_synergy(
    x1_disc: np.ndarray,
    x2_disc: np.ndarray,
    y_disc: np.ndarray,
) -> float:
    """Williams-Beer I_min PID synergy between feature pair and target.

    synergy = I(X1,X2 ; Y) - unique_1 - unique_2 - redundancy
    where redundancy = min(I(X1;Y), I(X2;Y))  [I_min measure]
    """
    mi_x1_y = compute_mi(x1_disc, y_disc)
    mi_x2_y = compute_mi(x2_disc, y_disc)
    # Joint MI: combine x1,x2 into single variable
    joint_x = np.array([f"{a}_{b}" for a, b in zip(x1_disc, x2_disc)])
    mi_joint_y = compute_mi(joint_x, y_disc)

    redundancy = min(mi_x1_y, mi_x2_y)
    unique_1 = mi_x1_y - redundancy  # >= 0 by construction
    unique_2 = mi_x2_y - redundancy  # >= 0 by construction
    synergy = mi_joint_y - unique_1 - unique_2 - redundancy
    return max(float(synergy), 0.0)


def build_synergy_graph(
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = 5,
    percentile_threshold: int = 75,
) -> tuple[nx.Graph, dict[tuple[int, int], float], float]:
    """Build synergy graph over features.

    1. Discretize continuous features into n_bins equal-frequency bins
    2. Compute pairwise PID synergy for all feature pairs
    3. Build graph with edges for pairs above the threshold percentile

    Returns: (graph, synergy_scores dict, threshold value)
    """
    n_features = X.shape[1]

    # Discretize (handle constant/near-constant features gracefully)
    X_disc = np.zeros_like(X, dtype=int)
    for f in range(n_features):
        col = X[:, f]
        n_unique = len(np.unique(col))
        if n_unique <= 1:
            X_disc[:, f] = 0
        else:
            actual_bins = min(n_bins, n_unique)
            try:
                kbd = KBinsDiscretizer(
                    n_bins=actual_bins,
                    encode="ordinal",
                    strategy="quantile",
                )
                X_disc[:, f] = (
                    kbd.fit_transform(col.reshape(-1, 1)).ravel().astype(int)
                )
            except ValueError:
                X_disc[:, f] = 0

    y_disc = y.astype(int)

    # Compute all pairwise synergies
    synergy_scores: dict[tuple[int, int], float] = {}
    for i in range(n_features):
        for j in range(i + 1, n_features):
            synergy_scores[(i, j)] = compute_pid_synergy(
                X_disc[:, i], X_disc[:, j], y_disc
            )

    # Build graph at threshold
    all_syns = list(synergy_scores.values())
    if not all_syns:
        return nx.Graph(), synergy_scores, 0.0

    threshold = float(np.percentile(all_syns, percentile_threshold))

    # Ensure at least 3 edges even if threshold is high
    edges_above = sum(1 for s in all_syns if s >= threshold)
    if edges_above < 3 and len(all_syns) >= 3:
        sorted_syns = sorted(all_syns, reverse=True)
        threshold = sorted_syns[min(2, len(sorted_syns) - 1)]

    G = nx.Graph()
    G.add_nodes_from(range(n_features))
    for (i, j), s in synergy_scores.items():
        if s >= threshold:
            G.add_edge(i, j, synergy=s)

    return G, synergy_scores, threshold


def get_candidate_subsets(
    G: nx.Graph,
    max_clique_size: int = 4,
) -> list[tuple[int, ...]]:
    """Extract candidate feature subsets from synergy graph.

    Candidates = all edges (size-2) + all triangles (size-3) +
                 all 4-cliques (size-4, if any)
    """
    candidates: set[tuple[int, ...]] = set()

    for clique in nx.enumerate_all_cliques(G):
        if len(clique) < 2:
            continue
        if len(clique) > max_clique_size:
            break
        candidates.add(tuple(sorted(clique)))

    return sorted(candidates)


# ============================================================
# PHASE 2: OBLIQUE FIGS IMPLEMENTATION
# ============================================================


class ObliqueSplitNode:
    """A node in an oblique FIGS tree."""

    def __init__(self) -> None:
        self.feature_indices: list[int] = []
        self.weights: np.ndarray = np.array([])
        self.threshold: float = 0.0
        self.value: np.ndarray = np.array([])  # leaf value (n_classes,)
        self.left: ObliqueSplitNode | None = None
        self.right: ObliqueSplitNode | None = None
        self.is_leaf: bool = True
        self.impurity_reduction: float = 0.0
        self.n_samples: int = 0


class ObliqueFIGSClassifier:
    """Oblique FIGS classifier with configurable feature subset selection.

    Implements the FIGS greedy algorithm with oblique splits:
    1. Start with root nodes as a pool of potential splits
    2. Greedily pick the split with highest impurity reduction
    3. Add its children to the pool
    4. Update residuals (subtract predictions from other trees)
    5. Repeat until max_rules reached
    """

    def __init__(
        self,
        max_rules: int = 10,
        candidate_mode: str = "synergy",
        synergy_graph: nx.Graph | None = None,
        synergy_candidates: list[tuple[int, ...]] | None = None,
        beam_size: int = 5,
        subset_size_range: tuple[int, int] = (2, 4),
        random_state: int = 42,
    ) -> None:
        self.max_rules = max_rules
        self.candidate_mode = candidate_mode
        self.synergy_graph = synergy_graph
        self.synergy_candidates = synergy_candidates or []
        self.beam_size = beam_size
        self.subset_size_range = subset_size_range
        self.random_state = random_state
        self.trees_: list[ObliqueSplitNode] = []
        self.complexity_: int = 0
        self.n_classes_: int = 0
        self.classes_: np.ndarray = np.array([])

    def _get_candidate_subsets(
        self,
        n_features: int,
        rng: np.random.RandomState,
    ) -> list[tuple[int, ...]]:
        """Get feature subsets to evaluate for oblique split."""
        if self.candidate_mode == "synergy":
            candidates = list(self.synergy_candidates)
            if len(candidates) > 50:
                idx = rng.choice(len(candidates), size=50, replace=False)
                candidates = [candidates[i] for i in idx]
            return candidates
        elif self.candidate_mode == "random":
            candidates: list[tuple[int, ...]] = []
            for _ in range(self.beam_size):
                size = rng.randint(
                    self.subset_size_range[0],
                    self.subset_size_range[1] + 1,
                )
                size = min(size, n_features)
                subset = tuple(sorted(rng.choice(n_features, size=size, replace=False)))
                candidates.append(subset)
            return candidates
        else:
            raise ValueError(f"Unknown candidate_mode: {self.candidate_mode}")

    def _find_best_oblique_split(
        self,
        X: np.ndarray,
        y_residuals: np.ndarray,
        idxs: np.ndarray,
        candidates: list[tuple[int, ...]],
    ) -> dict | None:
        """Find best oblique split among candidate feature subsets.

        For each candidate subset:
        1. Extract X[:, subset] for samples in idxs
        2. Fit Ridge regression -> get projection weights
        3. Project data, fit stump -> find threshold + impurity reduction
        4. Pick the combo with highest impurity reduction

        Also evaluates axis-aligned split as fallback.
        """
        X_node = X[idxs]
        y_node = y_residuals[idxs]

        if X_node.shape[0] < 4:
            return None

        n_outputs = y_node.shape[1] if y_node.ndim > 1 else 1
        if y_node.ndim == 1:
            y_node = y_node.reshape(-1, 1)

        best: dict | None = None
        best_imp_red = -np.inf

        for subset in candidates:
            subset_list = list(subset)
            X_sub = X_node[:, subset_list]

            if X_sub.shape[0] < 4:
                continue

            # Try each output column to find best projection direction
            for c in range(n_outputs):
                y_target_col = y_node[:, c]

                if np.std(y_target_col) < 1e-10:
                    continue

                # Fit Ridge to get weights for this subset
                try:
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_sub, y_target_col)
                except Exception:
                    continue

                weights = ridge.coef_
                if np.all(np.abs(weights) < 1e-10):
                    continue

                # Project
                proj_node = X_node[:, subset_list] @ weights

                # Find best threshold via stump on all residual outputs
                try:
                    stump = DecisionTreeRegressor(max_depth=1)
                    stump.fit(proj_node.reshape(-1, 1), y_node)
                except Exception:
                    continue

                if stump.tree_.feature[0] < 0:
                    continue

                imp = stump.tree_.impurity
                ns = stump.tree_.n_node_samples
                if len(imp) < 3:
                    continue

                imp_red = (
                    imp[0] - imp[1] * ns[1] / ns[0] - imp[2] * ns[2] / ns[0]
                ) * ns[0]

                if imp_red > best_imp_red:
                    best_imp_red = imp_red
                    threshold_val = stump.tree_.threshold[0]

                    # Compute masks on full X
                    proj_all = X[:, subset_list] @ weights
                    # But only within idxs
                    left_mask = np.zeros(len(X), dtype=bool)
                    right_mask = np.zeros(len(X), dtype=bool)
                    for ii in np.where(idxs)[0]:
                        if proj_all[ii] <= threshold_val:
                            left_mask[ii] = True
                        else:
                            right_mask[ii] = True

                    left_val = stump.tree_.value[1].flatten()
                    right_val = stump.tree_.value[2].flatten()

                    best = {
                        "feature_indices": subset_list,
                        "weights": weights.copy(),
                        "threshold": threshold_val,
                        "impurity_reduction": imp_red,
                        "left_value": left_val.copy(),
                        "right_value": right_val.copy(),
                        "left_idxs": left_mask.copy(),
                        "right_idxs": right_mask.copy(),
                    }

        # Also evaluate axis-aligned split as fallback
        try:
            stump_aa = DecisionTreeRegressor(max_depth=1)
            stump_aa.fit(X_node, y_node)
            if stump_aa.tree_.feature[0] >= 0 and len(stump_aa.tree_.impurity) >= 3:
                imp_aa = stump_aa.tree_.impurity
                ns_aa = stump_aa.tree_.n_node_samples
                imp_red_aa = (
                    imp_aa[0]
                    - imp_aa[1] * ns_aa[1] / ns_aa[0]
                    - imp_aa[2] * ns_aa[2] / ns_aa[0]
                ) * ns_aa[0]

                if imp_red_aa > best_imp_red:
                    feat = stump_aa.tree_.feature[0]
                    thresh = stump_aa.tree_.threshold[0]

                    left_mask = np.zeros(len(X), dtype=bool)
                    right_mask = np.zeros(len(X), dtype=bool)
                    for ii in np.where(idxs)[0]:
                        if X[ii, feat] <= thresh:
                            left_mask[ii] = True
                        else:
                            right_mask[ii] = True

                    best = {
                        "feature_indices": [feat],
                        "weights": np.array([1.0]),
                        "threshold": thresh,
                        "impurity_reduction": imp_red_aa,
                        "left_value": stump_aa.tree_.value[1].flatten().copy(),
                        "right_value": stump_aa.tree_.value[2].flatten().copy(),
                        "left_idxs": left_mask.copy(),
                        "right_idxs": right_mask.copy(),
                    }
        except Exception:
            pass

        return best

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ObliqueFIGSClassifier":
        """Fit ObliqueFIGS following the FIGS greedy algorithm.

        y is label-encoded. Internally one-hot encode for multi-output regression.
        """
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape

        # One-hot encode y for multi-output regression
        Y_onehot = np.zeros((n_samples, self.n_classes_))
        for i, c in enumerate(self.classes_):
            Y_onehot[y == c, i] = 1.0

        # Get candidate subsets once
        candidates = self._get_candidate_subsets(n_features, rng)
        if not candidates:
            # Fallback: create random subsets
            for _ in range(5):
                size = min(rng.randint(2, 4), n_features)
                s = tuple(sorted(rng.choice(n_features, size=size, replace=False)))
                candidates.append(s)

        # Initialize: one tree root per class? No — FIGS starts with one root
        # The "pool" is a list of (tree_index, node, idxs, residuals)
        # Each tree starts as a leaf node. The greedy loop finds the best
        # node to split and creates children.

        # Start with a single root tree
        root = ObliqueSplitNode()
        root.is_leaf = True
        all_idxs = np.ones(n_samples, dtype=bool)
        root.value = Y_onehot[all_idxs].mean(axis=0)
        root.n_samples = n_samples

        self.trees_ = [root]
        # Pool: list of (tree_idx, node, idxs_mask)
        pool: list[tuple[int, ObliqueSplitNode, np.ndarray]] = [
            (0, root, all_idxs.copy())
        ]

        total_splits = 0

        for _step in range(self.max_rules):
            if not pool:
                break

            # Compute current predictions from all trees
            preds = np.zeros((n_samples, self.n_classes_))
            for tree_root in self.trees_:
                preds += self._predict_tree(X, tree_root)

            # Residuals
            residuals = Y_onehot - preds

            best_split_info = None
            best_pool_idx = -1
            best_imp = -np.inf

            for pidx, (tree_idx, node, node_idxs) in enumerate(pool):
                if np.sum(node_idxs) < 4:
                    continue

                split_info = self._find_best_oblique_split(
                    X, residuals, node_idxs, candidates
                )
                if split_info is not None and split_info["impurity_reduction"] > best_imp:
                    best_imp = split_info["impurity_reduction"]
                    best_split_info = split_info
                    best_pool_idx = pidx

            if best_split_info is None or best_imp <= 0:
                break

            # Apply the best split
            tree_idx, node, node_idxs = pool[best_pool_idx]
            pool.pop(best_pool_idx)

            node.is_leaf = False
            node.feature_indices = best_split_info["feature_indices"]
            node.weights = best_split_info["weights"]
            node.threshold = best_split_info["threshold"]
            node.impurity_reduction = best_split_info["impurity_reduction"]

            left_idxs = best_split_info["left_idxs"]
            right_idxs = best_split_info["right_idxs"]

            # Create children
            left_child = ObliqueSplitNode()
            left_child.is_leaf = True
            n_left = np.sum(left_idxs)
            left_child.value = (
                residuals[left_idxs].mean(axis=0) if n_left > 0 else np.zeros(self.n_classes_)
            )
            left_child.n_samples = int(n_left)

            right_child = ObliqueSplitNode()
            right_child.is_leaf = True
            n_right = np.sum(right_idxs)
            right_child.value = (
                residuals[right_idxs].mean(axis=0) if n_right > 0 else np.zeros(self.n_classes_)
            )
            right_child.n_samples = int(n_right)

            node.left = left_child
            node.right = right_child

            total_splits += 1

            # Add children to pool if they have enough samples
            if n_left >= 4:
                pool.append((tree_idx, left_child, left_idxs))
            if n_right >= 4:
                pool.append((tree_idx, right_child, right_idxs))

            # Optionally start a new tree (every 3 splits, FIGS-style)
            if total_splits % 3 == 0 and total_splits < self.max_rules:
                new_root = ObliqueSplitNode()
                new_root.is_leaf = True
                new_root.value = residuals.mean(axis=0)
                new_root.n_samples = n_samples
                new_tree_idx = len(self.trees_)
                self.trees_.append(new_root)
                pool.append((new_tree_idx, new_root, np.ones(n_samples, dtype=bool)))

        self.complexity_ = total_splits
        return self

    def _predict_tree(self, X: np.ndarray, node: ObliqueSplitNode) -> np.ndarray:
        """Predict from a single tree, returning (n_samples, n_classes) values."""
        n_samples = X.shape[0]
        result = np.zeros((n_samples, self.n_classes_))

        def _recurse(
            nd: ObliqueSplitNode,
            sample_mask: np.ndarray,
        ) -> None:
            if nd.is_leaf:
                result[sample_mask] += nd.value
                return

            # Compute projection
            feat_idx = nd.feature_indices
            proj = X[sample_mask][:, feat_idx] @ nd.weights

            left_local = proj <= nd.threshold
            right_local = ~left_local

            # Map back to full indices
            full_indices = np.where(sample_mask)[0]
            left_mask = np.zeros(n_samples, dtype=bool)
            right_mask = np.zeros(n_samples, dtype=bool)
            left_mask[full_indices[left_local]] = True
            right_mask[full_indices[right_local]] = True

            if nd.left is not None and np.any(left_mask):
                _recurse(nd.left, left_mask)
            if nd.right is not None and np.any(right_mask):
                _recurse(nd.right, right_mask)

        all_mask = np.ones(n_samples, dtype=bool)
        _recurse(node, all_mask)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities via softmax of summed tree outputs."""
        n_samples = X.shape[0]
        raw = np.zeros((n_samples, self.n_classes_))
        for tree_root in self.trees_:
            raw += self._predict_tree(X, tree_root)

        # Softmax
        raw_shifted = raw - raw.max(axis=1, keepdims=True)
        exp_raw = np.exp(raw_shifted)
        proba = exp_raw / exp_raw.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by summing tree outputs, then argmax."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ============================================================
# PHASE 3: EVALUATION PROTOCOL
# ============================================================


def traverse_tree(node: ObliqueSplitNode) -> list[ObliqueSplitNode]:
    """Traverse tree and return all nodes."""
    nodes = [node]
    if node.left is not None:
        nodes.extend(traverse_tree(node.left))
    if node.right is not None:
        nodes.extend(traverse_tree(node.right))
    return nodes


def compute_split_interpretability_score(
    model: ObliqueFIGSClassifier,
    synergy_scores: dict[tuple[int, int], float],
    median_synergy: float,
) -> float:
    """Fraction of oblique splits whose feature pairs have above-median synergy.

    Axis-aligned splits (1 feature): counted as 1.0 (trivially interpretable).
    """
    n_interpretable = 0
    n_total = 0
    for tree in model.trees_:
        for node in traverse_tree(tree):
            if node.is_leaf:
                continue
            n_total += 1
            if len(node.feature_indices) == 1:
                n_interpretable += 1
            else:
                pairs_above = 0
                total_pairs = 0
                for i in range(len(node.feature_indices)):
                    for j in range(i + 1, len(node.feature_indices)):
                        fi, fj = sorted(
                            [node.feature_indices[i], node.feature_indices[j]]
                        )
                        total_pairs += 1
                        if synergy_scores.get((fi, fj), 0) >= median_synergy:
                            pairs_above += 1
                if total_pairs > 0 and pairs_above == total_pairs:
                    n_interpretable += 1
    return n_interpretable / max(n_total, 1)


def compute_mean_features_per_split(model: ObliqueFIGSClassifier) -> float:
    """Average number of features used per split."""
    counts = []
    for tree in model.trees_:
        for node in traverse_tree(tree):
            if not node.is_leaf:
                counts.append(len(node.feature_indices))
    return float(np.mean(counts)) if counts else 0.0


def compute_n_splits(model: ObliqueFIGSClassifier) -> int:
    """Total number of splits across all trees."""
    n = 0
    for tree in model.trees_:
        for node in traverse_tree(tree):
            if not node.is_leaf:
                n += 1
    return n


def safe_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
    classes: np.ndarray,
) -> float:
    """Compute AUC safely, handling edge cases."""
    try:
        if n_classes == 2:
            # Binary: use probability of positive class (class 1)
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                return float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                return float("nan")
        else:
            # Multiclass OVR
            # Check that test set has all classes
            present = np.unique(y_true)
            if len(present) < 2:
                return float("nan")
            y_bin = label_binarize(y_true, classes=classes)
            return float(
                roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
            )
    except Exception:
        return float("nan")


def count_figs_splits(model: FIGSClassifier) -> int:
    """Count total splits in an imodels FIGS model."""
    n = 0
    for tree in model.trees_:
        stack = [tree]
        while stack:
            node = stack.pop()
            if node.left is not None or node.right is not None:
                n += 1
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
    return n


# ============================================================
# PHASE 4: MAIN EXPERIMENT
# ============================================================


@logger.catch
def run_experiment() -> None:
    """Main experiment runner."""
    t_start = time.time()

    # Load data
    datasets = load_all_datasets(DATA_PATHS, max_examples=MAX_EXAMPLES)
    logger.info(f"Loaded {len(datasets)} datasets total")

    synergy_graph_stats: dict[str, dict] = {}

    # Per-dataset example-level predictions: ds_name -> list of example dicts
    # Each example = one test sample with predictions from all methods
    dataset_examples: dict[str, list[dict]] = {}

    # Per-fold aggregate metrics for summary
    fold_metrics: list[dict] = []

    # Sort by dataset size (smallest first) for gradual scaling
    dataset_order = sorted(datasets.keys(), key=lambda d: datasets[d]["X"].shape[0])
    logger.info(f"Dataset order: {dataset_order}")

    for ds_idx, ds_name in enumerate(dataset_order):
        ds = datasets[ds_name]
        X, y, folds = ds["X"], ds["y"], ds["folds"]
        feature_names = ds["feature_names"]
        class_names = ds["class_names"]
        n_classes = ds["n_classes"]
        classes_arr = np.arange(n_classes)

        logger.info(
            f"[{ds_idx+1}/{len(datasets)}] Processing {ds_name}: "
            f"{X.shape}, {n_classes} classes"
        )

        # Phase 1: Build synergy graph (once per dataset)
        t_syn = time.time()
        G, synergy_scores, syn_threshold = build_synergy_graph(X, y)
        candidates = get_candidate_subsets(G)
        all_syns = list(synergy_scores.values())
        median_synergy = float(np.median(all_syns)) if all_syns else 0.0
        syn_time = time.time() - t_syn

        synergy_graph_stats[ds_name] = {
            "n_edges": G.number_of_edges(),
            "n_candidates": len(candidates),
            "density": float(nx.density(G)),
            "threshold": float(syn_threshold),
            "median_synergy": median_synergy,
            "synergy_computation_time_seconds": round(syn_time, 2),
        }

        logger.info(
            f"  Synergy graph: {G.number_of_edges()} edges, "
            f"{len(candidates)} candidates, "
            f"threshold={syn_threshold:.4f}, time={syn_time:.1f}s"
        )

        # 5-fold CV using pre-assigned folds
        unique_folds = sorted(np.unique(folds))
        if len(unique_folds) != 5:
            logger.warning(
                f"  Expected 5 folds, got {len(unique_folds)} for {ds_name}"
            )

        # Collect per-sample predictions across all folds
        # Index by original sample index
        sample_preds: dict[int, dict[str, str]] = {}  # row_idx -> {method: pred_label}

        for fold_id in unique_folds:
            test_mask = folds == fold_id
            train_mask = ~test_mask
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            test_indices = np.where(test_mask)[0]

            logger.debug(
                f"  Fold {fold_id}: train={X_train.shape[0]}, test={X_test.shape[0]}"
            )

            # Initialize sample_preds for test samples
            for idx in test_indices:
                if idx not in sample_preds:
                    sample_preds[idx] = {}

            # ---- Method A: Standard FIGS (axis-aligned) ----
            best_figs_acc = -1.0
            best_figs_preds: np.ndarray | None = None
            best_figs_metrics: dict = {}
            best_max_rules = MAX_RULES_CANDIDATES[0]

            for max_rules in MAX_RULES_CANDIDATES:
                try:
                    figs = FIGSClassifier(max_rules=max_rules)
                    t0 = time.time()
                    figs.fit(X_train, y_train)
                    train_time = time.time() - t0

                    y_pred = figs.predict(X_test)
                    acc = float(accuracy_score(y_test, y_pred))
                    proba = figs.predict_proba(X_test)
                    auc_val = safe_auc(
                        y_true=y_test,
                        y_proba=proba,
                        n_classes=n_classes,
                        classes=classes_arr,
                    )
                    n_splits = count_figs_splits(figs)

                    if acc > best_figs_acc:
                        best_figs_acc = acc
                        best_max_rules = max_rules
                        best_figs_preds = y_pred.copy()
                        best_figs_metrics = {
                            "accuracy": round(acc, 6),
                            "auc": round(auc_val, 6) if not np.isnan(auc_val) else None,
                            "n_splits": n_splits,
                            "mean_features_per_split": 1.0,
                            "split_interpretability_score": 1.0,
                            "train_time_seconds": round(train_time, 4),
                        }
                except Exception:
                    logger.exception(
                        f"  FIGS failed on {ds_name} fold={fold_id} mr={max_rules}"
                    )

            if best_figs_preds is not None:
                for i, idx in enumerate(test_indices):
                    sample_preds[idx]["FIGS"] = class_names[best_figs_preds[i]]
                fold_metrics.append({
                    "dataset": ds_name, "fold": int(fold_id), "method": "FIGS",
                    **best_figs_metrics,
                })

            # ---- Method B: RO-FIGS (random oblique) ----
            try:
                ro_figs = ObliqueFIGSClassifier(
                    max_rules=best_max_rules,
                    candidate_mode="random",
                    beam_size=5,
                    subset_size_range=(2, 4),
                    random_state=42,
                )
                t0 = time.time()
                ro_figs.fit(X_train, y_train)
                train_time = time.time() - t0

                y_pred = ro_figs.predict(X_test)
                acc = float(accuracy_score(y_test, y_pred))
                proba = ro_figs.predict_proba(X_test)
                auc_val = safe_auc(
                    y_true=y_test,
                    y_proba=proba,
                    n_classes=n_classes,
                    classes=classes_arr,
                )

                for i, idx in enumerate(test_indices):
                    sample_preds[idx]["RO_FIGS"] = class_names[y_pred[i]]

                fold_metrics.append({
                    "dataset": ds_name, "fold": int(fold_id), "method": "RO-FIGS",
                    "accuracy": round(acc, 6),
                    "auc": round(auc_val, 6) if not np.isnan(auc_val) else None,
                    "n_splits": compute_n_splits(ro_figs),
                    "mean_features_per_split": round(
                        compute_mean_features_per_split(ro_figs), 3
                    ),
                    "split_interpretability_score": round(
                        compute_split_interpretability_score(
                            ro_figs, synergy_scores, median_synergy
                        ), 4,
                    ),
                    "train_time_seconds": round(train_time, 4),
                })
            except Exception:
                logger.exception(f"  RO-FIGS failed on {ds_name} fold={fold_id}")

            # ---- Method C: SG-FIGS (synergy-guided oblique) ----
            try:
                sg_figs = ObliqueFIGSClassifier(
                    max_rules=best_max_rules,
                    candidate_mode="synergy",
                    synergy_graph=G,
                    synergy_candidates=candidates,
                    random_state=42,
                )
                t0 = time.time()
                sg_figs.fit(X_train, y_train)
                train_time = time.time() - t0

                y_pred = sg_figs.predict(X_test)
                acc = float(accuracy_score(y_test, y_pred))
                proba = sg_figs.predict_proba(X_test)
                auc_val = safe_auc(
                    y_true=y_test,
                    y_proba=proba,
                    n_classes=n_classes,
                    classes=classes_arr,
                )

                for i, idx in enumerate(test_indices):
                    sample_preds[idx]["SG_FIGS"] = class_names[y_pred[i]]

                fold_metrics.append({
                    "dataset": ds_name, "fold": int(fold_id), "method": "SG-FIGS",
                    "accuracy": round(acc, 6),
                    "auc": round(auc_val, 6) if not np.isnan(auc_val) else None,
                    "n_splits": compute_n_splits(sg_figs),
                    "mean_features_per_split": round(
                        compute_mean_features_per_split(sg_figs), 3
                    ),
                    "split_interpretability_score": round(
                        compute_split_interpretability_score(
                            sg_figs, synergy_scores, median_synergy
                        ), 4,
                    ),
                    "train_time_seconds": round(train_time, 4),
                })
            except Exception:
                logger.exception(f"  SG-FIGS failed on {ds_name} fold={fold_id}")

            # ---- Method D: GradientBoosting baseline ----
            try:
                gb = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    random_state=42,
                )
                t0 = time.time()
                gb.fit(X_train, y_train)
                train_time = time.time() - t0

                y_pred = gb.predict(X_test)
                acc = float(accuracy_score(y_test, y_pred))
                proba = gb.predict_proba(X_test)
                auc_val = safe_auc(
                    y_true=y_test,
                    y_proba=proba,
                    n_classes=n_classes,
                    classes=classes_arr,
                )

                for i, idx in enumerate(test_indices):
                    sample_preds[idx]["GradientBoosting"] = class_names[y_pred[i]]

                fold_metrics.append({
                    "dataset": ds_name, "fold": int(fold_id), "method": "GradientBoosting",
                    "accuracy": round(acc, 6),
                    "auc": round(auc_val, 6) if not np.isnan(auc_val) else None,
                    "n_splits": 100 * (2**3 - 1),
                    "mean_features_per_split": 1.0,
                    "split_interpretability_score": None,
                    "train_time_seconds": round(train_time, 4),
                })
            except Exception:
                logger.exception(f"  GradientBoosting failed on {ds_name} fold={fold_id}")

        # Build examples for this dataset: one example per sample
        examples: list[dict] = []
        for idx in sorted(sample_preds.keys()):
            # Reconstruct input from features
            input_dict = {fname: round(float(X[idx, fi]), 6) for fi, fname in enumerate(feature_names)}
            input_str = json.dumps(input_dict)

            # True label
            true_label = class_names[y[idx]]

            # Build example dict
            ex: dict = {
                "input": input_str,
                "output": true_label,
                "metadata_fold": int(folds[idx]),
                "metadata_row_index": int(idx),
            }

            # Add predict_* fields for each method
            preds = sample_preds[idx]
            if "FIGS" in preds:
                ex["predict_FIGS"] = preds["FIGS"]
            if "RO_FIGS" in preds:
                ex["predict_RO_FIGS"] = preds["RO_FIGS"]
            if "SG_FIGS" in preds:
                ex["predict_SG_FIGS"] = preds["SG_FIGS"]
            if "GradientBoosting" in preds:
                ex["predict_GradientBoosting"] = preds["GradientBoosting"]

            examples.append(ex)

        dataset_examples[ds_name] = examples
        logger.success(f"Completed {ds_name}: {len(examples)} test examples")

    total_runtime = time.time() - t_start

    # ---- Build summary: mean +/- std across folds ----
    summary: list[dict] = []
    for ds_name in dataset_order:
        for method in METHODS:
            rows = [
                r for r in fold_metrics
                if r["dataset"] == ds_name and r["method"] == method
            ]
            if not rows:
                continue

            accs = [r["accuracy"] for r in rows]
            aucs = [r["auc"] for r in rows if r["auc"] is not None]
            n_splits_list = [r["n_splits"] for r in rows]
            interp_list = [
                r["split_interpretability_score"]
                for r in rows
                if r["split_interpretability_score"] is not None
            ]
            mfps_list = [r["mean_features_per_split"] for r in rows]
            times_list = [r["train_time_seconds"] for r in rows]

            summary.append({
                "dataset": ds_name,
                "method": method,
                "accuracy_mean": round(float(np.mean(accs)), 6),
                "accuracy_std": round(float(np.std(accs)), 6),
                "auc_mean": round(float(np.mean(aucs)), 6) if aucs else None,
                "auc_std": round(float(np.std(aucs)), 6) if aucs else None,
                "n_splits_mean": round(float(np.mean(n_splits_list)), 2),
                "n_splits_std": round(float(np.std(n_splits_list)), 2),
                "mean_features_per_split_mean": round(float(np.mean(mfps_list)), 3),
                "interpretability_mean": round(float(np.mean(interp_list)), 4) if interp_list else None,
                "interpretability_std": round(float(np.std(interp_list)), 4) if interp_list else None,
                "train_time_mean": round(float(np.mean(times_list)), 4),
            })

    # ---- Build output in exp_gen_sol_out schema ----
    # Each example = one test sample with input, output (true label),
    # and predict_* fields for each method's prediction
    output_datasets: list[dict] = []

    for ds_name in dataset_order:
        if ds_name in dataset_examples and dataset_examples[ds_name]:
            output_datasets.append({
                "dataset": ds_name,
                "examples": dataset_examples[ds_name],
            })

    # Add fold-level metrics as a separate dataset entry
    metrics_examples: list[dict] = []
    for m in fold_metrics:
        input_str = json.dumps({
            "dataset": m["dataset"],
            "fold": m["fold"],
            "method": m["method"],
        })
        output_str = json.dumps({
            "accuracy": m["accuracy"],
            "auc": m["auc"],
            "n_splits": m["n_splits"],
            "mean_features_per_split": m["mean_features_per_split"],
            "split_interpretability_score": m["split_interpretability_score"],
            "train_time_seconds": m["train_time_seconds"],
        })
        metrics_examples.append({
            "input": input_str,
            "output": output_str,
            "predict_metric_result": output_str,
            "metadata_fold": m["fold"],
            "metadata_method": m["method"],
            "metadata_dataset": m["dataset"],
        })

    if metrics_examples:
        output_datasets.append({
            "dataset": "fold_metrics",
            "examples": metrics_examples,
        })

    # Add summary as a separate dataset entry
    summary_examples: list[dict] = []
    for s in summary:
        input_str = json.dumps({
            "dataset": s["dataset"],
            "method": s["method"],
        })
        output_str = json.dumps({
            "accuracy_mean": s["accuracy_mean"],
            "accuracy_std": s["accuracy_std"],
            "auc_mean": s["auc_mean"],
            "auc_std": s["auc_std"],
            "n_splits_mean": s["n_splits_mean"],
            "n_splits_std": s["n_splits_std"],
            "mean_features_per_split_mean": s["mean_features_per_split_mean"],
            "interpretability_mean": s["interpretability_mean"],
            "interpretability_std": s["interpretability_std"],
            "train_time_mean": s["train_time_mean"],
        })
        summary_examples.append({
            "input": input_str,
            "output": output_str,
            "predict_summary_result": output_str,
            "metadata_dataset": s["dataset"],
            "metadata_method": s["method"],
        })

    if summary_examples:
        output_datasets.append({
            "dataset": "summary",
            "examples": summary_examples,
        })

    # Add synergy graph stats as a dataset entry
    synergy_examples: list[dict] = []
    for ds_name_key, stats in synergy_graph_stats.items():
        input_str = json.dumps({"dataset": ds_name_key})
        output_str = json.dumps(stats)
        synergy_examples.append({
            "input": input_str,
            "output": output_str,
            "predict_synergy_stats": output_str,
            "metadata_dataset": ds_name_key,
        })

    if synergy_examples:
        output_datasets.append({
            "dataset": "synergy_graph_stats",
            "examples": synergy_examples,
        })

    final_output = {"datasets": output_datasets}

    # Save
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(final_output, indent=2))
    logger.success(f"Saved results to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Print summary table
    logger.info("=" * 70)
    logger.info("SUMMARY (mean accuracy across 5 folds)")
    logger.info(f"{'Dataset':<25} {'FIGS':>8} {'RO-FIGS':>8} {'SG-FIGS':>8} {'GB':>8}")
    logger.info("-" * 70)
    for ds_name in dataset_order:
        row = f"{ds_name:<25}"
        for method in METHODS:
            matches = [
                s for s in summary
                if s["dataset"] == ds_name and s["method"] == method
            ]
            if matches:
                row += f" {matches[0]['accuracy_mean']:>7.4f}"
            else:
                row += f" {'N/A':>7}"
        logger.info(row)

    logger.info("=" * 70)
    logger.info(f"Total runtime: {total_runtime:.1f}s")


if __name__ == "__main__":
    run_experiment()
