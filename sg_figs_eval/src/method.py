#!/usr/bin/env python3
"""SG-FIGS: Fold-Aware Synergy-Guided Oblique Splits with Threshold Ablation.

Re-implements and evaluates SG-FIGS with fold-aware synergy graphs (no leakage),
three synergy threshold levels, non-circular interpretability scoring, statistical
significance testing (Wilcoxon + Friedman/Nemenyi), and qualitative oblique split
inspection across 12 tabular classification benchmarks.

Methods compared:
  1. FIGS (standard axis-aligned)
  2. RO-FIGS (random oblique splits)
  3. SG-FIGS-10 (synergy threshold p90 — aggressive)
  4. SG-FIGS-25 (synergy threshold p75 — moderate)
  5. SG-FIGS-50 (synergy threshold p50 — permissive)
  6. GradientBoosting (non-interpretable baseline)
"""

# ── NUMPY MONKEY-PATCH (must be before dit import) ──────────────────────────
import numpy as np
if not hasattr(np, 'alltrue'):
    np.alltrue = np.all
if not hasattr(np, 'cumproduct'):
    np.cumproduct = np.cumprod
if not hasattr(np, 'product'):
    np.product = np.prod

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import json
import sys
import time
import resource
import warnings
from pathlib import Path

from loguru import logger
import dit
from dit.pid import PID_WB
from imodels import FIGSClassifier
from imodels.tree.figs import Node
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import (
    KBinsDiscretizer, StandardScaler, OrdinalEncoder, LabelEncoder
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, ttest_rel
from scipy.stats import studentized_range

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ── RESOURCE LIMITS ──────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))  # 14GB RAM
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))  # ~58 min CPU

# ── LOGGING ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")

# ── CONSTANTS ────────────────────────────────────────────────────────────────
DATASETS = {
    'breast_cancer': {'source': 'sklearn', 'domain': 'medical'},
    'wine': {'source': 'sklearn', 'domain': 'chemistry'},
    'diabetes': {'source': 'openml', 'openml_id': 37, 'domain': 'medical'},
    'heart_statlog': {'source': 'openml', 'openml_id': 53, 'domain': 'medical'},
    'ionosphere': {'source': 'openml', 'openml_id': 59, 'domain': 'physics'},
    'sonar': {'source': 'openml', 'openml_id': 40, 'domain': 'signal'},
    'vehicle': {'source': 'openml', 'openml_id': 54, 'domain': 'vision'},
    'segment': {'source': 'openml', 'openml_id': 36, 'domain': 'vision'},
    'glass': {'source': 'openml', 'openml_id': 41, 'domain': 'forensics'},
    'banknote': {'source': 'openml', 'openml_id': 1462, 'domain': 'image'},
    'credit_g': {'source': 'openml', 'openml_id': 31, 'domain': 'finance'},
    'australian': {'source': 'openml', 'openml_id': 40981, 'domain': 'finance'},
}

DATASET_ORDER = [
    'banknote', 'wine', 'glass',
    'diabetes', 'heart_statlog', 'sonar',
    'breast_cancer', 'ionosphere', 'vehicle',
    'segment', 'credit_g', 'australian',
]

THRESHOLDS = {'SG-FIGS-10': 90, 'SG-FIGS-25': 75, 'SG-FIGS-50': 50}
MAX_RULES_VALUES = [5, 10, 15]
N_FOLDS = 5
PER_DATASET_TIMEOUT = 600  # 10 min


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(name: str, info: dict) -> tuple:
    """Load and preprocess dataset. Returns X (float64), y (int), feature_names."""
    if info['source'] == 'sklearn':
        loader = load_breast_cancer if name == 'breast_cancer' else load_wine
        data = loader()
        X, y, feat_names = data.data, data.target, list(data.feature_names)
    else:
        d = fetch_openml(data_id=info['openml_id'], as_frame=True, parser='auto')
        df = d.frame
        target_col = d.target.name
        feat_names = [c for c in df.columns if c != target_col]
        X_df = df[feat_names]
        y_raw = df[target_col]

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        cat_cols = X_df.select_dtypes(include=['category', 'object']).columns.tolist()

        X = np.zeros((len(X_df), len(feat_names)), dtype=float)
        for i, col in enumerate(feat_names):
            if col in cat_cols:
                oe = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                )
                X[:, i] = oe.fit_transform(X_df[[col]]).ravel()
            else:
                vals = X_df[col].values
                # Handle potential NaN
                vals = np.where(np.isnan(vals.astype(float)), 0.0, vals.astype(float))
                X[:, i] = vals

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.astype(int)
    return X, y, feat_names


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: SYNERGY GRAPH CONSTRUCTION (FOLD-AWARE)
# ═══════════════════════════════════════════════════════════════════════════

def compute_pairwise_synergy_matrix(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_bins: int = 5,
    max_features: int = 30,
) -> tuple:
    """Compute pairwise PID synergy for all feature pairs using ONLY training data.

    For datasets with >max_features features, pre-filters to top features by
    individual mutual information to keep computation tractable.

    Returns synergy_matrix (p×p symmetric), computation_time, feature_mask.
    """
    t0 = time.time()
    p = X_train.shape[1]

    # Pre-filter features for high-dimensional datasets
    feature_mask = np.arange(p)
    if p > max_features:
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(
            X_train, y_train, random_state=42, n_neighbors=5,
        )
        top_idx = np.argsort(mi_scores)[-max_features:]
        feature_mask = np.sort(top_idx)
        logger.debug(f"    Pre-filtered {p} features to {max_features} by MI")

    # Discretize on training data only
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    try:
        X_disc = kbd.fit_transform(X_train).astype(int)
    except ValueError:
        # Fallback: if quantile fails (constant features), use uniform
        kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        X_disc = kbd.fit_transform(X_train).astype(int)
    y_disc = y_train.astype(int)

    synergy_matrix = np.zeros((p, p))
    for ii in range(len(feature_mask)):
        i = feature_mask[ii]
        for jj in range(ii + 1, len(feature_mask)):
            j = feature_mask[jj]
            try:
                syn = _compute_single_synergy(X_disc[:, i], X_disc[:, j], y_disc)
            except Exception:
                syn = 0.0
            synergy_matrix[i, j] = syn
            synergy_matrix[j, i] = syn

    elapsed = time.time() - t0
    return synergy_matrix, elapsed


def _compute_single_synergy(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute PID synergy between two discretized features and target."""
    probs: dict = {}
    for idx in range(len(y)):
        key = (int(x1[idx]), int(x2[idx]), int(y[idx]))
        probs[key] = probs.get(key, 0) + 1
    total = sum(probs.values())
    if total == 0:
        return 0.0
    outcomes = list(probs.keys())
    pmf = [probs[o] / total for o in outcomes]

    d = dit.Distribution(outcomes, pmf)
    pid = PID_WB(d, [[0], [1]], [2])
    syn = pid.get_pi(pid._lattice.top)
    return float(syn)


def compute_fast_synergy_proxy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """Fast MI-based synergy proxy for validation set.

    Synergy ≈ I(X_i, X_j; Y) - I(X_i; Y) - I(X_j; Y)
    This is the interaction information, which approximates synergy.
    Much faster than full PID computation.
    """
    from sklearn.metrics import mutual_info_score
    p = X_train.shape[1]

    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    try:
        X_disc = kbd.fit_transform(X_train).astype(int)
    except ValueError:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        X_disc = kbd.fit_transform(X_train).astype(int)
    y_disc = y_train.astype(int)

    # Compute individual MI(X_i; Y)
    mi_individual = np.zeros(p)
    for i in range(p):
        mi_individual[i] = mutual_info_score(X_disc[:, i], y_disc)

    # Compute pairwise joint MI(X_i, X_j; Y)
    synergy_proxy = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            # Joint variable: combine bins
            joint = X_disc[:, i] * (n_bins + 1) + X_disc[:, j]
            mi_joint = mutual_info_score(joint.astype(int), y_disc)
            # Interaction information ≈ synergy
            interaction = mi_joint - mi_individual[i] - mi_individual[j]
            synergy_proxy[i, j] = max(interaction, 0.0)
            synergy_proxy[j, i] = synergy_proxy[i, j]

    return synergy_proxy


def build_synergy_graph(
    synergy_matrix: np.ndarray,
    threshold_percentile: float,
) -> tuple:
    """Build adjacency from synergy matrix by keeping edges above the given percentile.

    Returns adjacency dict, n_edges, cutoff value.
    """
    p = synergy_matrix.shape[0]
    upper_tri = synergy_matrix[np.triu_indices(p, k=1)]

    if len(upper_tri) == 0 or np.all(upper_tri == 0):
        return {i: set() for i in range(p)}, 0, 0.0

    cutoff = float(np.percentile(upper_tri, threshold_percentile))

    # Minimum absolute synergy to avoid noise edges
    MIN_SYNERGY = 1e-6

    adj: dict = {i: set() for i in range(p)}
    n_edges = 0
    for i in range(p):
        for j in range(i + 1, p):
            if synergy_matrix[i, j] >= cutoff and synergy_matrix[i, j] > MIN_SYNERGY:
                adj[i].add(j)
                adj[j].add(i)
                n_edges += 1

    return adj, n_edges, cutoff


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: SG-FIGS IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

class SynergyGuidedFIGS(FIGSClassifier):
    """FIGS with synergy-guided oblique splits."""

    def __init__(
        self,
        synergy_adj: dict,
        synergy_matrix: np.ndarray,
        max_rules: int = 12,
        ridge_alpha: float = 1.0,
        max_features_per_split: int = 5,
        random_state: int = None,
    ):
        super().__init__(max_rules=max_rules, random_state=random_state)
        self.synergy_adj = synergy_adj
        self.synergy_matrix = synergy_matrix
        self.ridge_alpha = ridge_alpha
        self.max_features_per_split = max_features_per_split
        self.oblique_splits_info: list = []

    def _construct_node_with_stump(
        self, X, y, idxs, tree_num,
        sample_weight=None, compare_nodes_with_sample_weight=True,
        max_features=None, depth=None,
    ):
        """Override: try synergy-guided oblique splits alongside axis-aligned."""
        node_axis = super()._construct_node_with_stump(
            X, y, idxs, tree_num,
            sample_weight=sample_weight,
            compare_nodes_with_sample_weight=compare_nodes_with_sample_weight,
            max_features=max_features,
            depth=depth,
        )

        if not hasattr(node_axis, 'left_temp') or node_axis.left_temp is None:
            node_axis.is_oblique = False
            return node_axis

        best_node = node_axis
        best_impurity_reduction = node_axis.impurity_reduction or 0.0

        best_feature = node_axis.feature
        if best_feature is not None and best_feature in self.synergy_adj:
            neighbors = self.synergy_adj[best_feature]
            if len(neighbors) > 0:
                feat_subset = sorted([best_feature] + list(neighbors))

                if len(feat_subset) > self.max_features_per_split:
                    syn_scores = [
                        (f, self.synergy_matrix[best_feature, f])
                        for f in neighbors
                    ]
                    syn_scores.sort(key=lambda x: x[1], reverse=True)
                    feat_subset = [best_feature] + [
                        f for f, _ in syn_scores[:self.max_features_per_split - 1]
                    ]

                if len(feat_subset) >= 2:
                    oblique_node = self._try_oblique_split(
                        X, y, idxs, tree_num, feat_subset,
                        sample_weight=sample_weight,
                        depth=depth,
                    )
                    if oblique_node is not None:
                        obl_imp = oblique_node.impurity_reduction or 0.0
                        if obl_imp > best_impurity_reduction:
                            best_node = oblique_node
                            best_impurity_reduction = obl_imp

        if not hasattr(best_node, 'is_oblique'):
            best_node.is_oblique = False
        return best_node

    def _try_oblique_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        idxs: np.ndarray,
        tree_num: int,
        feat_subset: list,
        sample_weight: np.ndarray = None,
        depth: int = None,
    ):
        """Try an oblique split using Ridge on the feature subset."""
        try:
            X_sub = X[idxs][:, feat_subset]
            y_sub = y[idxs]
            if y_sub.ndim > 1:
                y_fit = y_sub[:, 0]
            else:
                y_fit = y_sub.copy().astype(float)

            if len(np.unique(y_fit)) < 2:
                return None

            ridge = Ridge(alpha=self.ridge_alpha)
            ridge.fit(X_sub, y_fit)
            weights = ridge.coef_
            intercept = float(ridge.intercept_)
            projection = X_sub @ weights + intercept

            dt = DecisionTreeRegressor(max_depth=1)
            sweight = None
            if sample_weight is not None:
                sweight = sample_weight[idxs]
            dt.fit(projection.reshape(-1, 1), y_sub, sample_weight=sweight)

            if dt.tree_.feature[0] == -2 or len(dt.tree_.feature) < 3:
                return None

            threshold = float(dt.tree_.threshold[0])
            impurity = dt.tree_.impurity
            n_samples = dt.tree_.n_node_samples

            if sample_weight is not None:
                proj_full = X[idxs][:, feat_subset] @ weights + intercept
                idxs_left_mask = proj_full <= threshold
                n_left = sample_weight[idxs][idxs_left_mask].sum()
                n_right = sample_weight[idxs][~idxs_left_mask].sum()
                n_total = n_left + n_right
            else:
                n_left = n_samples[1]
                n_right = n_samples[2]
                n_total = n_samples[0]

            if n_total == 0:
                return None

            imp_red = (
                impurity[0]
                - impurity[1] * n_left / n_total
                - impurity[2] * n_right / n_total
            ) * n_total

            if imp_red <= 0:
                return None

            # Compute full projection for all samples in idxs
            proj_full = X[idxs][:, feat_subset] @ weights + intercept

            idxs_left = idxs.copy()
            idxs_right = idxs.copy()
            idxs_left[idxs] = proj_full <= threshold
            idxs_right[idxs] = proj_full > threshold

            node_oblique = Node(
                idxs=idxs,
                value=dt.tree_.value[0],
                tree_num=tree_num,
                feature=feat_subset[0],
                threshold=threshold,
                impurity=float(impurity[0]),
                impurity_reduction=float(imp_red),
                depth=depth,
            )
            node_oblique.is_oblique = True
            node_oblique.oblique_features = feat_subset
            node_oblique.oblique_weights = weights.tolist()
            node_oblique.oblique_bias = intercept

            node_oblique.setattrs(
                left_temp=Node(
                    idxs=idxs_left,
                    value=dt.tree_.value[1],
                    tree_num=tree_num,
                    depth=(depth or 0) + 1,
                ),
                right_temp=Node(
                    idxs=idxs_right,
                    value=dt.tree_.value[2],
                    tree_num=tree_num,
                    depth=(depth or 0) + 1,
                ),
            )

            self.oblique_splits_info.append({
                'features': feat_subset,
                'weights': weights.tolist(),
                'threshold': float(threshold),
                'impurity_reduction': float(imp_red),
            })

            return node_oblique

        except Exception:
            return None

    def _predict_tree(self, root: Node, X: np.ndarray) -> np.ndarray:
        """Override to handle oblique splits during prediction."""
        def _predict_single(node, x):
            if node.left is None and node.right is None:
                return node.value

            if getattr(node, 'is_oblique', False):
                proj = sum(
                    w * x[f]
                    for w, f in zip(node.oblique_weights, node.oblique_features)
                )
                proj += node.oblique_bias
                go_left = proj <= node.threshold
            else:
                go_left = x[node.feature] <= node.threshold

            if go_left:
                return _predict_single(node.left, x) if node.left else node.value
            else:
                return _predict_single(node.right, x) if node.right else node.value

        preds = np.zeros((X.shape[0], self.n_outputs))
        for i in range(X.shape[0]):
            preds[i] = _predict_single(root, X[i])
        return preds


class RandomObliqueFIGS(FIGSClassifier):
    """Baseline: oblique splits with RANDOM feature subsets (no synergy guidance)."""

    def __init__(
        self,
        beam_size: int = 3,
        max_rules: int = 12,
        ridge_alpha: float = 1.0,
        random_state: int = None,
    ):
        super().__init__(max_rules=max_rules, random_state=random_state)
        self.beam_size = beam_size
        self.ridge_alpha = ridge_alpha
        self.oblique_splits_info: list = []
        self._rng = np.random.RandomState(random_state)

    def _construct_node_with_stump(
        self, X, y, idxs, tree_num,
        sample_weight=None, compare_nodes_with_sample_weight=True,
        max_features=None, depth=None,
    ):
        node_axis = super()._construct_node_with_stump(
            X, y, idxs, tree_num,
            sample_weight=sample_weight,
            compare_nodes_with_sample_weight=compare_nodes_with_sample_weight,
            max_features=max_features,
            depth=depth,
        )

        if not hasattr(node_axis, 'left_temp') or node_axis.left_temp is None:
            node_axis.is_oblique = False
            return node_axis

        best_node = node_axis
        best_imp_red = node_axis.impurity_reduction or 0.0

        p = X.shape[1]
        for _ in range(3):  # r=3 repetitions
            feat_size = min(self.beam_size, p)
            if feat_size < 2:
                continue
            feat_subset = sorted(
                self._rng.choice(p, size=feat_size, replace=False).tolist()
            )

            oblique_node = self._try_oblique_split(
                X, y, idxs, tree_num, feat_subset,
                sample_weight=sample_weight,
                depth=depth,
            )
            if oblique_node is not None:
                obl_imp = oblique_node.impurity_reduction or 0.0
                if obl_imp > best_imp_red:
                    best_node = oblique_node
                    best_imp_red = obl_imp

        if not hasattr(best_node, 'is_oblique'):
            best_node.is_oblique = False
        return best_node

    def _try_oblique_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        idxs: np.ndarray,
        tree_num: int,
        feat_subset: list,
        sample_weight: np.ndarray = None,
        depth: int = None,
    ):
        """Try an oblique split with random feature subset."""
        try:
            X_sub = X[idxs][:, feat_subset]
            y_sub = y[idxs]
            if y_sub.ndim > 1:
                y_fit = y_sub[:, 0]
            else:
                y_fit = y_sub.copy().astype(float)

            if len(np.unique(y_fit)) < 2:
                return None

            ridge = Ridge(alpha=self.ridge_alpha)
            ridge.fit(X_sub, y_fit)
            weights = ridge.coef_
            intercept = float(ridge.intercept_)
            projection = X_sub @ weights + intercept

            dt = DecisionTreeRegressor(max_depth=1)
            sweight = None
            if sample_weight is not None:
                sweight = sample_weight[idxs]
            dt.fit(projection.reshape(-1, 1), y_sub, sample_weight=sweight)

            if dt.tree_.feature[0] == -2 or len(dt.tree_.feature) < 3:
                return None

            threshold = float(dt.tree_.threshold[0])
            impurity = dt.tree_.impurity
            n_samples = dt.tree_.n_node_samples

            if sample_weight is not None:
                proj_full = X[idxs][:, feat_subset] @ weights + intercept
                idxs_left_mask = proj_full <= threshold
                n_left = sample_weight[idxs][idxs_left_mask].sum()
                n_right = sample_weight[idxs][~idxs_left_mask].sum()
                n_total = n_left + n_right
            else:
                n_left = n_samples[1]
                n_right = n_samples[2]
                n_total = n_samples[0]

            if n_total == 0:
                return None

            imp_red = (
                impurity[0]
                - impurity[1] * n_left / n_total
                - impurity[2] * n_right / n_total
            ) * n_total

            if imp_red <= 0:
                return None

            proj_full = X[idxs][:, feat_subset] @ weights + intercept
            idxs_left = idxs.copy()
            idxs_right = idxs.copy()
            idxs_left[idxs] = proj_full <= threshold
            idxs_right[idxs] = proj_full > threshold

            node_oblique = Node(
                idxs=idxs,
                value=dt.tree_.value[0],
                tree_num=tree_num,
                feature=feat_subset[0],
                threshold=threshold,
                impurity=float(impurity[0]),
                impurity_reduction=float(imp_red),
                depth=depth,
            )
            node_oblique.is_oblique = True
            node_oblique.oblique_features = feat_subset
            node_oblique.oblique_weights = weights.tolist()
            node_oblique.oblique_bias = intercept

            node_oblique.setattrs(
                left_temp=Node(
                    idxs=idxs_left,
                    value=dt.tree_.value[1],
                    tree_num=tree_num,
                    depth=(depth or 0) + 1,
                ),
                right_temp=Node(
                    idxs=idxs_right,
                    value=dt.tree_.value[2],
                    tree_num=tree_num,
                    depth=(depth or 0) + 1,
                ),
            )

            self.oblique_splits_info.append({
                'features': feat_subset,
                'weights': weights.tolist(),
                'threshold': float(threshold),
                'impurity_reduction': float(imp_red),
            })

            return node_oblique

        except Exception:
            return None

    def _predict_tree(self, root: Node, X: np.ndarray) -> np.ndarray:
        """Override to handle oblique splits during prediction."""
        def _predict_single(node, x):
            if node.left is None and node.right is None:
                return node.value

            if getattr(node, 'is_oblique', False):
                proj = sum(
                    w * x[f]
                    for w, f in zip(node.oblique_weights, node.oblique_features)
                )
                proj += node.oblique_bias
                go_left = proj <= node.threshold
            else:
                go_left = x[node.feature] <= node.threshold

            if go_left:
                return _predict_single(node.left, x) if node.left else node.value
            else:
                return _predict_single(node.right, x) if node.right else node.value

        preds = np.zeros((X.shape[0], self.n_outputs))
        for i in range(X.shape[0]):
            preds[i] = _predict_single(root, X[i])
        return preds


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def compute_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute AUC. Handles binary and multi-class via OVR."""
    try:
        n_classes = y_proba.shape[1]
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            return float(roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='weighted'
            ))
    except Exception:
        return float('nan')


def count_splits_in_tree(node) -> tuple:
    """Count total splits and oblique splits in a tree.

    Returns (total_splits, oblique_splits).
    """
    if node is None or (node.left is None and node.right is None):
        return 0, 0

    is_obl = 1 if getattr(node, 'is_oblique', False) else 0
    left_total, left_obl = count_splits_in_tree(node.left)
    right_total, right_obl = count_splits_in_tree(node.right)
    return 1 + left_total + right_total, is_obl + left_obl + right_obl


def count_model_splits(model) -> tuple:
    """Count total and oblique splits across all trees in a FIGS model."""
    total = 0
    oblique = 0
    for tree in model.trees_:
        t, o = count_splits_in_tree(tree)
        total += t
        oblique += o
    return total, oblique


def compute_mean_features_per_oblique(model) -> float:
    """Mean number of features used per oblique split."""
    def _collect(node, counts):
        if node is None:
            return
        if getattr(node, 'is_oblique', False):
            counts.append(len(node.oblique_features))
        _collect(node.left, counts)
        _collect(node.right, counts)

    all_counts: list = []
    for tree in model.trees_:
        _collect(tree, all_counts)
    if not all_counts:
        return 0.0
    return float(np.mean(all_counts))


def compute_interpretability_score(
    model,
    synergy_matrix_validate: np.ndarray,
) -> float:
    """Non-circular interpretability score.

    Fraction of oblique splits whose feature pairs ALL rank in top-25%
    of synergy scores computed on the VALIDATE subset (not the build subset).
    """
    def _collect_oblique_features(node, oblique_feats):
        if node is None:
            return
        if getattr(node, 'is_oblique', False):
            oblique_feats.append(node.oblique_features)
        _collect_oblique_features(node.left, oblique_feats)
        _collect_oblique_features(node.right, oblique_feats)

    oblique_feats: list = []
    for tree in model.trees_:
        _collect_oblique_features(tree, oblique_feats)

    if not oblique_feats:
        return float('nan')

    # Get top-25% synergy threshold from validate matrix
    p = synergy_matrix_validate.shape[0]
    upper_tri = synergy_matrix_validate[np.triu_indices(p, k=1)]
    if len(upper_tri) == 0 or np.all(upper_tri == 0):
        return 0.0
    top25_cutoff = float(np.percentile(upper_tri, 75))

    n_aligned = 0
    for feat_list in oblique_feats:
        # Check all pairs in this oblique split
        all_high = True
        for i in range(len(feat_list)):
            for j in range(i + 1, len(feat_list)):
                if synergy_matrix_validate[feat_list[i], feat_list[j]] < top25_cutoff:
                    all_high = False
                    break
            if not all_high:
                break
        if all_high:
            n_aligned += 1

    return float(n_aligned / len(oblique_feats))


def extract_split_descriptions(
    model,
    feat_names: list,
    synergy_matrix: np.ndarray,
) -> list:
    """Extract human-readable descriptions of splits for qualitative inspection."""
    descriptions: list = []

    def _describe(node, tree_idx, split_idx):
        if node is None or (node.left is None and node.right is None):
            return split_idx
        desc = {
            'tree': tree_idx,
            'split_index': split_idx,
            'is_oblique': getattr(node, 'is_oblique', False),
            'depth': node.depth,
            'impurity_reduction': float(node.impurity_reduction or 0),
        }
        if getattr(node, 'is_oblique', False):
            desc['type'] = 'oblique'
            desc['features'] = [
                feat_names[f] if f < len(feat_names) else f"feat_{f}"
                for f in node.oblique_features
            ]
            desc['feature_indices'] = node.oblique_features
            desc['weights'] = node.oblique_weights
            desc['bias'] = node.oblique_bias
            desc['threshold'] = float(node.threshold)
            # Synergy between features
            feats = node.oblique_features
            synergies = []
            for fi in range(len(feats)):
                for fj in range(fi + 1, len(feats)):
                    synergies.append({
                        'pair': [feat_names[feats[fi]], feat_names[feats[fj]]],
                        'synergy': float(synergy_matrix[feats[fi], feats[fj]]),
                    })
            desc['pairwise_synergies'] = synergies
            desc['rule_str'] = (
                " + ".join(
                    f"{w:.3f}*{feat_names[f] if f < len(feat_names) else f'feat_{f}'}"
                    for w, f in zip(node.oblique_weights, node.oblique_features)
                )
                + f" + {node.oblique_bias:.3f} <= {node.threshold:.3f}"
            )
        else:
            desc['type'] = 'axis_aligned'
            fname = feat_names[node.feature] if node.feature < len(feat_names) else f"feat_{node.feature}"
            desc['feature'] = fname
            desc['feature_index'] = node.feature
            desc['threshold'] = float(node.threshold)
            desc['rule_str'] = f"{fname} <= {node.threshold:.3f}"

        descriptions.append(desc)
        split_idx = _describe(node.left, tree_idx, split_idx + 1)
        split_idx = _describe(node.right, tree_idx, split_idx)
        return split_idx

    for ti, tree in enumerate(model.trees_):
        _describe(tree, ti, 0)

    return descriptions


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: MAIN EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    global_start = time.time()
    logger.info("=" * 60)
    logger.info("SG-FIGS Experiment — Starting")
    logger.info("=" * 60)

    results: list = []
    synergy_stability: dict = {}
    qualitative_splits: dict = {}
    dataset_timings: dict = {}

    method_names = [
        'FIGS', 'RO-FIGS',
        'SG-FIGS-10', 'SG-FIGS-25', 'SG-FIGS-50',
        'GradientBoosting',
    ]

    # ── MAIN LOOP OVER DATASETS ──────────────────────────────────────────
    for ds_idx, ds_name in enumerate(DATASET_ORDER):
        ds_start = time.time()
        logger.info(f"[{ds_idx+1}/{len(DATASET_ORDER)}] Processing dataset: {ds_name}")

        try:
            X, y, feat_names = load_dataset(ds_name, DATASETS[ds_name])
        except Exception:
            logger.exception(f"Failed to load dataset {ds_name}")
            continue

        logger.info(f"  Shape: {X.shape}, classes: {len(np.unique(y))}, features: {len(feat_names)}")
        n_features = X.shape[1]
        n_pairs = n_features * (n_features - 1) // 2
        logger.info(f"  Feature pairs for synergy: {n_pairs}")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_synergy_graphs: list = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            fold_start = time.time()
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # ── FOLD-AWARE SYNERGY COMPUTATION ──
            n_train = len(train_idx)
            rng = np.random.RandomState(42 + fold_idx)
            perm = rng.permutation(n_train)
            split_pt = int(n_train * 0.8)
            syn_build_idx = perm[:split_pt]
            syn_validate_idx = perm[split_pt:]

            X_syn_build = X_train[syn_build_idx]
            y_syn_build = y_train[syn_build_idx]
            X_syn_validate = X_train[syn_validate_idx]
            y_syn_validate = y_train[syn_validate_idx]

            # Compute synergy on synergy-build subset ONLY
            synergy_matrix, syn_time = compute_pairwise_synergy_matrix(
                X_syn_build, y_syn_build,
            )
            fold_synergy_graphs.append(synergy_matrix)
            logger.debug(f"  Fold {fold_idx}: synergy computed in {syn_time:.1f}s")

            # Compute synergy on validate subset for interpretability
            # Use fast MI-based approximation to avoid doubling PID cost
            synergy_matrix_validate = compute_fast_synergy_proxy(
                X_syn_validate, y_syn_validate,
            )

            for max_rules in MAX_RULES_VALUES:
                # ── METHOD 1: Standard FIGS ──
                try:
                    figs = FIGSClassifier(
                        max_rules=max_rules, random_state=42,
                    )
                    figs.fit(X_train, y_train)
                    y_pred_figs = figs.predict(X_test)
                    y_proba_figs = figs.predict_proba(X_test)
                    acc_figs = float(accuracy_score(y_test, y_pred_figs))
                    auc_figs = compute_auc(y_test, y_proba_figs)
                    n_splits_figs, _ = count_model_splits(figs)

                    results.append({
                        'method': 'FIGS',
                        'dataset': ds_name,
                        'fold': fold_idx,
                        'max_rules': max_rules,
                        'accuracy': acc_figs,
                        'auc': auc_figs,
                        'n_splits': n_splits_figs,
                        'n_oblique': 0,
                        'oblique_fraction': 0.0,
                        'mean_features_per_oblique': 0.0,
                        'interpretability_score': float('nan'),
                        'synergy_time_s': 0.0,
                        'n_synergy_edges': 0,
                        'synergy_cutoff': 0.0,
                    })
                except Exception:
                    logger.exception(f"FIGS failed: {ds_name} fold={fold_idx} mr={max_rules}")

                # ── METHOD 2: RO-FIGS ──
                try:
                    rofigs = RandomObliqueFIGS(
                        beam_size=3, max_rules=max_rules, random_state=42,
                    )
                    rofigs.fit(X_train, y_train)
                    y_pred_ro = rofigs.predict(X_test)
                    y_proba_ro = rofigs.predict_proba(X_test)
                    acc_ro = float(accuracy_score(y_test, y_pred_ro))
                    auc_ro = compute_auc(y_test, y_proba_ro)
                    n_splits_ro, n_obl_ro = count_model_splits(rofigs)

                    results.append({
                        'method': 'RO-FIGS',
                        'dataset': ds_name,
                        'fold': fold_idx,
                        'max_rules': max_rules,
                        'accuracy': acc_ro,
                        'auc': auc_ro,
                        'n_splits': n_splits_ro,
                        'n_oblique': n_obl_ro,
                        'oblique_fraction': n_obl_ro / max(n_splits_ro, 1),
                        'mean_features_per_oblique': compute_mean_features_per_oblique(rofigs),
                        'interpretability_score': float('nan'),
                        'synergy_time_s': 0.0,
                        'n_synergy_edges': 0,
                        'synergy_cutoff': 0.0,
                    })
                except Exception:
                    logger.exception(f"RO-FIGS failed: {ds_name} fold={fold_idx} mr={max_rules}")

                # ── METHODS 3-5: SG-FIGS variants ──
                for sg_name, percentile in THRESHOLDS.items():
                    try:
                        adj, n_edges, cutoff = build_synergy_graph(
                            synergy_matrix, percentile,
                        )
                        sgfigs = SynergyGuidedFIGS(
                            synergy_adj=adj,
                            synergy_matrix=synergy_matrix,
                            max_rules=max_rules,
                            random_state=42,
                        )
                        sgfigs.fit(X_train, y_train)
                        y_pred_sg = sgfigs.predict(X_test)
                        y_proba_sg = sgfigs.predict_proba(X_test)
                        acc_sg = float(accuracy_score(y_test, y_pred_sg))
                        auc_sg = compute_auc(y_test, y_proba_sg)
                        n_splits_sg, n_obl_sg = count_model_splits(sgfigs)

                        interp_score = compute_interpretability_score(
                            sgfigs, synergy_matrix_validate,
                        )

                        results.append({
                            'method': sg_name,
                            'dataset': ds_name,
                            'fold': fold_idx,
                            'max_rules': max_rules,
                            'accuracy': acc_sg,
                            'auc': auc_sg,
                            'n_splits': n_splits_sg,
                            'n_oblique': n_obl_sg,
                            'oblique_fraction': n_obl_sg / max(n_splits_sg, 1),
                            'mean_features_per_oblique': compute_mean_features_per_oblique(sgfigs),
                            'interpretability_score': interp_score,
                            'synergy_time_s': syn_time,
                            'n_synergy_edges': n_edges,
                            'synergy_cutoff': cutoff,
                        })
                    except Exception:
                        logger.exception(f"{sg_name} failed: {ds_name} fold={fold_idx} mr={max_rules}")

                # ── METHOD 6: GradientBoosting baseline ──
                try:
                    gbc = GradientBoostingClassifier(
                        n_estimators=100, max_depth=3, random_state=42,
                    )
                    gbc.fit(X_train, y_train)
                    y_pred_gb = gbc.predict(X_test)
                    y_proba_gb = gbc.predict_proba(X_test)
                    acc_gb = float(accuracy_score(y_test, y_pred_gb))
                    auc_gb = compute_auc(y_test, y_proba_gb)

                    results.append({
                        'method': 'GradientBoosting',
                        'dataset': ds_name,
                        'fold': fold_idx,
                        'max_rules': max_rules,
                        'accuracy': acc_gb,
                        'auc': auc_gb,
                        'n_splits': -1,
                        'n_oblique': 0,
                        'oblique_fraction': 0.0,
                        'mean_features_per_oblique': 0.0,
                        'interpretability_score': float('nan'),
                        'synergy_time_s': 0.0,
                        'n_synergy_edges': 0,
                        'synergy_cutoff': 0.0,
                    })
                except Exception:
                    logger.exception(f"GBC failed: {ds_name} fold={fold_idx} mr={max_rules}")

            # ── QUALITATIVE SPLIT INSPECTION ──
            if fold_idx == N_FOLDS - 1 and ds_name in ['diabetes', 'breast_cancer', 'heart_statlog']:
                for sg_name, percentile in THRESHOLDS.items():
                    try:
                        adj, _, _ = build_synergy_graph(synergy_matrix, percentile)
                        sgfigs_q = SynergyGuidedFIGS(
                            synergy_adj=adj,
                            synergy_matrix=synergy_matrix,
                            max_rules=10,
                            random_state=42,
                        )
                        sgfigs_q.fit(X_train, y_train)
                        splits_desc = extract_split_descriptions(
                            sgfigs_q, feat_names, synergy_matrix,
                        )
                        qualitative_splits[f'{ds_name}_{sg_name}'] = splits_desc
                    except Exception:
                        logger.exception(f"Qual split failed: {ds_name} {sg_name}")

            fold_elapsed = time.time() - fold_start
            logger.debug(f"  Fold {fold_idx} completed in {fold_elapsed:.1f}s")

        # ── SYNERGY GRAPH STABILITY (Jaccard) ──
        jaccard_pairs: list = []
        for fi in range(len(fold_synergy_graphs)):
            for fj in range(fi + 1, len(fold_synergy_graphs)):
                adj_i, _, _ = build_synergy_graph(fold_synergy_graphs[fi], 75)
                adj_j, _, _ = build_synergy_graph(fold_synergy_graphs[fj], 75)
                edges_i = {(a, b) for a in adj_i for b in adj_i[a] if a < b}
                edges_j = {(a, b) for a in adj_j for b in adj_j[a] if a < b}
                union = edges_i | edges_j
                if len(union) > 0:
                    jacc = len(edges_i & edges_j) / len(union)
                else:
                    jacc = 1.0
                jaccard_pairs.append(jacc)

        synergy_stability[ds_name] = {
            'mean_jaccard': float(np.mean(jaccard_pairs)) if jaccard_pairs else 0.0,
            'std_jaccard': float(np.std(jaccard_pairs)) if jaccard_pairs else 0.0,
            'all_jaccards': [float(j) for j in jaccard_pairs],
        }

        ds_elapsed = time.time() - ds_start
        dataset_timings[ds_name] = ds_elapsed
        logger.info(f"  Dataset {ds_name} done in {ds_elapsed:.1f}s | "
                     f"Results so far: {len(results)}")

        if ds_elapsed > PER_DATASET_TIMEOUT:
            logger.warning(f"  {ds_name} exceeded timeout ({ds_elapsed:.0f}s > {PER_DATASET_TIMEOUT}s)")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 5: STATISTICAL TESTS
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("Computing statistical tests...")

    stat_tests: dict = {}

    # Helper to get fold-level scores for a method at max_rules=10
    def get_scores(method_name: str, mr: int = 10) -> list:
        return [
            r['accuracy']
            for r in results
            if r['method'] == method_name and r['max_rules'] == mr
        ]

    def get_dataset_fold_scores(
        ds: str, method_name: str, mr: int = 10,
    ) -> list:
        return [
            r['accuracy']
            for r in results
            if r['method'] == method_name and r['max_rules'] == mr and r['dataset'] == ds
        ]

    # 5a. Wilcoxon signed-rank tests
    sg25_scores = get_scores('SG-FIGS-25', mr=10)
    rofigs_scores = get_scores('RO-FIGS', mr=10)
    figs_scores = get_scores('FIGS', mr=10)

    try:
        if len(sg25_scores) == len(rofigs_scores) and len(sg25_scores) > 0:
            sg25_arr = np.array(sg25_scores)
            ro_arr = np.array(rofigs_scores)
            diff_ro = sg25_arr - ro_arr
            if np.any(diff_ro != 0):
                wstat_ro, wp_ro = wilcoxon(sg25_arr, ro_arr, alternative='two-sided')
                N_ro = np.sum(diff_ro != 0)
                z_ro = (wstat_ro - N_ro * (N_ro + 1) / 4) / np.sqrt(
                    N_ro * (N_ro + 1) * (2 * N_ro + 1) / 24
                )
                effect_ro = float(abs(z_ro) / np.sqrt(N_ro)) if N_ro > 0 else 0.0
            else:
                wstat_ro, wp_ro, effect_ro = 0.0, 1.0, 0.0
        else:
            wstat_ro, wp_ro, effect_ro = 0.0, 1.0, 0.0
        stat_tests['wilcoxon_sg25_vs_rofigs'] = {
            'statistic': float(wstat_ro),
            'p_value': float(wp_ro),
            'effect_size_r': effect_ro,
            'n_samples': len(sg25_scores),
        }
    except Exception:
        logger.exception("Wilcoxon SG25 vs RO-FIGS failed")
        stat_tests['wilcoxon_sg25_vs_rofigs'] = {'error': 'computation_failed'}

    try:
        if len(sg25_scores) == len(figs_scores) and len(sg25_scores) > 0:
            sg25_arr = np.array(sg25_scores)
            figs_arr = np.array(figs_scores)
            diff_f = sg25_arr - figs_arr
            if np.any(diff_f != 0):
                wstat_f, wp_f = wilcoxon(sg25_arr, figs_arr, alternative='two-sided')
                N_f = np.sum(diff_f != 0)
                z_f = (wstat_f - N_f * (N_f + 1) / 4) / np.sqrt(
                    N_f * (N_f + 1) * (2 * N_f + 1) / 24
                )
                effect_f = float(abs(z_f) / np.sqrt(N_f)) if N_f > 0 else 0.0
            else:
                wstat_f, wp_f, effect_f = 0.0, 1.0, 0.0
        else:
            wstat_f, wp_f, effect_f = 0.0, 1.0, 0.0
        stat_tests['wilcoxon_sg25_vs_figs'] = {
            'statistic': float(wstat_f),
            'p_value': float(wp_f),
            'effect_size_r': effect_f,
            'n_samples': len(sg25_scores),
        }
    except Exception:
        logger.exception("Wilcoxon SG25 vs FIGS failed")
        stat_tests['wilcoxon_sg25_vs_figs'] = {'error': 'computation_failed'}

    # 5b. Per-dataset paired t-tests with Bonferroni correction
    n_comparisons = len(DATASET_ORDER)
    per_dataset_tests: dict = {}
    for ds_name in DATASET_ORDER:
        try:
            ds_sg25 = get_dataset_fold_scores(ds_name, 'SG-FIGS-25', mr=10)
            ds_rofigs = get_dataset_fold_scores(ds_name, 'RO-FIGS', mr=10)
            if len(ds_sg25) == len(ds_rofigs) and len(ds_sg25) > 1:
                stat_t, p_t = ttest_rel(ds_sg25, ds_rofigs)
                per_dataset_tests[ds_name] = {
                    'statistic': float(stat_t),
                    'p_value': float(p_t),
                    'p_corrected': float(min(p_t * n_comparisons, 1.0)),
                    'sg25_mean': float(np.mean(ds_sg25)),
                    'rofigs_mean': float(np.mean(ds_rofigs)),
                    'diff_mean': float(np.mean(np.array(ds_sg25) - np.array(ds_rofigs))),
                }
            else:
                per_dataset_tests[ds_name] = {'error': 'insufficient_data'}
        except Exception:
            per_dataset_tests[ds_name] = {'error': 'computation_failed'}
    stat_tests['per_dataset_ttests'] = per_dataset_tests

    # 5c. Friedman test + Nemenyi post-hoc
    try:
        datasets_with_data = []
        for ds_name in DATASET_ORDER:
            has_all = True
            for m in method_names:
                scores = get_dataset_fold_scores(ds_name, m, mr=10)
                if len(scores) == 0:
                    has_all = False
                    break
            if has_all:
                datasets_with_data.append(ds_name)

        N_ds = len(datasets_with_data)
        k = len(method_names)

        if N_ds >= 3:
            mean_scores = np.zeros((N_ds, k))
            for di, ds_name in enumerate(datasets_with_data):
                for mi, m in enumerate(method_names):
                    scores = get_dataset_fold_scores(ds_name, m, mr=10)
                    mean_scores[di, mi] = np.mean(scores) if scores else 0.0

            friedman_stat, friedman_p = friedmanchisquare(
                *[mean_scores[:, i] for i in range(k)]
            )

            # Nemenyi CD
            try:
                q_alpha = studentized_range.isf(0.05, k, np.inf) / np.sqrt(2)
            except Exception:
                # Fallback: use tabulated value for k=6
                q_alpha = 2.85  # approximate for k=6, alpha=0.05
            CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N_ds))

            # Mean ranks
            ranks = np.zeros_like(mean_scores)
            for i in range(N_ds):
                ranks[i] = rankdata(-mean_scores[i])  # lower rank = better
            mean_ranks = ranks.mean(axis=0)

            stat_tests['friedman'] = {
                'chi_sq': float(friedman_stat),
                'p_value': float(friedman_p),
                'n_datasets': N_ds,
            }
            stat_tests['nemenyi_cd'] = float(CD)
            stat_tests['mean_ranks'] = {
                name: float(r) for name, r in zip(method_names, mean_ranks)
            }
        else:
            stat_tests['friedman'] = {'error': 'insufficient_datasets', 'n': N_ds}
    except Exception:
        logger.exception("Friedman/Nemenyi failed")
        stat_tests['friedman'] = {'error': 'computation_failed'}

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 6: BUILD OUTPUT JSON (exp_gen_sol_out schema)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("Building output JSON...")

    total_runtime = time.time() - global_start

    # Build per-dataset examples for the schema
    output_datasets: list = []
    for ds_name in DATASET_ORDER:
        ds_results = [r for r in results if r['dataset'] == ds_name]
        if not ds_results:
            continue

        examples: list = []
        for r in ds_results:
            # Input: JSON string describing the experimental configuration
            input_str = json.dumps({
                'method': r['method'],
                'dataset': r['dataset'],
                'fold': r['fold'],
                'max_rules': r['max_rules'],
            })

            # Output: JSON string of results
            output_str = json.dumps({
                'accuracy': r['accuracy'],
                'auc': r['auc'],
                'n_splits': r['n_splits'],
                'n_oblique': r['n_oblique'],
                'oblique_fraction': r['oblique_fraction'],
                'mean_features_per_oblique': r['mean_features_per_oblique'],
                'interpretability_score': r['interpretability_score'],
            })

            example = {
                'input': input_str,
                'output': output_str,
                'metadata_method': r['method'],
                'metadata_dataset': r['dataset'],
                'metadata_fold': r['fold'],
                'metadata_max_rules': r['max_rules'],
                'metadata_accuracy': r['accuracy'],
                'metadata_auc': r['auc'],
                'metadata_n_splits': r['n_splits'],
                'metadata_n_oblique': r['n_oblique'],
                'metadata_oblique_fraction': r['oblique_fraction'],
                'metadata_mean_features_per_oblique': r['mean_features_per_oblique'],
                'metadata_interpretability_score': r['interpretability_score'],
                'metadata_synergy_time_s': r['synergy_time_s'],
                'metadata_n_synergy_edges': r['n_synergy_edges'],
                'metadata_synergy_cutoff': r['synergy_cutoff'],
            }

            # Add predict_ fields
            example['predict_method_accuracy'] = str(r['accuracy'])
            example['predict_method_auc'] = str(r['auc'])

            examples.append(example)

        output_datasets.append({
            'dataset': ds_name,
            'examples': examples,
        })

    output = {
        'datasets': output_datasets,
    }

    # Save main output
    output_path = WORKSPACE / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved main output to {output_path}")

    # Also save detailed analysis as separate metadata file
    analysis = {
        'experiment': 'SG-FIGS_fold_aware_comparison',
        'n_datasets': len(DATASET_ORDER),
        'n_folds': N_FOLDS,
        'methods': method_names,
        'max_rules_values': MAX_RULES_VALUES,
        'threshold_percentiles': {k: v for k, v in THRESHOLDS.items()},
        'total_results': len(results),
        'synergy_graph_stability': synergy_stability,
        'statistical_tests': stat_tests,
        'qualitative_split_inspection': qualitative_splits,
        'dataset_timings': dataset_timings,
        'total_runtime_seconds': total_runtime,
    }

    # Compute summary table
    summary_table: dict = {}
    for m in method_names:
        m_results = [r for r in results if r['method'] == m and r['max_rules'] == 10]
        if m_results:
            accs = [r['accuracy'] for r in m_results]
            aucs = [r['auc'] for r in m_results if not np.isnan(r['auc'])]
            summary_table[m] = {
                'mean_accuracy': float(np.mean(accs)),
                'std_accuracy': float(np.std(accs)),
                'mean_auc': float(np.mean(aucs)) if aucs else float('nan'),
                'std_auc': float(np.std(aucs)) if aucs else float('nan'),
                'n_results': len(m_results),
            }
    analysis['summary_table'] = summary_table

    analysis_path = WORKSPACE / "analysis_metadata.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, default=str))
    logger.info(f"Saved analysis metadata to {analysis_path}")

    # ── LOG SUMMARY ──
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT COMPLETE — Total runtime: {total_runtime:.1f}s")
    logger.info(f"Total results: {len(results)}")
    logger.info("")
    logger.info("Summary (max_rules=10, mean accuracy):")
    for m in method_names:
        if m in summary_table:
            s = summary_table[m]
            logger.info(f"  {m:20s}: {s['mean_accuracy']:.4f} ± {s['std_accuracy']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
