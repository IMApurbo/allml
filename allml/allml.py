# allml.py — v1.0.2 

import os
import sys
import warnings
import pickle
import json
import inspect
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_recall_curve, silhouette_score, davies_bouldin_score,
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, SGDRegressor, SGDClassifier,
    PassiveAggressiveRegressor, PassiveAggressiveClassifier,
    Perceptron, HuberRegressor, RANSACRegressor,
    TheilSenRegressor, ARDRegression,
    Lars, LassoLars, RidgeClassifier,
)
from sklearn.ensemble import (
    RandomForestRegressor,     RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor,         AdaBoostClassifier,
    ExtraTreesRegressor,       ExtraTreesClassifier,
    BaggingRegressor,          BaggingClassifier,
)
from sklearn.tree import (
    DecisionTreeRegressor, DecisionTreeClassifier,
    ExtraTreeRegressor,    ExtraTreeClassifier,
)
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC, NuSVR, NuSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import (
    GaussianProcessRegressor, GaussianProcessClassifier,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, MeanShift,
    SpectralClustering, Birch, MiniBatchKMeans,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.isotonic import IsotonicRegression

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# ============================================================
# Terminal colors
# ============================================================
class Colors:
    HEADER    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    END       = '\033[0m'
    MAGENTA   = '\033[35m'
    WHITE     = '\033[97m'


def cprint(text, color=Colors.WHITE, bold=False):
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.END}")


def banner():
    print(f"""
{Colors.CYAN}{Colors.BOLD}
AllML  -  All-in-One Machine Learning Library
                           Author - @IMApurbo
{Colors.END}""")


# ============================================================
# Task-type constants
# ============================================================
class TaskType:
    REGRESSION                  = "regression"
    BINARY_CLASSIFICATION       = "binary_classification"
    MULTICLASS_CLASSIFICATION   = "multiclass_classification"
    MULTI_OUTPUT_REGRESSION     = "multi_output_regression"
    MULTI_OUTPUT_CLASSIFICATION = "multi_output_classification"
    CLUSTERING                  = "clustering"


# ============================================================
# Parameter-mapping constants
# ============================================================

# Models that accept a plain float learning_rate
_FLOAT_LR: set = {
    "gradient_boosting_regressor",
    "gradient_boosting_classifier",
    "adaboost_regressor",
    "adaboost_classifier",
}
if XGBOOST_AVAILABLE:
    _FLOAT_LR |= {"xgboost_regressor",  "xgboost_classifier"}
if LIGHTGBM_AVAILABLE:
    _FLOAT_LR |= {"lightgbm_regressor", "lightgbm_classifier"}
if CATBOOST_AVAILABLE:
    _FLOAT_LR |= {"catboost_regressor", "catboost_classifier"}

# Models whose learning_rate param is a string schedule
_SGD_MODELS: set = {"sgd_regressor", "sgd_classifier"}
_MLP_MODELS: set = {"mlp_regressor", "mlp_classifier"}

# GradientBoosting alpha must stay in (0,1) - skip user value
_SKIP_ALPHA: set = {
    "gradient_boosting_regressor",
    "gradient_boosting_classifier",
}

# SVM models that accept a kernel string
_SVM_MODELS: set = {
    "svr", "svc", "linear_svr", "linear_svc", "nu_svr", "nu_svc",
}

# Nu-SVM models that need auto nu tuning
_NU_MODELS: set = {"nu_svc", "nu_svr"}


# ============================================================
# Nu-SVM safe nu finder
# ============================================================
def _find_safe_nu(y_train: np.ndarray, task: str,
                  candidates: tuple = (0.1, 0.2, 0.3, 0.4, 0.5,
                                       0.6, 0.7, 0.8, 0.9)) -> float:
    """
    For NuSVC / NuSVR the parameter nu must satisfy:
        0 < nu <= min_class_fraction   (classification)
        0 < nu <= 1                    (regression, any value works)

    We probe candidate values and return the largest one that is
    guaranteed to be feasible.  Falls back to 0.1 if nothing works.
    """
    if task in (TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION):
        # For regression nu in (0,1] is always valid; use 0.5
        return 0.5

    # Classification: nu must be <= fraction of the smallest class
    y_flat = np.array(y_train).ravel()
    classes, counts = np.unique(y_flat, return_counts=True)
    min_fraction = counts.min() / len(y_flat)

    # Pick largest candidate that is strictly less than min_fraction
    safe = 0.1
    for c in sorted(candidates):
        if c < min_fraction:
            safe = c
    return safe


# ============================================================
# AllML
# ============================================================
class AllML:
    """
    AllML - All-in-One Machine Learning Library
    ============================================
    Usage:
        from allml import AllML

        ml = AllML("data.csv")
        ml.col_to_feed(["f1","f2"]).col_to_pred(["target"])
        ml.method("random_forest_regressor").split(70, 20, 10)
        ml.train(n_estimators=100)
        ml.show()
        ml.show_graph("all")
        ml.save("models/mymodel")
        pred = ml.pred(val1, val2)
        ml.load("models/mymodel.allml")
    """

    # ----------------------------------------------------------
    # Method registries
    # ----------------------------------------------------------
    REGRESSION_METHODS: Dict[str, Any] = {
        "linear_regression"           : LinearRegression,
        "ridge"                        : Ridge,
        "lasso"                        : Lasso,
        "elastic_net"                  : ElasticNet,
        "bayesian_ridge"               : BayesianRidge,
        "sgd_regressor"                : SGDRegressor,
        "huber_regressor"              : HuberRegressor,
        "ransac_regressor"             : RANSACRegressor,
        "theilsen_regressor"           : TheilSenRegressor,
        "ard_regression"               : ARDRegression,
        "passive_aggressive_regressor" : PassiveAggressiveRegressor,
        "decision_tree_regressor"      : DecisionTreeRegressor,
        "extra_tree_regressor"         : ExtraTreeRegressor,
        "random_forest_regressor"      : RandomForestRegressor,
        "extra_trees_regressor"        : ExtraTreesRegressor,
        "gradient_boosting_regressor"  : GradientBoostingRegressor,
        "adaboost_regressor"           : AdaBoostRegressor,
        "bagging_regressor"            : BaggingRegressor,
        "svr"                          : SVR,
        "linear_svr"                   : LinearSVR,
        "nu_svr"                       : NuSVR,
        "knn_regressor"                : KNeighborsRegressor,
        "mlp_regressor"                : MLPRegressor,
        "gaussian_process_regressor"   : GaussianProcessRegressor,
        "pls_regression"               : PLSRegression,
        "isotonic_regression"          : IsotonicRegression,
        "lars"                         : Lars,
        "lasso_lars"                   : LassoLars,
    }

    CLASSIFICATION_METHODS: Dict[str, Any] = {
        "logistic_regression"           : LogisticRegression,
        "ridge_classifier"              : RidgeClassifier,
        "sgd_classifier"                : SGDClassifier,
        "passive_aggressive_classifier" : PassiveAggressiveClassifier,
        "perceptron"                    : Perceptron,
        "decision_tree_classifier"      : DecisionTreeClassifier,
        "extra_tree_classifier"         : ExtraTreeClassifier,
        "random_forest_classifier"      : RandomForestClassifier,
        "extra_trees_classifier"        : ExtraTreesClassifier,
        "gradient_boosting_classifier"  : GradientBoostingClassifier,
        "adaboost_classifier"           : AdaBoostClassifier,
        "bagging_classifier"            : BaggingClassifier,
        "svc"                           : SVC,
        "linear_svc"                    : LinearSVC,
        "nu_svc"                        : NuSVC,
        "knn_classifier"                : KNeighborsClassifier,
        "gaussian_nb"                   : GaussianNB,
        "bernoulli_nb"                  : BernoulliNB,
        "mlp_classifier"                : MLPClassifier,
        "gaussian_process_classifier"   : GaussianProcessClassifier,
        "lda"                           : LinearDiscriminantAnalysis,
        "qda"                           : QuadraticDiscriminantAnalysis,
    }

    CLUSTERING_METHODS: Dict[str, Any] = {
        "kmeans"                   : KMeans,
        "dbscan"                   : DBSCAN,
        "agglomerative_clustering" : AgglomerativeClustering,
        "mean_shift"               : MeanShift,
        "spectral_clustering"      : SpectralClustering,
        "birch"                    : Birch,
        "mini_batch_kmeans"        : MiniBatchKMeans,
    }

    if XGBOOST_AVAILABLE:
        REGRESSION_METHODS["xgboost_regressor"]       = XGBRegressor
        CLASSIFICATION_METHODS["xgboost_classifier"]  = XGBClassifier
    if LIGHTGBM_AVAILABLE:
        REGRESSION_METHODS["lightgbm_regressor"]      = LGBMRegressor
        CLASSIFICATION_METHODS["lightgbm_classifier"] = LGBMClassifier
    if CATBOOST_AVAILABLE:
        REGRESSION_METHODS["catboost_regressor"]      = CatBoostRegressor
        CLASSIFICATION_METHODS["catboost_classifier"] = CatBoostClassifier

    GRAPH_OPTIONS = [
        "confusion_matrix", "roc_curve", "precision_recall_curve",
        "feature_importance", "learning_curve", "residuals",
        "prediction_vs_actual", "error_distribution",
        "correlation_heatmap", "pairplot", "distribution",
        "boxplot", "cluster_plot", "model_comparison", "all",
    ]

    # ----------------------------------------------------------
    # Constructor
    # ----------------------------------------------------------
    def __init__(self, csv_filepath: str):
        banner()
        cprint(f"Loading dataset: {csv_filepath}", Colors.CYAN, bold=True)

        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"File not found: {csv_filepath}")

        self._filepath       = csv_filepath
        self._df             = pd.read_csv(csv_filepath)
        self._feed_cols      : List[str]       = []
        self._pred_cols      : List[str]       = []
        self._method_names   : List[str]       = []
        self._task_type      : Optional[str]   = None
        self._models         : Dict[str, Any]  = {}
        self._metrics        : Dict[str, Dict] = {}
        self._scaler_X       = StandardScaler()
        self._scaler_y       = StandardScaler()
        self._label_encoders : Dict[str, LabelEncoder] = {}
        self._is_trained     = False
        self._split_done     = False
        self._train_kwargs   : Dict = {}

        self._X_train = self._X_test = self._X_val = None
        self._y_train = self._y_test = self._y_val = None
        self._X_raw   = self._y_raw = None
        self._meta    : Dict[str, Any] = {}

        cprint(f"Dataset loaded successfully.", Colors.GREEN, bold=True)
        cprint(f"  Rows    : {self._df.shape[0]}", Colors.YELLOW)
        cprint(f"  Columns : {self._df.shape[1]}", Colors.YELLOW)
        cprint(f"  Names   : {list(self._df.columns)}", Colors.YELLOW)
        print()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def col_to_feed(self, columns: Union[str, List[str]]) -> "AllML":
        """Set feature (input) columns."""
        if isinstance(columns, str):
            columns = [columns]
        missing = [c for c in columns if c not in self._df.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}\n"
                f"Available: {list(self._df.columns)}"
            )
        self._feed_cols = columns
        cprint(f"Feature columns set: {columns}", Colors.GREEN)
        return self

    def col_to_pred(self, columns: Union[str, List[str]]) -> "AllML":
        """Set target (output) columns."""
        if isinstance(columns, str):
            columns = [columns]
        missing = [c for c in columns if c not in self._df.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}\n"
                f"Available: {list(self._df.columns)}"
            )
        self._pred_cols = columns
        cprint(f"Target columns set: {columns}", Colors.GREEN)
        self._detect_task_type()
        return self

    def method(self, method_name: Union[str, List[str]]) -> "AllML":
        """
        Select ML method(s).
        Pass 'all' to use every applicable algorithm.
        """
        all_methods = {
            **self.REGRESSION_METHODS,
            **self.CLASSIFICATION_METHODS,
            **self.CLUSTERING_METHODS,
        }

        if method_name == "all":
            if self._task_type is None:
                cprint("Call col_to_pred() before method().", Colors.YELLOW)
                return self
            if self._task_type in (TaskType.REGRESSION,
                                   TaskType.MULTI_OUTPUT_REGRESSION):
                self._method_names = list(self.REGRESSION_METHODS.keys())
            elif self._task_type in (TaskType.BINARY_CLASSIFICATION,
                                     TaskType.MULTICLASS_CLASSIFICATION,
                                     TaskType.MULTI_OUTPUT_CLASSIFICATION):
                self._method_names = list(self.CLASSIFICATION_METHODS.keys())
            elif self._task_type == TaskType.CLUSTERING:
                self._method_names = list(self.CLUSTERING_METHODS.keys())
            cprint(
                f"All {len(self._method_names)} methods selected "
                f"for task: {self._task_type}",
                Colors.GREEN,
            )
            return self

        if isinstance(method_name, str):
            method_name = [method_name]

        validated = []
        for m in method_name:
            if m not in all_methods:
                cprint(f"Unknown method: '{m}'", Colors.RED, bold=True)
                cprint(
                    f"  Regression    : {list(self.REGRESSION_METHODS.keys())}",
                    Colors.YELLOW,
                )
                cprint(
                    f"  Classification: {list(self.CLASSIFICATION_METHODS.keys())}",
                    Colors.YELLOW,
                )
                cprint(
                    f"  Clustering    : {list(self.CLUSTERING_METHODS.keys())}",
                    Colors.YELLOW,
                )
                raise ValueError(f"Unknown method: {m}")
            validated.append(m)

        self._method_names = validated
        cprint(f"Method(s) selected: {validated}", Colors.GREEN)
        return self

    def split(
        self,
        train_pct: float,
        test_pct: float,
        val_pct: float = 0.0,
        random_state: int = 42,
    ) -> "AllML":
        """
        Split dataset into train / test / (optional) validation sets.
        Percentages must sum to 100.
        Example: ml.split(70, 20, 10)
        """
        total = train_pct + test_pct + val_pct
        if not (99.0 <= total <= 101.0):
            raise ValueError(
                f"Percentages must sum to 100. Got: {total}"
            )
        if not self._feed_cols or not self._pred_cols:
            raise ValueError(
                "Call col_to_feed() and col_to_pred() before split()."
            )

        cprint(
            f"\nSplitting  ->  Train:{train_pct}%  "
            f"Test:{test_pct}%  Val:{val_pct}%",
            Colors.CYAN,
            bold=True,
        )

        X, y = self._prepare_data()
        self._X_raw, self._y_raw = X, y

        test_r  = test_pct  / 100.0
        val_r   = val_pct   / 100.0
        train_r = train_pct / 100.0

        if val_pct > 0:
            X_tmp, X_test, y_tmp, y_test = train_test_split(
                X, y, test_size=test_r, random_state=random_state
            )
            val_ratio = val_r / (train_r + val_r)
            X_train, X_val, y_train, y_val = train_test_split(
                X_tmp, y_tmp, test_size=val_ratio, random_state=random_state
            )
            self._X_val, self._y_val = X_val, y_val
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_r, random_state=random_state
            )

        self._X_train, self._y_train = X_train, y_train
        self._X_test,  self._y_test  = X_test,  y_test
        self._split_done = True

        info = (
            f"  Train : {len(X_train)} samples\n"
            f"  Test  : {len(X_test)}  samples"
        )
        if val_pct > 0:
            info += f"\n  Val   : {len(self._X_val)} samples"
        cprint(info, Colors.GREEN)
        print()
        return self

    def train(
        self,
        epochs: int = 100,
        learning_rate: float = 0.01,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        n_neighbors: int = 5,
        kernel: str = "rbf",
        n_clusters: int = 3,
        alpha: float = 1.0,
        **kwargs,
    ) -> "AllML":
        """
        Train selected model(s).

        Parameters
        ----------
        epochs        : iterations for iterative models
        learning_rate : step size for gradient-based models
        n_estimators  : number of trees for ensemble models
        max_depth     : maximum tree depth
        n_neighbors   : k for KNN
        kernel        : kernel type for SVM  ('rbf','linear','poly',...)
        n_clusters    : number of clusters for clustering models
        alpha         : regularisation strength
        """
        if not self._split_done:
            raise RuntimeError("Call ml.split() before ml.train().")
        if not self._method_names:
            raise RuntimeError("Call ml.method() before ml.train().")

        self._train_kwargs = dict(
            epochs=epochs,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_neighbors=n_neighbors,
            kernel=kernel,
            n_clusters=n_clusters,
            alpha=alpha,
            **kwargs,
        )

        cprint(
            f"\nStarting Training  |  task: {self._task_type}",
            Colors.MAGENTA,
            bold=True,
        )
        cprint(f"Methods: {self._method_names}", Colors.CYAN)
        print("-" * 65)

        self._models  = {}
        self._metrics = {}

        for mname in self._method_names:
            try:
                cprint(
                    f"\nTraining: {mname.upper().replace('_', ' ')}",
                    Colors.YELLOW,
                    bold=True,
                )

                model = self._build_model(
                    mname,
                    epochs,
                    learning_rate,
                    n_estimators,
                    max_depth,
                    n_neighbors,
                    kernel,
                    n_clusters,
                    alpha,
                    **kwargs,
                )
                if model is None:
                    continue

                # IsotonicRegression: 1-D input only
                if mname == "isotonic_regression":
                    if self._X_train.shape[1] != 1:
                        cprint(
                            f"  Skipping 'isotonic_regression': needs exactly "
                            f"1 feature column, you supplied "
                            f"{self._X_train.shape[1]}.\n"
                            f"  Suggestion: use col_to_feed() with a single "
                            f"column when using isotonic_regression.",
                            Colors.YELLOW,
                        )
                        continue
                    model.fit(
                        self._X_train[:, 0], self._y_train.ravel()
                    )

                else:
                    model = self._wrap_multioutput(mname, model)
                    y_fit = (
                        self._y_train.ravel()
                        if (
                            isinstance(self._y_train, np.ndarray)
                            and self._y_train.ndim == 2
                            and self._y_train.shape[1] == 1
                            and not isinstance(
                                model,
                                (MultiOutputRegressor, MultiOutputClassifier),
                            )
                        )
                        else self._y_train
                    )
                    model.fit(self._X_train, y_fit)

                metrics = self._evaluate(mname, model)
                self._models[mname]  = model
                self._metrics[mname] = metrics
                self._print_metrics(mname, metrics)

            except Exception as e:
                cprint(f"  Skipping '{mname}': {e}", Colors.RED)

        self._is_trained = True
        cprint(
            f"\nTraining complete. "
            f"{len(self._models)}/{len(self._method_names)} models trained.",
            Colors.GREEN,
            bold=True,
        )
        print()
        return self

    def save(self, path: str) -> "AllML":
        """
        Save trained model(s).
        Single model  -> <path>.allml
        Multiple      -> <path>_<method>.allml  (one file per model)
        """
        if not self._is_trained:
            raise RuntimeError("Call ml.train() before ml.save().")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "feed_cols"     : self._feed_cols,
            "pred_cols"     : self._pred_cols,
            "task_type"     : self._task_type,
            "method_names"  : list(self._models.keys()),
            "metrics"       : self._metrics,
            "train_kwargs"  : self._train_kwargs,
            "label_encoders": self._label_encoders,
        }

        if len(self._models) == 1:
            mname     = list(self._models.keys())[0]
            save_path = f"{path}.allml"
            with open(save_path, "wb") as f:
                pickle.dump(
                    {
                        "model"   : self._models[mname],
                        "scaler_X": self._scaler_X,
                        "scaler_y": self._scaler_y,
                        "meta"    : meta,
                    },
                    f,
                )
            cprint(f"Model saved -> {save_path}", Colors.GREEN, bold=True)
        else:
            for mname, model in self._models.items():
                save_path = f"{path}_{mname}.allml"
                with open(save_path, "wb") as f:
                    pickle.dump(
                        {
                            "model"        : model,
                            "scaler_X"     : self._scaler_X,
                            "scaler_y"     : self._scaler_y,
                            "meta"         : meta,
                            "active_method": mname,
                        },
                        f,
                    )
                cprint(f"Saved -> {save_path}", Colors.GREEN)

        # JSON summary (human-readable)
        meta_safe = {
            k: v
            for k, v in meta.items()
            if k not in ("label_encoders", "metrics")
        }
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta_safe, f, indent=2, default=str)
        cprint(f"Meta saved -> {path}_meta.json", Colors.CYAN)
        return self

    def load(self, model_path: str) -> "AllML":
        """Load a previously saved AllML model."""
        if not model_path.endswith(".allml"):
            model_path += ".allml"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        meta                 = bundle["meta"]
        self._feed_cols      = meta["feed_cols"]
        self._pred_cols      = meta["pred_cols"]
        self._task_type      = meta["task_type"]
        self._scaler_X       = bundle["scaler_X"]
        self._scaler_y       = bundle["scaler_y"]
        self._metrics        = meta.get("metrics", {})
        self._label_encoders = meta.get("label_encoders", {})
        self._train_kwargs   = meta.get("train_kwargs", {})

        active = bundle.get(
            "active_method",
            meta["method_names"][0] if meta["method_names"] else "model",
        )
        self._models       = {active: bundle["model"]}
        self._method_names = [active]
        self._is_trained   = True

        cprint(f"Model loaded: {model_path}", Colors.GREEN, bold=True)
        cprint(f"  Method   : {active}",          Colors.CYAN)
        cprint(f"  Task     : {self._task_type}",  Colors.CYAN)
        cprint(f"  Features : {self._feed_cols}",  Colors.CYAN)
        cprint(f"  Targets  : {self._pred_cols}",  Colors.CYAN)
        return self

    def pred(self, *args, method: Optional[str] = None):
        """
        Run prediction.
        Values must match col_to_feed() column order.

        Examples
        --------
        result = ml.pred(2000, 3, 2)
        result = ml.pred(2000, 3, 2, method="ridge")
        """
        if not self._is_trained:
            raise RuntimeError("Call ml.train() or ml.load() first.")
        if len(args) != len(self._feed_cols):
            raise ValueError(
                f"Expected {len(self._feed_cols)} values for "
                f"columns {self._feed_cols}, but got {len(args)}."
            )

        input_df = pd.DataFrame([list(args)], columns=self._feed_cols)
        for col in self._feed_cols:
            if col in self._label_encoders:
                input_df[col] = self._label_encoders[col].transform(
                    input_df[col].astype(str)
                )

        X_in = self._scaler_X.transform(input_df.values.astype(float))

        targets = (
            {method: self._models[method]} if method else self._models
        )

        results = {}
        for mname, model in targets.items():
            try:
                if mname == "isotonic_regression":
                    raw = model.predict(X_in[:, 0])
                else:
                    raw = model.predict(X_in)

                if self._task_type in (
                    TaskType.REGRESSION,
                    TaskType.MULTI_OUTPUT_REGRESSION,
                ):
                    try:
                        r   = raw.reshape(-1, 1) if raw.ndim == 1 else raw
                        raw = self._scaler_y.inverse_transform(r).flatten()
                    except Exception:
                        pass

                results[mname] = (
                    raw.tolist() if hasattr(raw, "tolist") else raw
                )
            except Exception as e:
                results[mname] = f"Error: {e}"

        cprint("\nPrediction result:", Colors.CYAN, bold=True)
        if len(results) == 1:
            val = list(results.values())[0]
            for i, pv in enumerate(
                val if hasattr(val, "__iter__") else [val]
            ):
                col = (
                    self._pred_cols[i]
                    if i < len(self._pred_cols)
                    else f"output_{i}"
                )
                cprint(f"  {col}: {pv}", Colors.GREEN)
            return val[0] if (isinstance(val, list) and len(val) == 1) else val

        for mname, val in results.items():
            cprint(f"  [{mname}] -> {val}", Colors.GREEN)
        return results

    def show(self) -> "AllML":
        """Print detailed information about the current model state."""
        print(f"""
{Colors.CYAN}{Colors.BOLD}
+--------------------------------------------------------------+
|                     MODEL DETAILS                            |
+--------------------------------------------------------------+
{Colors.END}""")
        cprint(f"  File      : {self._filepath}",       Colors.WHITE)
        cprint(f"  Shape     : {self._df.shape}",       Colors.WHITE)
        cprint(f"  Features  : {self._feed_cols}",      Colors.CYAN)
        cprint(f"  Targets   : {self._pred_cols}",      Colors.CYAN)
        cprint(f"  Task      : {self._task_type}",      Colors.YELLOW)
        cprint(f"  Methods   : {self._method_names}",   Colors.YELLOW)
        cprint(
            f"  Trained   : {self._is_trained}",
            Colors.GREEN if self._is_trained else Colors.RED,
        )

        if self._split_done and self._X_train is not None:
            cprint(f"\n  Data Splits:", Colors.MAGENTA, bold=True)
            cprint(f"    Train : {self._X_train.shape[0]} samples", Colors.WHITE)
            cprint(f"    Test  : {self._X_test.shape[0]}  samples", Colors.WHITE)
            if self._X_val is not None:
                cprint(
                    f"    Val   : {self._X_val.shape[0]}  samples", Colors.WHITE
                )

        if self._train_kwargs:
            cprint(f"\n  Train Config:", Colors.MAGENTA, bold=True)
            for k, v in self._train_kwargs.items():
                cprint(f"    {k:25s}: {v}", Colors.WHITE)

        if self._metrics:
            cprint(f"\n  Performance Metrics:", Colors.MAGENTA, bold=True)
            for mname, m in self._metrics.items():
                cprint(f"\n  [ {mname} ]", Colors.YELLOW, bold=True)
                for k, v in m.items():
                    if k in (
                        "y_pred", "y_true", "y_prob",
                        "fpr", "tpr", "conf_matrix", "labels",
                    ):
                        continue
                    line = (
                        f"    {k:30s}: {v:.6f}"
                        if isinstance(v, float)
                        else f"    {k:30s}: {v}"
                    )
                    cprint(line, Colors.WHITE)
                if "conf_matrix" in m:
                    cprint("    Confusion Matrix:", Colors.CYAN)
                    print(m["conf_matrix"])

        miss = self._df.isnull().sum()
        miss = miss[miss > 0]
        cprint(f"\n  Missing Values:", Colors.MAGENTA, bold=True)
        if len(miss):
            print(miss.to_string())
        else:
            cprint("    None found.", Colors.GREEN)
        print()
        return self

    def show_graph(
        self, graph_type: Union[str, List[str]] = "all"
    ) -> "AllML":
        """
        Plot graphs.

        Options
        -------
        confusion_matrix, roc_curve, precision_recall_curve,
        feature_importance, learning_curve, residuals,
        prediction_vs_actual, error_distribution,
        correlation_heatmap, pairplot, distribution,
        boxplot, cluster_plot, model_comparison, all

        Examples
        --------
        ml.show_graph("confusion_matrix")
        ml.show_graph(["roc_curve", "feature_importance"])
        ml.show_graph("all")
        """
        if isinstance(graph_type, str):
            graph_type = [graph_type]
        if "all" in graph_type:
            graph_type = [g for g in self.GRAPH_OPTIONS if g != "all"]

        plt.style.use("seaborn-v0_8-darkgrid")
        pal = sns.color_palette("husl", max(len(self._models), 1))

        for gt in graph_type:
            try:
                cprint(f"\nPlotting: {gt}", Colors.CYAN)
                self._plot(gt, pal)
            except Exception as e:
                cprint(f"  Could not plot '{gt}': {e}", Colors.YELLOW)
        return self

    def __repr__(self) -> str:
        return (
            f"AllML(file='{self._filepath}', "
            f"task='{self._task_type}', "
            f"methods={self._method_names}, "
            f"trained={self._is_trained})"
        )

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _detect_task_type(self):
        if not self._pred_cols:
            return
        targets  = self._df[self._pred_cols]
        n_tgt    = len(self._pred_cols)

        if n_tgt == 1:
            col      = self._pred_cols[0]
            n_unique = self._df[col].nunique()
            is_float = pd.api.types.is_float_dtype(self._df[col])
            is_num   = pd.api.types.is_numeric_dtype(self._df[col])

            if not is_num or (n_unique <= 20 and not is_float):
                self._task_type = (
                    TaskType.BINARY_CLASSIFICATION
                    if n_unique == 2
                    else TaskType.MULTICLASS_CLASSIFICATION
                )
            else:
                self._task_type = TaskType.REGRESSION
        else:
            all_float = all(
                pd.api.types.is_float_dtype(targets[c])
                for c in self._pred_cols
            )
            all_few = all(
                targets[c].nunique() <= 20 for c in self._pred_cols
            )
            self._task_type = (
                TaskType.MULTI_OUTPUT_REGRESSION
                if (all_float or not all_few)
                else TaskType.MULTI_OUTPUT_CLASSIFICATION
            )

        cprint(f"Task type detected: {self._task_type}", Colors.CYAN)

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        df_c = self._df[self._feed_cols + self._pred_cols].dropna().copy()

        for col in self._feed_cols:
            if not pd.api.types.is_numeric_dtype(df_c[col]):
                le = LabelEncoder()
                df_c[col] = le.fit_transform(df_c[col].astype(str))
                self._label_encoders[col] = le

        for col in self._pred_cols:
            if not pd.api.types.is_numeric_dtype(df_c[col]):
                le = LabelEncoder()
                df_c[col] = le.fit_transform(df_c[col].astype(str))
                self._label_encoders[col] = le

        X = df_c[self._feed_cols].values.astype(float)
        y = df_c[self._pred_cols].values
        if y.shape[1] == 1:
            y = y.ravel()

        X = self._scaler_X.fit_transform(X)

        if self._task_type in (
            TaskType.REGRESSION,
            TaskType.MULTI_OUTPUT_REGRESSION,
        ):
            y = (
                self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
                if y.ndim == 1
                else self._scaler_y.fit_transform(y)
            )

        return X, y

    def _build_model(
        self,
        mname,
        epochs,
        lr,
        n_estimators,
        max_depth,
        n_neighbors,
        kernel,
        n_clusters,
        alpha,
        **kwargs,
    ):
        """
        Instantiate a model with safe hyperparameter mapping.

        Fixed in v1.0.2
        ---------------
        * NuSVC / NuSVR  : nu auto-tuned to a feasible value
          based on minority-class fraction (classification)
          or 0.5 (regression).
        * GradientBoosting alpha in (0,1) enforced by skipping.
        * SGD learning_rate -> "constant" + eta0.
        * MLP learning_rate -> "constant" + learning_rate_init.
        * GaussianProcess kernel left unset (no string kernel).
        * IsotonicRegression caught at fit time.
        * PLSRegression n_components <= n_features enforced.
        """
        all_map = {
            **self.REGRESSION_METHODS,
            **self.CLASSIFICATION_METHODS,
            **self.CLUSTERING_METHODS,
        }
        task   = self._task_type
        is_reg  = mname in self.REGRESSION_METHODS
        is_cls  = mname in self.CLASSIFICATION_METHODS
        is_clus = mname in self.CLUSTERING_METHODS

        # ---- task / method compatibility ---------------------
        if task in (
            TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION
        ) and not is_reg:
            cprint(
                f"  '{mname}' is not a regression method "
                f"(task={task}).\n"
                f"  Suggestion: {list(self.REGRESSION_METHODS.keys())[:4]}",
                Colors.YELLOW,
            )
            return None

        if task in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTI_OUTPUT_CLASSIFICATION,
        ) and not is_cls:
            cprint(
                f"  '{mname}' is not a classification method "
                f"(task={task}).\n"
                f"  Suggestion: "
                f"{list(self.CLASSIFICATION_METHODS.keys())[:4]}",
                Colors.YELLOW,
            )
            return None

        if task == TaskType.CLUSTERING and not is_clus:
            cprint(
                f"  '{mname}' is not a clustering method.\n"
                f"  Suggestion: {list(self.CLUSTERING_METHODS.keys())}",
                Colors.YELLOW,
            )
            return None

        ModelCls    = all_map[mname]
        sig         = inspect.signature(ModelCls.__init__).parameters
        init_params : Dict[str, Any] = {}

        # ---- random_state ------------------------------------
        if "random_state" in sig:
            init_params["random_state"] = 42

        # ---- verbose off -------------------------------------
        if "verbose" in sig:
            init_params["verbose"] = 0

        # ---- max_iter / n_iter  (epoch-like) -----------------
        if "max_iter" in sig:
            init_params["max_iter"] = int(epochs)
        if "n_iter" in sig:
            init_params["n_iter"] = int(epochs)

        # ---- learning_rate  (model-specific) -----------------
        if mname in _FLOAT_LR:
            if "learning_rate" in sig:
                init_params["learning_rate"] = float(lr)

        elif mname in _SGD_MODELS:
            if "learning_rate" in sig:
                init_params["learning_rate"] = "constant"
            if "eta0" in sig:
                init_params["eta0"] = float(lr)

        elif mname in _MLP_MODELS:
            if "learning_rate" in sig:
                init_params["learning_rate"] = "constant"
            if "learning_rate_init" in sig:
                init_params["learning_rate_init"] = float(lr)

        else:
            if "learning_rate" in sig:
                init_params["learning_rate"] = float(lr)
            if "learning_rate_init" in sig:
                init_params["learning_rate_init"] = float(lr)
            if "eta0" in sig:
                init_params["eta0"] = float(lr)

        # ---- n_estimators ------------------------------------
        if "n_estimators" in sig:
            init_params["n_estimators"] = int(n_estimators)

        # ---- max_depth ---------------------------------------
        if "max_depth" in sig and max_depth is not None:
            init_params["max_depth"] = int(max_depth)

        # ---- n_neighbors ------------------------------------
        if "n_neighbors" in sig:
            init_params["n_neighbors"] = int(n_neighbors)

        # ---- kernel  (SVM string, not GaussianProcess) ------
        if mname in _SVM_MODELS and "kernel" in sig:
            init_params["kernel"] = kernel

        # ---- n_clusters -------------------------------------
        if "n_clusters" in sig:
            init_params["n_clusters"] = int(n_clusters)

        # ---- alpha ------------------------------------------
        if mname not in _SKIP_ALPHA and "alpha" in sig:
            init_params["alpha"] = float(alpha)

        # ---- SVC probability (for predict_proba / ROC) ------
        if mname in ("svc", "nu_svc"):
            init_params["probability"] = True

        # ---- NuSVC / NuSVR  ->  safe nu  --------------------
        #
        # NuSVC raises "specified nu is infeasible" when nu >
        # fraction of the smallest class.  We calculate the
        # maximum valid nu from the training labels and pick
        # a safe value below that threshold.
        #
        if mname in _NU_MODELS and "nu" in sig:
            if self._y_train is not None:
                safe_nu = _find_safe_nu(self._y_train, task)
            else:
                safe_nu = 0.1          # conservative fallback
            init_params["nu"] = safe_nu
            cprint(
                f"  NuSVM  ->  auto-selected nu = {safe_nu:.2f} "
                f"(feasible for this dataset)",
                Colors.CYAN,
            )

        # ---- PLSRegression: n_components <= n_features ------
        if mname == "pls_regression":
            n_feat = len(self._feed_cols)
            init_params["n_components"] = max(1, min(2, n_feat))
            init_params.pop("random_state", None)
            init_params.pop("verbose",      None)

        # ---- forward extra user kwargs ----------------------
        for k, v in kwargs.items():
            if k in sig and k not in init_params:
                init_params[k] = v

        return ModelCls(**init_params)

    def _wrap_multioutput(self, mname, model):
        task = self._task_type
        _no_wrap_reg = {
            "isotonic_regression",
            "gaussian_process_regressor",
            "pls_regression",
            "lars",
            "lasso_lars",
        }
        if task == TaskType.MULTI_OUTPUT_REGRESSION:
            if mname not in _no_wrap_reg:
                try:
                    return MultiOutputRegressor(model)
                except Exception:
                    pass
        elif task == TaskType.MULTI_OUTPUT_CLASSIFICATION:
            try:
                return MultiOutputClassifier(model)
            except Exception:
                pass
        return model

    def _evaluate(self, mname, model) -> Dict:
        metrics = {}
        task    = self._task_type

        try:
            X_te = self._X_test
            y_te = self._y_test

            # IsotonicRegression: 1-D input
            if mname == "isotonic_regression":
                if X_te.shape[1] != 1:
                    metrics["error"] = (
                        "IsotonicRegression requires exactly 1 feature."
                    )
                    return metrics
                y_pred = model.predict(X_te[:, 0])
            else:
                y_pred = model.predict(X_te)

            # -- Regression ------------------------------------
            if task in (
                TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION
            ):
                try:
                    yp_inv = self._scaler_y.inverse_transform(
                        y_pred.reshape(-1, 1)
                        if y_pred.ndim == 1 else y_pred
                    ).flatten()
                    yt_inv = self._scaler_y.inverse_transform(
                        y_te.reshape(-1, 1)
                        if y_te.ndim == 1 else y_te
                    ).flatten()
                except Exception:
                    yp_inv = y_pred.ravel()
                    yt_inv = y_te.ravel()

                metrics["MSE"]    = float(mean_squared_error(yt_inv, yp_inv))
                metrics["RMSE"]   = float(np.sqrt(metrics["MSE"]))
                metrics["MAE"]    = float(mean_absolute_error(yt_inv, yp_inv))
                try:
                    metrics["R2"] = float(r2_score(yt_inv, yp_inv))
                except Exception:
                    pass
                metrics["y_pred"] = yp_inv
                metrics["y_true"] = yt_inv

            # -- Classification --------------------------------
            elif task in (
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTICLASS_CLASSIFICATION,
            ):
                metrics["accuracy"]    = float(accuracy_score(y_te, y_pred))
                metrics["f1_weighted"] = float(
                    f1_score(y_te, y_pred, average="weighted", zero_division=0)
                )
                metrics["precision"]   = float(
                    precision_score(
                        y_te, y_pred, average="weighted", zero_division=0
                    )
                )
                metrics["recall"]      = float(
                    recall_score(
                        y_te, y_pred, average="weighted", zero_division=0
                    )
                )
                metrics["conf_matrix"] = confusion_matrix(y_te, y_pred)
                metrics["y_pred"]      = y_pred
                metrics["y_true"]      = y_te

                if task == TaskType.BINARY_CLASSIFICATION:
                    try:
                        if hasattr(model, "predict_proba"):
                            y_prob = model.predict_proba(X_te)[:, 1]
                        elif hasattr(model, "decision_function"):
                            y_prob = model.decision_function(X_te)
                        else:
                            y_prob = y_pred
                        metrics["roc_auc"] = float(
                            roc_auc_score(y_te, y_prob)
                        )
                        metrics["y_prob"]  = y_prob
                        fpr, tpr, _        = roc_curve(y_te, y_prob)
                        metrics["fpr"]     = fpr
                        metrics["tpr"]     = tpr
                    except Exception:
                        pass

            # -- Multi-output ----------------------------------
            elif task in (
                TaskType.MULTI_OUTPUT_REGRESSION,
                TaskType.MULTI_OUTPUT_CLASSIFICATION,
            ):
                metrics["y_pred"] = y_pred
                metrics["y_true"] = y_te

            # -- Clustering ------------------------------------
            elif task == TaskType.CLUSTERING:
                metrics["labels"] = y_pred
                try:
                    metrics["silhouette"] = float(
                        silhouette_score(X_te, y_pred)
                    )
                except Exception:
                    pass
                try:
                    metrics["davies_bouldin"] = float(
                        davies_bouldin_score(X_te, y_pred)
                    )
                except Exception:
                    pass

        except Exception as e:
            metrics["error"] = str(e)

        return metrics

    def _print_metrics(self, mname, metrics):
        print(f"  {'-'*55}")
        for k, v in metrics.items():
            if k in (
                "y_pred", "y_true", "y_prob",
                "fpr", "tpr", "conf_matrix", "labels",
            ):
                continue
            line = (
                f"  {k:30s}: {v:.6f}"
                if isinstance(v, float)
                else f"  {k:30s}: {v}"
            )
            cprint(line, Colors.WHITE)
        if "conf_matrix" in metrics:
            cprint("  Confusion Matrix:", Colors.CYAN)
            print(metrics["conf_matrix"])

    # ----------------------------------------------------------
    # Graph helpers
    # ----------------------------------------------------------

    def _plot(self, graph_type, palette):
        dispatch = {
            "correlation_heatmap"  : self._plot_correlation_heatmap,
            "pairplot"             : self._plot_pairplot,
            "distribution"         : self._plot_distribution,
            "boxplot"              : self._plot_boxplot,
            "confusion_matrix"     : self._plot_confusion_matrix,
            "roc_curve"            : self._plot_roc_curve,
            "precision_recall_curve": self._plot_precision_recall,
            "feature_importance"   : lambda: self._plot_feature_importance(palette),
            "learning_curve"       : self._plot_learning_curve,
            "residuals"            : lambda: self._plot_residuals(palette),
            "prediction_vs_actual" : lambda: self._plot_pred_vs_actual(palette),
            "error_distribution"   : lambda: self._plot_error_distribution(palette),
            "cluster_plot"         : lambda: self._plot_cluster(palette),
            "model_comparison"     : self._plot_model_comparison,
        }
        fn = dispatch.get(graph_type)
        if fn:
            fn()
        else:
            cprint(
                f"  Unknown graph: '{graph_type}'. "
                f"Options: {self.GRAPH_OPTIONS}",
                Colors.YELLOW,
            )

    def _plot_correlation_heatmap(self):
        cols     = self._feed_cols + self._pred_cols
        num_cols = [
            c for c in cols
            if pd.api.types.is_numeric_dtype(self._df[c])
        ]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            self._df[num_cols].corr(),
            annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5,
            ax=ax, square=True,
        )
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_pairplot(self):
        cols = self._feed_cols + self._pred_cols
        g    = sns.pairplot(
            self._df[cols], diag_kind="kde",
            plot_kws={"alpha": 0.5},
        )
        g.fig.suptitle("Pair Plot", y=1.02, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_distribution(self):
        cols  = self._feed_cols + self._pred_cols
        ncols = 3
        nrows = (len(cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        pal  = sns.color_palette("husl", len(cols))
        for i, col in enumerate(cols):
            sns.histplot(
                self._df[col].dropna(), kde=True,
                ax=axes[i], color=pal[i],
            )
            axes[i].set_title(f"Distribution: {col}")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(
            "Feature Distributions", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    def _plot_boxplot(self):
        cols     = self._feed_cols + self._pred_cols
        num_cols = [
            c for c in cols
            if pd.api.types.is_numeric_dtype(self._df[c])
        ]
        ncols = 3
        nrows = (len(num_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        pal  = sns.color_palette("husl", len(num_cols))
        for i, col in enumerate(num_cols):
            sns.boxplot(
                y=self._df[col].dropna(), ax=axes[i], color=pal[i]
            )
            axes[i].set_title(f"Boxplot: {col}")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Box Plots", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_confusion_matrix(self):
        task = self._task_type
        if task not in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ):
            cprint(
                "  Confusion matrix is only for classification tasks.",
                Colors.YELLOW,
            )
            return
        cms = {
            k: v for k, v in self._metrics.items()
            if "conf_matrix" in v
        }
        if not cms:
            return
        n    = len(cms)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
        if n == 1:
            axes = [axes]
        for ax, (mname, m) in zip(axes, cms.items()):
            sns.heatmap(
                m["conf_matrix"], annot=True, fmt="d",
                cmap="Blues", ax=ax, linewidths=0.5,
            )
            ax.set_title(f"Confusion Matrix\n{mname}", fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_roc_curve(self):
        if self._task_type != TaskType.BINARY_CLASSIFICATION:
            cprint(
                "  ROC Curve is only for binary classification.",
                Colors.YELLOW,
            )
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        pal     = sns.color_palette("husl", len(self._metrics))
        for i, (mname, m) in enumerate(self._metrics.items()):
            if "fpr" in m and "tpr" in m:
                auc_val = m.get("roc_auc", auc(m["fpr"], m["tpr"]))
                ax.plot(
                    m["fpr"], m["tpr"],
                    label=f"{mname} (AUC={auc_val:.4f})",
                    color=pal[i],
                )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _plot_precision_recall(self):
        if self._task_type != TaskType.BINARY_CLASSIFICATION:
            cprint(
                "  Precision-Recall Curve is only for binary classification.",
                Colors.YELLOW,
            )
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        pal     = sns.color_palette("husl", len(self._metrics))
        for i, (mname, m) in enumerate(self._metrics.items()):
            if "y_prob" in m and "y_true" in m:
                prec, rec, _ = precision_recall_curve(
                    m["y_true"], m["y_prob"]
                )
                ax.plot(rec, prec, label=mname, color=pal[i])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _plot_feature_importance(self, palette):
        found = False
        for mname, model in self._models.items():
            inner = (
                model.estimators_[0]
                if isinstance(
                    model, (MultiOutputRegressor, MultiOutputClassifier)
                )
                else model
            )
            if hasattr(inner, "feature_importances_"):
                imp  = inner.feature_importances_
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.barh(
                    self._feed_cols, imp,
                    color=sns.color_palette("husl", len(self._feed_cols)),
                )
                ax.set_xlabel("Importance Score")
                ax.set_title(
                    f"Feature Importance: {mname}", fontweight="bold"
                )
                ax.invert_yaxis()
                for bar, v in zip(bars, imp):
                    ax.text(
                        bar.get_width() + 0.001,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.4f}", va="center", fontsize=9,
                    )
                plt.tight_layout()
                plt.show()
                found = True

            elif hasattr(inner, "coef_"):
                coef = np.abs(
                    inner.coef_.ravel()[: len(self._feed_cols)]
                )
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(
                    self._feed_cols[: len(coef)], coef,
                    color=sns.color_palette("husl", len(self._feed_cols)),
                )
                ax.set_xlabel("|Coefficient|")
                ax.set_title(
                    f"Feature Coefficients: {mname}", fontweight="bold"
                )
                plt.tight_layout()
                plt.show()
                found = True

        if not found:
            cprint(
                "  No trained models support feature importance.",
                Colors.YELLOW,
            )

    def _plot_learning_curve(self):
        for mname, model in self._models.items():
            try:
                X_all = np.vstack([self._X_train, self._X_test])
                y_all = np.concatenate(
                    [
                        p.ravel()
                        if (
                            isinstance(p, np.ndarray)
                            and p.ndim == 2
                            and p.shape[1] == 1
                        )
                        else (p.ravel() if p.ndim == 1 else p)
                        for p in [self._y_train, self._y_test]
                    ]
                )
                scoring = (
                    "r2"
                    if self._task_type
                    in (
                        TaskType.REGRESSION,
                        TaskType.MULTI_OUTPUT_REGRESSION,
                    )
                    else "accuracy"
                )
                train_sz, tr_sc, te_sc = learning_curve(
                    model, X_all, y_all,
                    cv=3, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 8),
                    scoring=scoring,
                )
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.fill_between(
                    train_sz,
                    tr_sc.mean(1) - tr_sc.std(1),
                    tr_sc.mean(1) + tr_sc.std(1),
                    alpha=0.1, color="blue",
                )
                ax.fill_between(
                    train_sz,
                    te_sc.mean(1) - te_sc.std(1),
                    te_sc.mean(1) + te_sc.std(1),
                    alpha=0.1, color="orange",
                )
                ax.plot(
                    train_sz, tr_sc.mean(1), "o-",
                    color="blue", label="Train",
                )
                ax.plot(
                    train_sz, te_sc.mean(1), "o-",
                    color="orange", label="CV",
                )
                ax.set_xlabel("Training Samples")
                ax.set_ylabel(scoring)
                ax.set_title(
                    f"Learning Curve: {mname}", fontweight="bold"
                )
                ax.legend()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                cprint(
                    f"  Learning curve failed for '{mname}': {e}",
                    Colors.YELLOW,
                )

    def _plot_residuals(self, palette):
        if self._task_type not in (
            TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION
        ):
            cprint(
                "  Residuals plot is only for regression tasks.",
                Colors.YELLOW,
            )
            return
        n    = len(self._metrics)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (mname, m) in zip(axes, self._metrics.items()):
            if "y_pred" in m and "y_true" in m:
                res = (
                    np.array(m["y_true"]).ravel()
                    - np.array(m["y_pred"]).ravel()
                )
                ax.scatter(m["y_pred"], res, alpha=0.5, color=palette[0])
                ax.axhline(0, color="red", linestyle="--")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Residuals")
                ax.set_title(f"Residuals: {mname}", fontweight="bold")
        plt.suptitle("Residuals Plots", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_pred_vs_actual(self, palette):
        if self._task_type not in (
            TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION
        ):
            cprint(
                "  Prediction vs Actual is only for regression tasks.",
                Colors.YELLOW,
            )
            return
        n    = len(self._metrics)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (mname, m) in zip(axes, self._metrics.items()):
            if "y_pred" in m and "y_true" in m:
                yt = np.array(m["y_true"]).ravel()
                yp = np.array(m["y_pred"]).ravel()
                ax.scatter(yt, yp, alpha=0.5, color=palette[0])
                lo = min(yt.min(), yp.min())
                hi = max(yt.max(), yp.max())
                ax.plot([lo, hi], [lo, hi], "r--", label="Perfect fit")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(
                    f"Pred vs Actual: {mname}", fontweight="bold"
                )
                ax.legend()
        plt.suptitle(
            "Prediction vs Actual", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    def _plot_error_distribution(self, palette):
        if self._task_type not in (
            TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION
        ):
            cprint(
                "  Error distribution is only for regression tasks.",
                Colors.YELLOW,
            )
            return
        n    = len(self._metrics)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (mname, m) in zip(axes, self._metrics.items()):
            if "y_pred" in m and "y_true" in m:
                err = (
                    np.array(m["y_true"]).ravel()
                    - np.array(m["y_pred"]).ravel()
                )
                sns.histplot(err, kde=True, ax=ax, color=palette[0])
                ax.axvline(0, color="red", linestyle="--")
                ax.set_title(
                    f"Error Distribution: {mname}", fontweight="bold"
                )
                ax.set_xlabel("Prediction Error")
        plt.suptitle(
            "Error Distributions", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    def _plot_cluster(self, palette):
        if self._task_type != TaskType.CLUSTERING:
            cprint(
                "  Cluster plot is only for clustering tasks.",
                Colors.YELLOW,
            )
            return
        for mname, m in self._metrics.items():
            if "labels" not in m:
                continue
            fig, ax = plt.subplots(figsize=(9, 6))
            X2  = self._X_test[:, :2]
            sc  = ax.scatter(
                X2[:, 0], X2[:, 1],
                c=m["labels"], cmap="tab10", alpha=0.6,
            )
            plt.colorbar(sc, ax=ax, label="Cluster")
            ax.set_title(f"Cluster Plot: {mname}", fontweight="bold")
            plt.tight_layout()
            plt.show()

    def _plot_model_comparison(self):
        if not self._metrics:
            cprint("  No metrics available for comparison.", Colors.YELLOW)
            return

        task = self._task_type
        if task in (
            TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION
        ):
            key, ylabel = "R2", "R2 Score"
        elif task in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTI_OUTPUT_CLASSIFICATION,
        ):
            key, ylabel = "accuracy", "Accuracy"
        elif task == TaskType.CLUSTERING:
            key, ylabel = "silhouette", "Silhouette Score"
        else:
            return

        names, vals = [], []
        for mname, m in self._metrics.items():
            if key in m:
                names.append(mname.replace("_", "\n"))
                vals.append(m[key])

        if not names:
            cprint(f"  No '{key}' metric found.", Colors.YELLOW)
            return

        pal  = sns.color_palette("husl", len(names))
        fig, ax = plt.subplots(
            figsize=(max(10, len(names) * 1.5), 6)
        )
        bars = ax.bar(
            names, vals, color=pal, edgecolor="white", linewidth=0.8
        )
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Model Comparison: {ylabel}", fontweight="bold"
        )
        ax.set_ylim(0, max(vals) * 1.15 if vals else 1.0)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.4f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        plt.show()
