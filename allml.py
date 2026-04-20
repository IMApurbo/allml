# allml.py
# AllML - The Swiss Army Knife of Machine Learning
# Author: AllML Library
# Version: 1.0.0

import os
import sys
import warnings
import pickle
import json
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, MinMaxScaler, 
    PolynomialFeatures, LabelBinarizer
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, f1_score,
    precision_score, recall_score, roc_auc_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, SGDRegressor, SGDClassifier,
    PassiveAggressiveRegressor, PassiveAggressiveClassifier,
    Perceptron, HuberRegressor, RANSACRegressor,
    TheilSenRegressor, ARDRegression, OrthogonalMatchingPursuit,
    Lars, LassoLars, MultiTaskLasso, MultiTaskElasticNet,
    RidgeClassifier
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    BaggingRegressor, BaggingClassifier,
    VotingRegressor, VotingClassifier,
    StackingRegressor, StackingClassifier,
    IsolationForest
)
from sklearn.tree import (
    DecisionTreeRegressor, DecisionTreeClassifier,
    ExtraTreeRegressor, ExtraTreeClassifier
)
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC, NuSVR, NuSVC
from sklearn.neighbors import (
    KNeighborsRegressor, KNeighborsClassifier,
    RadiusNeighborsRegressor, RadiusNeighborsClassifier
)
from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, MeanShift,
    SpectralClustering, Birch, MiniBatchKMeans
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

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
# ANSI Colors for beautiful terminal output
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


def cprint(text: str, color: str = Colors.WHITE, bold: bool = False):
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.END}")


def banner():
    print(f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    ░█████╗░██╗░░░░░██╗░░░░░███╗░░░███╗██╗░░                 ║
║    ██╔══██╗██║░░░░░██║░░░░░████╗░████║██║░░                 ║
║    ███████║██║░░░░░██║░░░░░██╔████╔██║██║░░                 ║
║    ██╔══██║██║░░░░░██║░░░░░██║╚██╔╝██║██║░░                 ║
║    ██║░░██║███████╗███████╗██║░╚═╝░██║███████╗              ║
║    ╚═╝░░╚═╝╚══════╝╚══════╝╚═╝░░░░╚═╝╚══════╝              ║
║                                                              ║
║     🤖 All-in-One Machine Learning Library v1.0.0 🤖        ║
║          "One Import to Rule Them All"                       ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}""")


# ============================================================
# Task Type Detection
# ============================================================
class TaskType:
    REGRESSION          = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTI_OUTPUT_REGRESSION = "multi_output_regression"
    MULTI_OUTPUT_CLASSIFICATION = "multi_output_classification"
    CLUSTERING          = "clustering"


# ============================================================
# AllML Core Class
# ============================================================
class AllML:
    """
    AllML - The All-in-One Machine Learning Library
    ================================================
    Usage:
        from allml import AllML
        ml = AllML("data.csv")
        ml.col_to_feed(["feature1", "feature2"])
        ml.col_to_pred(["target"])
        ml.method("random_forest")
        ml.split(70, 20, 10)
        ml.train(epochs=100, learning_rate=0.01)
        ml.show()
        ml.show_graph("confusion_matrix")
        ml.save("models/my_model")
        pred = ml.pred(5.1, 3.2)
        ml.load("models/my_model")
    """

    # ------------------------------------------------------------------ #
    #  Available Methods Registry
    # ------------------------------------------------------------------ #
    REGRESSION_METHODS = {
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

    CLASSIFICATION_METHODS = {
        "logistic_regression"             : LogisticRegression,
        "ridge_classifier"                : RidgeClassifier,
        "sgd_classifier"                  : SGDClassifier,
        "passive_aggressive_classifier"   : PassiveAggressiveClassifier,
        "perceptron"                      : Perceptron,
        "decision_tree_classifier"        : DecisionTreeClassifier,
        "extra_tree_classifier"           : ExtraTreeClassifier,
        "random_forest_classifier"        : RandomForestClassifier,
        "extra_trees_classifier"          : ExtraTreesClassifier,
        "gradient_boosting_classifier"    : GradientBoostingClassifier,
        "adaboost_classifier"             : AdaBoostClassifier,
        "bagging_classifier"              : BaggingClassifier,
        "svc"                             : SVC,
        "linear_svc"                      : LinearSVC,
        "nu_svc"                          : NuSVC,
        "knn_classifier"                  : KNeighborsClassifier,
        "gaussian_nb"                     : GaussianNB,
        "bernoulli_nb"                    : BernoulliNB,
        "mlp_classifier"                  : MLPClassifier,
        "gaussian_process_classifier"     : GaussianProcessClassifier,
        "lda"                             : LinearDiscriminantAnalysis,
        "qda"                             : QuadraticDiscriminantAnalysis,
    }

    BOOSTING_METHODS = {}

    if XGBOOST_AVAILABLE:
        REGRESSION_METHODS["xgboost_regressor"]        = XGBRegressor
        CLASSIFICATION_METHODS["xgboost_classifier"]   = XGBClassifier

    if LIGHTGBM_AVAILABLE:
        REGRESSION_METHODS["lightgbm_regressor"]       = LGBMRegressor
        CLASSIFICATION_METHODS["lightgbm_classifier"]  = LGBMClassifier

    if CATBOOST_AVAILABLE:
        REGRESSION_METHODS["catboost_regressor"]       = CatBoostRegressor
        CLASSIFICATION_METHODS["catboost_classifier"]  = CatBoostClassifier

    CLUSTERING_METHODS = {
        "kmeans"                  : KMeans,
        "dbscan"                  : DBSCAN,
        "agglomerative_clustering": AgglomerativeClustering,
        "mean_shift"              : MeanShift,
        "spectral_clustering"     : SpectralClustering,
        "birch"                   : Birch,
        "mini_batch_kmeans"       : MiniBatchKMeans,
    }

    # Graph options
    GRAPH_OPTIONS = [
        "confusion_matrix", "roc_curve", "precision_recall_curve",
        "feature_importance", "learning_curve", "residuals",
        "prediction_vs_actual", "error_distribution", "correlation_heatmap",
        "pairplot", "distribution", "boxplot", "cluster_plot",
        "model_comparison", "all"
    ]

    # ------------------------------------------------------------------ #
    #  Constructor
    # ------------------------------------------------------------------ #
    def __init__(self, csv_filepath: str):
        banner()
        cprint(f"📂 Loading dataset: {csv_filepath}", Colors.CYAN, bold=True)

        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"❌ File not found: {csv_filepath}")

        self._filepath      = csv_filepath
        self._df            = pd.read_csv(csv_filepath)
        self._feed_cols     : List[str] = []
        self._pred_cols     : List[str] = []
        self._method_names  : List[str] = []
        self._task_type     : Optional[str] = None
        self._models        : Dict[str, Any] = {}
        self._metrics       : Dict[str, Dict] = {}
        self._scaler_X      = StandardScaler()
        self._scaler_y      = StandardScaler()
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._is_trained    = False
        self._split_done    = False
        self._train_kwargs  : Dict = {}

        # Data splits
        self._X_train = self._X_test = self._X_val = None
        self._y_train = self._y_test = self._y_val = None
        self._X_raw   = self._y_raw = None

        # Metadata for save/load
        self._meta: Dict[str, Any] = {}

        cprint(f"✅ Dataset loaded successfully!", Colors.GREEN, bold=True)
        cprint(f"   Rows: {self._df.shape[0]} | Columns: {self._df.shape[1]}", Colors.YELLOW)
        cprint(f"   Columns: {list(self._df.columns)}", Colors.YELLOW)
        print()

    # ------------------------------------------------------------------ #
    #  col_to_feed
    # ------------------------------------------------------------------ #
    def col_to_feed(self, columns: Union[str, List[str]]) -> "AllML":
        """Specify feature columns (input to the model)."""
        if isinstance(columns, str):
            columns = [columns]
        missing = [c for c in columns if c not in self._df.columns]
        if missing:
            raise ValueError(
                f"❌ Columns not found in dataset: {missing}\n"
                f"   Available columns: {list(self._df.columns)}"
            )
        self._feed_cols = columns
        cprint(f"✅ Feature columns set: {columns}", Colors.GREEN)
        return self

    # ------------------------------------------------------------------ #
    #  col_to_pred
    # ------------------------------------------------------------------ #
    def col_to_pred(self, columns: Union[str, List[str]]) -> "AllML":
        """Specify target/prediction columns."""
        if isinstance(columns, str):
            columns = [columns]
        missing = [c for c in columns if c not in self._df.columns]
        if missing:
            raise ValueError(
                f"❌ Columns not found in dataset: {missing}\n"
                f"   Available columns: {list(self._df.columns)}"
            )
        self._pred_cols = columns
        cprint(f"✅ Target columns set: {columns}", Colors.GREEN)
        self._detect_task_type()
        return self

    # ------------------------------------------------------------------ #
    #  method
    # ------------------------------------------------------------------ #
    def method(self, method_name: Union[str, List[str]]) -> "AllML":
        """
        Select ML method(s).
        Use 'all' to select all applicable methods.
        """
        all_methods = {**self.REGRESSION_METHODS,
                       **self.CLASSIFICATION_METHODS,
                       **self.CLUSTERING_METHODS}

        if method_name == "all":
            if self._task_type is None:
                cprint("⚠️  Please call col_to_pred() before method().", Colors.YELLOW)
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
            cprint(f"✅ All {len(self._method_names)} methods selected for task: {self._task_type}", Colors.GREEN)
            return self

        if isinstance(method_name, str):
            method_name = [method_name]

        validated = []
        for m in method_name:
            if m not in all_methods:
                cprint(f"❌ Unknown method: '{m}'", Colors.RED, bold=True)
                cprint(f"   Regression methods    : {list(self.REGRESSION_METHODS.keys())}", Colors.YELLOW)
                cprint(f"   Classification methods: {list(self.CLASSIFICATION_METHODS.keys())}", Colors.YELLOW)
                cprint(f"   Clustering methods    : {list(self.CLUSTERING_METHODS.keys())}", Colors.YELLOW)
                raise ValueError(f"Unknown method: {m}")
            validated.append(m)

        self._method_names = validated
        cprint(f"✅ Method(s) selected: {validated}", Colors.GREEN)
        return self

    # ------------------------------------------------------------------ #
    #  split
    # ------------------------------------------------------------------ #
    def split(self, train_pct: float, test_pct: float,
              val_pct: float = 0.0, random_state: int = 42) -> "AllML":
        """
        Split dataset.
        Percentages should sum to 100.
        Example: ml.split(70, 20, 10)
        """
        total = train_pct + test_pct + val_pct
        if not (99.0 <= total <= 101.0):
            raise ValueError(
                f"❌ Percentages must sum to 100. Got: {total}"
            )

        if not self._feed_cols or not self._pred_cols:
            raise ValueError("❌ Please call col_to_feed() and col_to_pred() first.")

        cprint(f"\n🔀 Splitting dataset → Train:{train_pct}% | Test:{test_pct}% | Val:{val_pct}%", Colors.CYAN, bold=True)

        # Prepare data
        X, y = self._prepare_data()
        self._X_raw = X
        self._y_raw = y

        train_r = train_pct / 100.0
        test_r  = test_pct  / 100.0
        val_r   = val_pct   / 100.0

        if val_pct > 0:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_r, random_state=random_state
            )
            val_ratio = val_r / (train_r + val_r)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state
            )
            self._X_val = X_val
            self._y_val = y_val
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_r, random_state=random_state
            )

        self._X_train = X_train
        self._X_test  = X_test
        self._y_train = y_train
        self._y_test  = y_test
        self._split_done = True

        cprint(f"   Train : {len(X_train)} samples", Colors.GREEN)
        cprint(f"   Test  : {len(X_test)} samples", Colors.GREEN)
        if val_pct > 0:
            cprint(f"   Val   : {len(X_val)} samples", Colors.GREEN)
        print()
        return self

    # ------------------------------------------------------------------ #
    #  train
    # ------------------------------------------------------------------ #
    def train(self, epochs: int = 100, learning_rate: float = 0.01,
              n_estimators: int = 100, max_depth: Optional[int] = None,
              n_neighbors: int = 5, kernel: str = "rbf",
              n_clusters: int = 3, alpha: float = 1.0,
              **kwargs) -> "AllML":
        """
        Train selected model(s).

        Parameters
        ----------
        epochs        : Number of iterations/epochs (for iterative models)
        learning_rate : Learning rate (for gradient-based models)
        n_estimators  : Number of trees (for ensemble methods)
        max_depth     : Max tree depth
        n_neighbors   : K for KNN
        kernel        : Kernel for SVM
        n_clusters    : Number of clusters (for clustering)
        alpha         : Regularization strength
        """
        if not self._split_done:
            raise RuntimeError("❌ Please call ml.split() before ml.train().")
        if not self._method_names:
            raise RuntimeError("❌ Please call ml.method() before ml.train().")

        self._train_kwargs = dict(
            epochs=epochs, learning_rate=learning_rate,
            n_estimators=n_estimators, max_depth=max_depth,
            n_neighbors=n_neighbors, kernel=kernel,
            n_clusters=n_clusters, alpha=alpha, **kwargs
        )

        cprint(f"\n🚀 Starting Training...", Colors.MAGENTA, bold=True)
        cprint(f"   Task Type : {self._task_type}", Colors.CYAN)
        cprint(f"   Methods   : {self._method_names}", Colors.CYAN)
        print("─" * 65)

        self._models  = {}
        self._metrics = {}

        for mname in self._method_names:
            try:
                cprint(f"\n⚙️  Training: {mname.upper().replace('_',' ')}", Colors.YELLOW, bold=True)
                model = self._build_model(
                    mname, epochs, learning_rate,
                    n_estimators, max_depth, n_neighbors,
                    kernel, n_clusters, alpha, **kwargs
                )
                if model is None:
                    continue

                # Wrap in multi-output if needed
                model = self._wrap_multioutput(mname, model)

                # Fit
                model.fit(self._X_train, self._y_train.ravel()
                          if (isinstance(self._y_train, np.ndarray) and
                              self._y_train.ndim == 2 and
                              self._y_train.shape[1] == 1 and
                              not isinstance(model, (MultiOutputRegressor, MultiOutputClassifier)))
                          else self._y_train)

                # Evaluate
                metrics = self._evaluate(mname, model)
                self._models[mname]  = model
                self._metrics[mname] = metrics

                self._print_metrics(mname, metrics)

            except Exception as e:
                cprint(f"   ⚠️  Skipping '{mname}': {e}", Colors.RED)

        self._is_trained = True
        cprint(f"\n✅ Training complete! {len(self._models)}/{len(self._method_names)} models trained.", Colors.GREEN, bold=True)
        print()
        return self

    # ------------------------------------------------------------------ #
    #  save
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> "AllML":
        """
        Save trained model(s) to disk.
        If multiple models, saves each with method name suffix.
        """
        if not self._is_trained:
            raise RuntimeError("❌ No trained model to save. Call ml.train() first.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "feed_cols"   : self._feed_cols,
            "pred_cols"   : self._pred_cols,
            "task_type"   : self._task_type,
            "method_names": list(self._models.keys()),
            "metrics"     : self._metrics,
            "train_kwargs": self._train_kwargs,
            "label_encoders": {k: v for k, v in self._label_encoders.items()},
        }

        if len(self._models) == 1:
            mname  = list(self._models.keys())[0]
            bundle = {
                "model"   : self._models[mname],
                "scaler_X": self._scaler_X,
                "scaler_y": self._scaler_y,
                "meta"    : meta,
            }
            save_path = f"{path}.allml"
            with open(save_path, "wb") as f:
                pickle.dump(bundle, f)
            cprint(f"💾 Model saved → {save_path}", Colors.GREEN, bold=True)
        else:
            for mname, model in self._models.items():
                bundle = {
                    "model"   : model,
                    "scaler_X": self._scaler_X,
                    "scaler_y": self._scaler_y,
                    "meta"    : meta,
                    "active_method": mname,
                }
                save_path = f"{path}_{mname}.allml"
                with open(save_path, "wb") as f:
                    pickle.dump(bundle, f)
                cprint(f"💾 Saved → {save_path}", Colors.GREEN)

        # Save meta summary as JSON too
        meta_json = {k: v for k, v in meta.items()
                     if k not in ("label_encoders",)}
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta_json, f, indent=2, default=str)
        cprint(f"📋 Meta saved → {path}_meta.json", Colors.CYAN)
        return self

    # ------------------------------------------------------------------ #
    #  load
    # ------------------------------------------------------------------ #
    def load(self, model_path: str) -> "AllML":
        """Load a saved AllML model."""
        if not model_path.endswith(".allml"):
            model_path += ".allml"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        meta              = bundle["meta"]
        self._feed_cols   = meta["feed_cols"]
        self._pred_cols   = meta["pred_cols"]
        self._task_type   = meta["task_type"]
        self._scaler_X    = bundle["scaler_X"]
        self._scaler_y    = bundle["scaler_y"]
        self._metrics     = meta.get("metrics", {})
        self._label_encoders = meta.get("label_encoders", {})
        self._train_kwargs   = meta.get("train_kwargs", {})

        active_method = bundle.get("active_method",
                                   meta["method_names"][0] if meta["method_names"] else "model")
        self._models  = {active_method: bundle["model"]}
        self._method_names = [active_method]
        self._is_trained   = True

        cprint(f"✅ Model loaded from: {model_path}", Colors.GREEN, bold=True)
        cprint(f"   Method    : {active_method}", Colors.CYAN)
        cprint(f"   Task Type : {self._task_type}", Colors.CYAN)
        cprint(f"   Features  : {self._feed_cols}", Colors.CYAN)
        cprint(f"   Targets   : {self._pred_cols}", Colors.CYAN)
        return self

    # ------------------------------------------------------------------ #
    #  pred
    # ------------------------------------------------------------------ #
    def pred(self, *args, method: Optional[str] = None) -> Union[np.ndarray, Dict]:
        """
        Predict using trained model.

        Usage:
            prediction = ml.pred(5.1, 3.2)
            prediction = ml.pred(5.1, 3.2, method="random_forest_classifier")

        Args are input values corresponding to col_to_feed() columns.
        """
        if not self._is_trained:
            raise RuntimeError("❌ No trained model. Call ml.train() or ml.load() first.")

        if len(args) != len(self._feed_cols):
            raise ValueError(
                f"❌ Expected {len(self._feed_cols)} input values for columns {self._feed_cols}, "
                f"but got {len(args)}."
            )

        # Build input DataFrame
        input_df = pd.DataFrame([list(args)], columns=self._feed_cols)

        # Encode categoricals
        for col in self._feed_cols:
            if col in self._label_encoders:
                le = self._label_encoders[col]
                input_df[col] = le.transform(input_df[col].astype(str))

        X_input = input_df.values.astype(float)
        X_scaled = self._scaler_X.transform(X_input)

        # Select model(s)
        if method:
            if method not in self._models:
                raise ValueError(
                    f"❌ Method '{method}' not found in trained models: {list(self._models.keys())}"
                )
            targets = {method: self._models[method]}
        else:
            targets = self._models

        results = {}
        for mname, model in targets.items():
            try:
                raw_pred = model.predict(X_scaled)

                # Inverse transform for regression
                if self._task_type in (TaskType.REGRESSION,
                                       TaskType.MULTI_OUTPUT_REGRESSION):
                    if hasattr(self._scaler_y, "inverse_transform"):
                        try:
                            if raw_pred.ndim == 1:
                                raw_pred = raw_pred.reshape(-1, 1)
                            raw_pred = self._scaler_y.inverse_transform(raw_pred).flatten()
                        except Exception:
                            pass

                results[mname] = raw_pred.tolist() if hasattr(raw_pred, "tolist") else raw_pred

            except Exception as e:
                results[mname] = f"Error: {e}"

        if len(results) == 1:
            val = list(results.values())[0]
            cprint(f"\n🎯 Prediction Result:", Colors.CYAN, bold=True)
            for i, pval in enumerate(val if hasattr(val, "__iter__") else [val]):
                col = self._pred_cols[i] if i < len(self._pred_cols) else f"output_{i}"
                cprint(f"   {col}: {pval}", Colors.GREEN)
            return val[0] if (isinstance(val, list) and len(val) == 1) else val

        cprint(f"\n🎯 Predictions from all models:", Colors.CYAN, bold=True)
        for mname, val in results.items():
            cprint(f"   [{mname}] → {val}", Colors.GREEN)
        return results

    # ------------------------------------------------------------------ #
    #  show
    # ------------------------------------------------------------------ #
    def show(self) -> "AllML":
        """Show detailed info about the model(s) and metrics."""
        print(f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                    📊 MODEL DETAILS                          ║
╚══════════════════════════════════════════════════════════════╝{Colors.END}""")

        cprint(f"  Dataset     : {self._filepath}", Colors.WHITE)
        cprint(f"  Rows        : {self._df.shape[0]}", Colors.WHITE)
        cprint(f"  Columns     : {self._df.shape[1]}", Colors.WHITE)
        cprint(f"  Features    : {self._feed_cols}", Colors.CYAN)
        cprint(f"  Targets     : {self._pred_cols}", Colors.CYAN)
        cprint(f"  Task Type   : {self._task_type}", Colors.YELLOW)
        cprint(f"  Methods     : {self._method_names}", Colors.YELLOW)
        cprint(f"  Trained     : {self._is_trained}", Colors.GREEN if self._is_trained else Colors.RED)

        if self._split_done and self._X_train is not None:
            cprint(f"\n  Data Splits:", Colors.MAGENTA, bold=True)
            cprint(f"    Train : {self._X_train.shape[0]} samples", Colors.WHITE)
            cprint(f"    Test  : {self._X_test.shape[0]} samples", Colors.WHITE)
            if self._X_val is not None:
                cprint(f"    Val   : {self._X_val.shape[0]} samples", Colors.WHITE)

        if self._train_kwargs:
            cprint(f"\n  Train Config:", Colors.MAGENTA, bold=True)
            for k, v in self._train_kwargs.items():
                cprint(f"    {k:25s}: {v}", Colors.WHITE)

        if self._metrics:
            cprint(f"\n  Performance Metrics:", Colors.MAGENTA, bold=True)
            print("  " + "─" * 62)
            for mname, m in self._metrics.items():
                cprint(f"\n  📌 {mname.upper().replace('_', ' ')}", Colors.YELLOW, bold=True)
                for metric_name, val in m.items():
                    if isinstance(val, float):
                        cprint(f"     {metric_name:30s}: {val:.6f}", Colors.WHITE)
                    elif isinstance(val, np.ndarray):
                        cprint(f"     {metric_name:30s}: [array, shape={val.shape}]", Colors.WHITE)
                    else:
                        cprint(f"     {metric_name:30s}: {val}", Colors.WHITE)

        # Dataset info
        cprint(f"\n  Dataset Stats:", Colors.MAGENTA, bold=True)
        print(self._df[self._feed_cols + self._pred_cols].describe().to_string())

        # Missing values
        cprint(f"\n  Missing Values:", Colors.MAGENTA, bold=True)
        miss = self._df.isnull().sum()
        miss = miss[miss > 0]
        if len(miss):
            print(miss.to_string())
        else:
            cprint("    No missing values found. ✅", Colors.GREEN)

        print()
        return self

    # ------------------------------------------------------------------ #
    #  show_graph
    # ------------------------------------------------------------------ #
    def show_graph(self, graph_type: Union[str, List[str]] = "all") -> "AllML":
        """
        Plot graphs.

        Options:
            confusion_matrix, roc_curve, precision_recall_curve,
            feature_importance, learning_curve, residuals,
            prediction_vs_actual, error_distribution, correlation_heatmap,
            pairplot, distribution, boxplot, cluster_plot,
            model_comparison, all

        Usage:
            ml.show_graph("confusion_matrix")
            ml.show_graph(["roc_curve", "feature_importance"])
            ml.show_graph("all")
        """
        if isinstance(graph_type, str):
            graph_type = [graph_type]

        if "all" in graph_type:
            graph_type = [g for g in self.GRAPH_OPTIONS if g != "all"]

        plt.style.use("seaborn-v0_8-darkgrid")
        palette = sns.color_palette("husl", len(self._models) if self._models else 1)

        for gt in graph_type:
            try:
                cprint(f"\n📈 Plotting: {gt}", Colors.CYAN)
                self._plot(gt, palette)
            except Exception as e:
                cprint(f"⚠️  Could not plot '{gt}': {e}", Colors.YELLOW)

        return self

    # ================================================================== #
    #  INTERNAL HELPERS
    # ================================================================== #

    # ------------------------------------------------------------------ #
    def _detect_task_type(self):
        """Auto-detect task type from target columns."""
        if not self._pred_cols:
            return

        n_targets = len(self._pred_cols)

        # Check if target columns exist
        targets = self._df[self._pred_cols]

        unique_per_col = [targets[c].nunique() for c in self._pred_cols]
        dtype_per_col  = [targets[c].dtype for c in self._pred_cols]

        # Infer
        is_numeric     = all(pd.api.types.is_numeric_dtype(targets[c]) for c in self._pred_cols)
        max_unique     = max(unique_per_col)

        if n_targets == 1:
            col = self._pred_cols[0]
            unique_vals = self._df[col].nunique()
            is_float = pd.api.types.is_float_dtype(self._df[col])

            if not is_numeric or unique_vals <= 20 and not is_float:
                if unique_vals == 2:
                    self._task_type = TaskType.BINARY_CLASSIFICATION
                else:
                    self._task_type = TaskType.MULTICLASS_CLASSIFICATION
            else:
                self._task_type = TaskType.REGRESSION
        else:
            # Multiple targets
            all_numeric = all(pd.api.types.is_numeric_dtype(targets[c]) for c in self._pred_cols)
            all_few_unique = all(targets[c].nunique() <= 20 for c in self._pred_cols)
            all_float = all(pd.api.types.is_float_dtype(targets[c]) for c in self._pred_cols)

            if all_numeric and (all_float or not all_few_unique):
                self._task_type = TaskType.MULTI_OUTPUT_REGRESSION
            else:
                self._task_type = TaskType.MULTI_OUTPUT_CLASSIFICATION

        cprint(f"🔍 Task type detected: {self._task_type}", Colors.CYAN)

    # ------------------------------------------------------------------ #
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and scale X, y arrays."""
        df_clean = self._df[self._feed_cols + self._pred_cols].dropna().copy()

        # Encode categorical features
        for col in self._feed_cols:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                self._label_encoders[col] = le

        # Encode categorical targets
        for col in self._pred_cols:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                self._label_encoders[col] = le

        X = df_clean[self._feed_cols].values.astype(float)
        y = df_clean[self._pred_cols].values

        if y.shape[1] == 1:
            y = y.ravel()

        # Scale X
        X = self._scaler_X.fit_transform(X)

        # Scale y only for regression
        if self._task_type in (TaskType.REGRESSION,
                               TaskType.MULTI_OUTPUT_REGRESSION):
            if y.ndim == 1:
                y = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            else:
                y = self._scaler_y.fit_transform(y)

        return X, y

    # ------------------------------------------------------------------ #
    def _build_model(self, mname, epochs, lr, n_estimators,
                     max_depth, n_neighbors, kernel,
                     n_clusters, alpha, **kwargs):
        """Instantiate a model with appropriate hyperparameters."""
        all_reg  = self.REGRESSION_METHODS
        all_cls  = self.CLASSIFICATION_METHODS
        all_clus = self.CLUSTERING_METHODS

        # ---- Validate task/method compatibility ----
        task = self._task_type
        is_reg  = mname in all_reg
        is_cls  = mname in all_cls
        is_clus = mname in all_clus

        if task in (TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION) and not is_reg:
            cprint(f"   ⚠️  '{mname}' is a classification/clustering method. "
                   f"Your task is {task}.", Colors.YELLOW)
            cprint(f"      💡 Suggested regression methods: {list(all_reg.keys())[:5]}...", Colors.CYAN)
            return None

        if task in (TaskType.BINARY_CLASSIFICATION,
                    TaskType.MULTICLASS_CLASSIFICATION,
                    TaskType.MULTI_OUTPUT_CLASSIFICATION) and not is_cls:
            cprint(f"   ⚠️  '{mname}' is a regression/clustering method. "
                   f"Your task is {task}.", Colors.YELLOW)
            cprint(f"      💡 Suggested classification methods: {list(all_cls.keys())[:5]}...", Colors.CYAN)
            return None

        if task == TaskType.CLUSTERING and not is_clus:
            cprint(f"   ⚠️  '{mname}' is not a clustering method.", Colors.YELLOW)
            cprint(f"      💡 Suggested clustering methods: {list(all_clus.keys())}", Colors.CYAN)
            return None

        # ---- Map hyperparameters ----
        cls_map = {**all_reg, **all_cls, **all_clus}
        ModelCls = cls_map[mname]

        init_params = {}
        import inspect
        sig = inspect.signature(ModelCls.__init__).parameters

        # Epoch-like params
        if "max_iter" in sig:
            init_params["max_iter"] = epochs
        if "n_iter" in sig:
            init_params["n_iter"] = epochs

        # Learning rate
        if "learning_rate" in sig:
            init_params["learning_rate"] = lr
        if "learning_rate_init" in sig:
            init_params["learning_rate_init"] = lr
        if "eta0" in sig:
            init_params["eta0"] = lr

        # Estimators
        if "n_estimators" in sig:
            init_params["n_estimators"] = n_estimators

        # Max depth
        if "max_depth" in sig and max_depth is not None:
            init_params["max_depth"] = max_depth

        # N neighbors
        if "n_neighbors" in sig:
            init_params["n_neighbors"] = n_neighbors

        # Kernel
        if "kernel" in sig:
            init_params["kernel"] = kernel

        # N clusters
        if "n_clusters" in sig:
            init_params["n_clusters"] = n_clusters

        # Alpha
        if "alpha" in sig:
            init_params["alpha"] = alpha

        # Probability for SVC (needed for ROC)
        if mname in ("svc", "nu_svc"):
            init_params["probability"] = True

        # Verbose off
        if "verbose" in sig:
            init_params["verbose"] = 0

        # Random state
        if "random_state" in sig:
            init_params["random_state"] = 42

        # Extra kwargs
        for k, v in kwargs.items():
            if k in sig:
                init_params[k] = v

        return ModelCls(**init_params)

    # ------------------------------------------------------------------ #
    def _wrap_multioutput(self, mname, model):
        """Wrap model in MultiOutput wrapper if needed."""
        task = self._task_type

        if task == TaskType.MULTI_OUTPUT_REGRESSION:
            multi_unsupported = [
                "isotonic_regression", "gaussian_process_regressor",
                "pls_regression", "lars", "lasso_lars"
            ]
            if mname not in multi_unsupported:
                if not hasattr(model, "_is_multitask"):
                    try:
                        return MultiOutputRegressor(model)
                    except Exception:
                        return model
            return model

        if task == TaskType.MULTI_OUTPUT_CLASSIFICATION:
            try:
                return MultiOutputClassifier(model)
            except Exception:
                return model

        return model

    # ------------------------------------------------------------------ #
    def _evaluate(self, mname, model) -> Dict:
        """Evaluate model and return metrics dictionary."""
        metrics = {}
        task    = self._task_type
        y_test  = self._y_test

        try:
            X_test_data = self._X_test
            y_pred_raw  = model.predict(X_test_data)

            # Inverse scale regression predictions
            if task in (TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION):
                y_pred = y_pred_raw
                y_true = y_test
                try:
                    if y_pred.ndim == 1:
                        y_pred_inv = self._scaler_y.inverse_transform(
                            y_pred.reshape(-1, 1)).ravel()
                        y_true_inv = self._scaler_y.inverse_transform(
                            y_true.reshape(-1, 1)).ravel()
                    else:
                        y_pred_inv = self._scaler_y.inverse_transform(y_pred)
                        y_true_inv = self._scaler_y.inverse_transform(y_true)
                    y_pred = y_pred_inv
                    y_true = y_true_inv
                except Exception:
                    y_true = y_test

                metrics["MSE"]  = mean_squared_error(y_true, y_pred)
                metrics["RMSE"] = np.sqrt(metrics["MSE"])
                metrics["MAE"]  = mean_absolute_error(y_true, y_pred)
                try:
                    metrics["R2"]   = r2_score(y_true, y_pred)
                except Exception:
                    pass
                metrics["y_pred"] = y_pred
                metrics["y_true"] = y_true

            elif task in (TaskType.BINARY_CLASSIFICATION,
                          TaskType.MULTICLASS_CLASSIFICATION):
                metrics["accuracy"]  = accuracy_score(y_test, y_pred_raw)
                metrics["f1_weighted"] = f1_score(y_test, y_pred_raw, average="weighted", zero_division=0)
                metrics["precision"] = precision_score(y_test, y_pred_raw, average="weighted", zero_division=0)
                metrics["recall"]    = recall_score(y_test, y_pred_raw, average="weighted", zero_division=0)
                metrics["conf_matrix"] = confusion_matrix(y_test, y_pred_raw)
                metrics["y_pred"]    = y_pred_raw
                metrics["y_true"]    = y_test

                if task == TaskType.BINARY_CLASSIFICATION:
                    try:
                        if hasattr(model, "predict_proba"):
                            y_prob = model.predict_proba(X_test_data)[:, 1]
                        elif hasattr(model, "decision_function"):
                            y_prob = model.decision_function(X_test_data)
                        else:
                            y_prob = y_pred_raw
                        metrics["roc_auc"]    = roc_auc_score(y_test, y_prob)
                        metrics["y_prob"]     = y_prob
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        metrics["fpr"] = fpr
                        metrics["tpr"] = tpr
                    except Exception:
                        pass

            elif task in (TaskType.MULTI_OUTPUT_REGRESSION,
                          TaskType.MULTI_OUTPUT_CLASSIFICATION):
                metrics["y_pred"] = y_pred_raw
                metrics["y_true"] = y_test

            elif task == TaskType.CLUSTERING:
                metrics["labels"] = y_pred_raw
                try:
                    metrics["silhouette"] = silhouette_score(X_test_data, y_pred_raw)
                except Exception:
                    pass
                try:
                    metrics["davies_bouldin"] = davies_bouldin_score(X_test_data, y_pred_raw)
                except Exception:
                    pass

        except Exception as e:
            metrics["error"] = str(e)

        return metrics

    # ------------------------------------------------------------------ #
    def _print_metrics(self, mname, metrics):
        """Pretty-print metrics after training."""
        task = self._task_type
        print(f"   {'─'*55}")
        for k, v in metrics.items():
            if k in ("y_pred", "y_true", "y_prob", "fpr", "tpr",
                     "conf_matrix", "labels"):
                continue
            if isinstance(v, float):
                cprint(f"   {k:30s}: {v:.6f}", Colors.WHITE)
            else:
                cprint(f"   {k:30s}: {v}", Colors.WHITE)

        if "conf_matrix" in metrics:
            cprint(f"   Confusion Matrix:", Colors.CYAN)
            print(metrics["conf_matrix"])

    # ------------------------------------------------------------------ #
    def _plot(self, graph_type: str, palette):
        """Internal plot dispatcher."""
        task = self._task_type

        if graph_type == "correlation_heatmap":
            self._plot_correlation_heatmap()

        elif graph_type == "pairplot":
            self._plot_pairplot()

        elif graph_type == "distribution":
            self._plot_distribution()

        elif graph_type == "boxplot":
            self._plot_boxplot()

        elif graph_type == "confusion_matrix":
            self._plot_confusion_matrix()

        elif graph_type == "roc_curve":
            self._plot_roc_curve()

        elif graph_type == "precision_recall_curve":
            self._plot_precision_recall()

        elif graph_type == "feature_importance":
            self._plot_feature_importance(palette)

        elif graph_type == "learning_curve":
            self._plot_learning_curve()

        elif graph_type == "residuals":
            self._plot_residuals(palette)

        elif graph_type == "prediction_vs_actual":
            self._plot_pred_vs_actual(palette)

        elif graph_type == "error_distribution":
            self._plot_error_distribution(palette)

        elif graph_type == "cluster_plot":
            self._plot_cluster(palette)

        elif graph_type == "model_comparison":
            self._plot_model_comparison()

        else:
            cprint(f"⚠️  Unknown graph type: {graph_type}. Options: {self.GRAPH_OPTIONS}", Colors.YELLOW)

    # ............ Individual Plot Methods ............ #

    def _plot_correlation_heatmap(self):
        cols = self._feed_cols + self._pred_cols
        fig, ax = plt.subplots(figsize=(12, 8))
        corr = self._df[cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, square=True)
        ax.set_title("🔥 Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_pairplot(self):
        cols = self._feed_cols + self._pred_cols
        g = sns.pairplot(self._df[cols], diag_kind="kde",
                         plot_kws={"alpha": 0.5})
        g.fig.suptitle("Pair Plot", y=1.02, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_distribution(self):
        cols  = self._feed_cols + self._pred_cols
        ncols = 3
        nrows = (len(cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for i, col in enumerate(cols):
            sns.histplot(self._df[col].dropna(), kde=True, ax=axes[i],
                         color=sns.color_palette("husl", len(cols))[i])
            axes[i].set_title(f"Distribution: {col}")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("📊 Feature Distributions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_boxplot(self):
        cols  = self._feed_cols + self._pred_cols
        ncols = 3
        nrows = (len(cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for i, col in enumerate(cols):
            sns.boxplot(y=self._df[col].dropna(), ax=axes[i],
                        color=sns.color_palette("husl", len(cols))[i])
            axes[i].set_title(f"Boxplot: {col}")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("📦 Box Plots", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_confusion_matrix(self):
        task = self._task_type
        if task not in (TaskType.BINARY_CLASSIFICATION,
                        TaskType.MULTICLASS_CLASSIFICATION):
            cprint("⚠️  Confusion matrix only for classification tasks.", Colors.YELLOW)
            return

        models_with_cm = {k: v for k, v in self._metrics.items()
                          if "conf_matrix" in v}
        if not models_with_cm:
            return

        n = len(models_with_cm)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
        if n == 1:
            axes = [axes]

        for ax, (mname, m) in zip(axes, models_with_cm.items()):
            cm = m["conf_matrix"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        linewidths=0.5)
            ax.set_title(f"Confusion Matrix\n{mname}", fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        plt.suptitle("🧩 Confusion Matrices", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_roc_curve(self):
        task = self._task_type
        if task != TaskType.BINARY_CLASSIFICATION:
            cprint("⚠️  ROC Curve only for binary classification.", Colors.YELLOW)
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (mname, m) in enumerate(self._metrics.items()):
            if "fpr" in m and "tpr" in m:
                roc_auc_val = m.get("roc_auc", auc(m["fpr"], m["tpr"]))
                ax.plot(m["fpr"], m["tpr"],
                        label=f"{mname} (AUC={roc_auc_val:.4f})",
                        color=sns.color_palette("husl", len(self._metrics))[i])

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("📉 ROC Curves", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    def _plot_precision_recall(self):
        task = self._task_type
        if task != TaskType.BINARY_CLASSIFICATION:
            cprint("⚠️  Precision-Recall Curve only for binary classification.", Colors.YELLOW)
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (mname, m) in enumerate(self._metrics.items()):
            if "y_prob" in m and "y_true" in m:
                prec, rec, _ = precision_recall_curve(m["y_true"], m["y_prob"])
                ax.plot(rec, prec,
                        label=mname,
                        color=sns.color_palette("husl", len(self._metrics))[i])

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("📐 Precision-Recall Curves", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _plot_feature_importance(self, palette):
        fig_count = 0
        for mname, model in self._models.items():
            actual = model.estimators_[0] if isinstance(
                model, (MultiOutputRegressor, MultiOutputClassifier)) else model
            if hasattr(actual, "feature_importances_"):
                importances = actual.feature_importances_
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.barh(self._feed_cols, importances,
                               color=sns.color_palette("husl", len(self._feed_cols)))
                ax.set_xlabel("Importance Score")
                ax.set_title(f"🌲 Feature Importance: {mname}", fontweight="bold")
                ax.invert_yaxis()
                for bar, imp in zip(bars, importances):
                    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                            f"{imp:.4f}", va="center", ha="left", fontsize=9)
                plt.tight_layout()
                plt.show()
                fig_count += 1
            elif hasattr(actual, "coef_"):
                coef = actual.coef_.ravel()[:len(self._feed_cols)]
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(self._feed_cols[:len(coef)], np.abs(coef),
                        color=sns.color_palette("husl", len(self._feed_cols)))
                ax.set_xlabel("|Coefficient|")
                ax.set_title(f"📊 Feature Coefficients: {mname}", fontweight="bold")
                plt.tight_layout()
                plt.show()
                fig_count += 1

        if fig_count == 0:
            cprint("⚠️  No models support feature importance/coefficients.", Colors.YELLOW)

    def _plot_learning_curve(self):
        for mname, model in self._models.items():
            try:
                task = self._task_type
                X_full = np.vstack([self._X_train, self._X_test])
                y_full_parts = [self._y_train, self._y_test]

                def _safe_ravel(arr):
                    return arr.ravel() if (isinstance(arr, np.ndarray) and arr.ndim == 2
                                           and arr.shape[1] == 1) else arr

                y_full = np.concatenate([_safe_ravel(yp) for yp in y_full_parts])

                scoring = "r2" if task in (TaskType.REGRESSION,
                                            TaskType.MULTI_OUTPUT_REGRESSION) else "accuracy"

                train_sizes, train_scores, test_scores = learning_curve(
                    model, X_full, y_full,
                    cv=3, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 8),
                    scoring=scoring
                )

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.fill_between(train_sizes,
                                train_scores.mean(1) - train_scores.std(1),
                                train_scores.mean(1) + train_scores.std(1), alpha=0.1, color="blue")
                ax.fill_between(train_sizes,
                                test_scores.mean(1) - test_scores.std(1),
                                test_scores.mean(1) + test_scores.std(1), alpha=0.1, color="orange")
                ax.plot(train_sizes, train_scores.mean(1), "o-", color="blue", label="Train Score")
                ax.plot(train_sizes, test_scores.mean(1), "o-", color="orange", label="CV Score")
                ax.set_xlabel("Training Examples")
                ax.set_ylabel(scoring)
                ax.set_title(f"📈 Learning Curve: {mname}", fontweight="bold")
                ax.legend()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                cprint(f"⚠️  Learning curve failed for {mname}: {e}", Colors.YELLOW)

    def _plot_residuals(self, palette):
        task = self._task_type
        if task not in (TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION):
            cprint("⚠️  Residuals plot only for regression tasks.", Colors.YELLOW)
            return

        n = len(self._metrics)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (mname, m) in zip(axes, self._metrics.items()):
            if "y_pred" in m and "y_true" in m:
                res = np.array(m["y_true"]).ravel() - np.array(m["y_pred"]).ravel()
                ax.scatter(m["y_pred"], res, alpha=0.5, color=palette[0])
                ax.axhline(0, color="red", linestyle="--")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Residuals")
                ax.set_title(f"Residuals: {mname}", fontweight="bold")

        plt.suptitle("📉 Residuals Plots", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_pred_vs_actual(self, palette):
        task = self._task_type
        if task not in (TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION):
            cprint("⚠️  Prediction vs Actual only for regression tasks.", Colors.YELLOW)
            return

        n = len(self._metrics)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (mname, m) in zip(axes, self._metrics.items()):
            if "y_pred" in m and "y_true" in m:
                yt = np.array(m["y_true"]).ravel()
                yp = np.array(m["y_pred"]).ravel()
                ax.scatter(yt, yp, alpha=0.5, color=palette[0])
                mn_val = min(yt.min(), yp.min())
                mx_val = max(yt.max(), yp.max())
                ax.plot([mn_val, mx_val], [mn_val, mx_val], "r--", label="Perfect fit")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"Pred vs Actual: {mname}", fontweight="bold")
                ax.legend()

        plt.suptitle("🎯 Prediction vs Actual", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_error_distribution(self, palette):
        task = self._task_type
        if task not in (TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION):
            cprint("⚠️  Error distribution only for regression tasks.", Colors.YELLOW)
            return

        n = len(self._metrics)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (mname, m) in zip(axes, self._metrics.items()):
            if "y_pred" in m and "y_true" in m:
                errors = np.array(m["y_true"]).ravel() - np.array(m["y_pred"]).ravel()
                sns.histplot(errors, kde=True, ax=ax, color=palette[0])
                ax.axvline(0, color="red", linestyle="--")
                ax.set_title(f"Error Distribution: {mname}", fontweight="bold")
                ax.set_xlabel("Prediction Error")

        plt.suptitle("📊 Error Distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _plot_cluster(self, palette):
        if self._task_type != TaskType.CLUSTERING:
            cprint("⚠️  Cluster plot only for clustering tasks.", Colors.YELLOW)
            return

        if self._X_test is None:
            cprint("⚠️  No test data available.", Colors.YELLOW)
            return

        for mname, m in self._metrics.items():
            if "labels" not in m:
                continue
            fig, ax = plt.subplots(figsize=(9, 6))
            X_2d = self._X_test[:, :2]
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                                 c=m["labels"], cmap="tab10", alpha=0.6)
            plt.colorbar(scatter, ax=ax, label="Cluster")
            ax.set_xlabel(self._feed_cols[0] if self._feed_cols else "Feature 1")
            ax.set_ylabel(self._feed_cols[1] if len(self._feed_cols) > 1 else "Feature 2")
            ax.set_title(f"🔵 Cluster Plot: {mname}", fontweight="bold")
            plt.tight_layout()
            plt.show()

    def _plot_model_comparison(self):
        """Bar chart comparing all models."""
        if not self._metrics:
            cprint("⚠️  No metrics available for comparison.", Colors.YELLOW)
            return

        task = self._task_type
        if task in (TaskType.REGRESSION, TaskType.MULTI_OUTPUT_REGRESSION):
            metric_key = "R2"
            ylabel = "R² Score"
        elif task in (TaskType.BINARY_CLASSIFICATION,
                      TaskType.MULTICLASS_CLASSIFICATION,
                      TaskType.MULTI_OUTPUT_CLASSIFICATION):
            metric_key = "accuracy"
            ylabel = "Accuracy"
        elif task == TaskType.CLUSTERING:
            metric_key = "silhouette"
            ylabel = "Silhouette Score"
        else:
            return

        names  = []
        values = []
        for mname, m in self._metrics.items():
            if metric_key in m:
                names.append(mname.replace("_", "\n"))
                values.append(m[metric_key])

        if not names:
            cprint(f"⚠️  No '{metric_key}' metric found for comparison.", Colors.YELLOW)
            return

        colors = sns.color_palette("husl", len(names))
        fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
        bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"🏆 Model Comparison: {ylabel}", fontweight="bold")
        ax.set_ylim(0, max(values) * 1.15 if values else 1.0)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    def __repr__(self):
        return (
            f"AllML(file='{self._filepath}', "
            f"task='{self._task_type}', "
            f"methods={self._method_names}, "
            f"trained={self._is_trained})"
        )
