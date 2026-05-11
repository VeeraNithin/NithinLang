# src/nithinlang/ml_ds_engine.py
"""
NithinLang ML / Data Science Engine
=====================================
Provides high-level, beginner-friendly wrappers around:
  - NumPy   (array operations)
  - Pandas  (DataFrames, CSV, filtering, grouping)
  - Scikit-learn (classification, regression, clustering, evaluation)

All functions are designed to be injected into the NithinLang global
execution namespace and called directly from .nl files.

Function glossary
-----------------
data_chudu(path)            → DataFrame from CSV / JSON / Excel
data_save(df, path)         → Save DataFrame to CSV
data_describe(df)           → Statistical summary
data_filter(df, col, op, v) → Filter rows
data_sort(df, col, asc)     → Sort DataFrame
data_group(df, col, agg)    → GroupBy + aggregation
data_plot(df, kind, x, y)   → Plot via matplotlib

model_train(X, y, kind)     → Train a sklearn estimator
model_test(model, X, y)     → Evaluate accuracy / R²
model_predict(model, X)     → Predict labels / values
model_save(model, path)     → Pickle model to disk
model_load(path)            → Unpickle model from disk

cluster_fit(X, k)           → KMeans clustering

np_array, np_zeros, np_ones, np_linspace,
np_mean,  np_std,  np_sum, np_max, np_min
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Union

# ── NumPy ─────────────────────────────────────────────────────────────────
try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False
    np = None  # type: ignore[assignment]

# ── Pandas ────────────────────────────────────────────────────────────────
try:
    import pandas as pd
    _PD = True
except ImportError:
    _PD = False
    pd = None  # type: ignore[assignment]

# ── Scikit-learn ──────────────────────────────────────────────────────────
try:
    from sklearn.linear_model       import LogisticRegression, LinearRegression
    from sklearn.ensemble           import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.svm                import SVC, SVR
    from sklearn.tree               import DecisionTreeClassifier
    from sklearn.neighbors          import KNeighborsClassifier
    from sklearn.cluster            import KMeans
    from sklearn.model_selection    import train_test_split, cross_val_score
    from sklearn.preprocessing      import StandardScaler, LabelEncoder
    from sklearn.metrics            import (
        accuracy_score, r2_score, mean_squared_error,
        classification_report, confusion_matrix,
    )
    _SK = True
except ImportError:
    _SK = False

# ── Matplotlib ────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend by default
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


def _require_np() -> None:
    if not _NP:
        raise RuntimeError(
            "NumPy is required for this operation. Install: pip install numpy"
        )

def _require_pd() -> None:
    if not _PD:
        raise RuntimeError(
            "Pandas is required for this operation. Install: pip install pandas"
        )

def _require_sk() -> None:
    if not _SK:
        raise RuntimeError(
            "scikit-learn is required for this operation. "
            "Install: pip install scikit-learn"
        )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, Any] = {
    # Classification
    "logistic"         : lambda: LogisticRegression(max_iter=1000) if _SK else None,
    "random_forest"    : lambda: RandomForestClassifier(n_estimators=100) if _SK else None,
    "random_forest_clf": lambda: RandomForestClassifier(n_estimators=100) if _SK else None,
    "svm"              : lambda: SVC(kernel="rbf", probability=True) if _SK else None,
    "svc"              : lambda: SVC(kernel="rbf", probability=True) if _SK else None,
    "decision_tree"    : lambda: DecisionTreeClassifier() if _SK else None,
    "knn"              : lambda: KNeighborsClassifier(n_neighbors=5) if _SK else None,
    "gradient_boost"   : lambda: GradientBoostingClassifier() if _SK else None,
    # Regression
    "linear"           : lambda: LinearRegression() if _SK else None,
    "linear_regression": lambda: LinearRegression() if _SK else None,
    "random_forest_reg": lambda: RandomForestRegressor(n_estimators=100) if _SK else None,
    "svr"              : lambda: SVR(kernel="rbf") if _SK else None,
}


# ---------------------------------------------------------------------------
# MLDSEngine
# ---------------------------------------------------------------------------

class MLDSEngine:
    """
    High-level ML/DS function container for NithinLang.
    All methods are injected into the exec namespace.
    """

    # =========================================================================
    # Data loading & manipulation
    # =========================================================================

    def data_chudu(
        self,
        path     : str,
        sep      : str = ",",
        encoding : str = "utf-8",
    ) -> "pd.DataFrame":
        """
        Load data from CSV, JSON, Excel, or Parquet file.

        Args:
            path     : Path to the file.
            sep      : Column separator for CSV files (default ',').
            encoding : File encoding (default 'utf-8').

        Returns:
            A pandas DataFrame.

        Example (telugu):
            df = data_chudu("iris.csv")
            raayi(df)
        """
        _require_pd()
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".csv":
            return pd.read_csv(path, sep=sep, encoding=encoding)
        elif ext == ".json":
            return pd.read_json(path, encoding=encoding)
        elif ext in (".xls", ".xlsx"):
            return pd.read_excel(path)
        elif ext == ".parquet":
            return pd.read_parquet(path)
        elif ext == ".tsv":
            return pd.read_csv(path, sep="\t", encoding=encoding)
        else:
            # Attempt CSV as default
            return pd.read_csv(path, sep=sep, encoding=encoding)

    def data_save(
        self,
        df       : "pd.DataFrame",
        path     : str,
        index    : bool = False,
        encoding : str = "utf-8",
    ) -> str:
        """
        Save a DataFrame to a CSV (or JSON/Parquet based on extension).

        Returns:
            Absolute path of saved file.
        """
        _require_pd()
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".json":
            df.to_json(path, orient="records", force_ascii=False, indent=2)
        elif ext == ".parquet":
            df.to_parquet(path, index=index)
        else:
            df.to_csv(path, index=index, encoding=encoding)
        return os.path.abspath(path)

    def data_describe(self, df: "pd.DataFrame") -> str:
        """
        Return a string summary of DataFrame statistics.

        Example:
            df = data_chudu("iris.csv")
            print(data_describe(df))
        """
        _require_pd()
        lines = [
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Columns: {list(df.columns)}",
            "",
            "Data Types:",
            str(df.dtypes),
            "",
            "Null counts:",
            str(df.isnull().sum()),
            "",
            "Statistical summary:",
            str(df.describe(include="all")),
        ]
        return "\n".join(lines)

    def data_filter(
        self,
        df       : "pd.DataFrame",
        column   : str,
        operator : str,
        value    : Any,
    ) -> "pd.DataFrame":
        """
        Filter DataFrame rows.

        Args:
            df       : Input DataFrame.
            column   : Column name to filter on.
            operator : One of "==", "!=", ">", ">=", "<", "<=", "contains",
                       "startswith", "endswith".
            value    : The value to compare against.

        Returns:
            Filtered DataFrame.

        Example:
            filtered = data_filter(df, "species", "==", "setosa")
        """
        _require_pd()
        col = df[column]
        ops = {
            "=="         : lambda c, v: c == v,
            "!="         : lambda c, v: c != v,
            ">"          : lambda c, v: c > v,
            ">="         : lambda c, v: c >= v,
            "<"          : lambda c, v: c < v,
            "<="         : lambda c, v: c <= v,
            "contains"   : lambda c, v: c.str.contains(str(v), na=False),
            "startswith" : lambda c, v: c.str.startswith(str(v), na=False),
            "endswith"   : lambda c, v: c.str.endswith(str(v), na=False),
        }
        if operator not in ops:
            raise ValueError(
                f"data_filter: Unknown operator '{operator}'. "
                f"Choose from: {list(ops.keys())}"
            )
        mask = ops[operator](col, value)
        return df[mask].reset_index(drop=True)

    def data_sort(
        self,
        df        : "pd.DataFrame",
        column    : Union[str, List[str]],
        ascending : bool = True,
    ) -> "pd.DataFrame":
        """
        Sort DataFrame by one or more columns.

        Example:
            sorted_df = data_sort(df, "sepal_length", ascending=False)
        """
        _require_pd()
        return df.sort_values(by=column, ascending=ascending).reset_index(drop=True)

    def data_group(
        self,
        df        : "pd.DataFrame",
        group_col : Union[str, List[str]],
        agg_col   : str,
        agg_func  : str = "mean",
    ) -> "pd.DataFrame":
        """
        GroupBy aggregation.

        Args:
            df        : Input DataFrame.
            group_col : Column(s) to group by.
            agg_col   : Column to aggregate.
            agg_func  : Aggregation function: "mean","sum","count","max","min","std".

        Returns:
            Grouped DataFrame.

        Example:
            grouped = data_group(df, "species", "sepal_length", "mean")
        """
        _require_pd()
        valid_funcs = {"mean", "sum", "count", "max", "min", "std", "median"}
        if agg_func not in valid_funcs:
            raise ValueError(
                f"data_group: Unknown agg_func '{agg_func}'. "
                f"Choose from: {valid_funcs}"
            )
        return (
            df.groupby(group_col)[agg_col]
            .agg(agg_func)
            .reset_index()
            .rename(columns={agg_col: f"{agg_col}_{agg_func}"})
        )

    def data_plot(
        self,
        df   : "pd.DataFrame",
        kind : str = "line",
        x    : Optional[str] = None,
        y    : Optional[str] = None,
        title: str = "",
        save_to: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot a DataFrame using matplotlib.

        Args:
            df      : DataFrame to plot.
            kind    : "line", "bar", "scatter", "hist", "box", "pie".
            x       : Column for x-axis (scatter / line / bar).
            y       : Column for y-axis (scatter / line).
            title   : Plot title.
            save_to : If given, save plot to this file path.

        Returns:
            File path if saved, else None.

        Example:
            data_plot(df, "scatter", x="sepal_length", y="sepal_width",
                      title="Iris", save_to="plot.png")
        """
        _require_pd()
        if not _MPL:
            warnings.warn("matplotlib not installed — cannot plot.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        plot_kwargs: Dict[str, Any] = {"ax": ax}
        if x:
            plot_kwargs["x"] = x
        if y:
            plot_kwargs["y"] = y

        if kind == "scatter":
            if x and y:
                ax.scatter(df[x], df[y])
                ax.set_xlabel(x)
                ax.set_ylabel(y)
            else:
                df.plot(kind="scatter", **plot_kwargs)
        elif kind == "hist":
            col = y or (df.select_dtypes(include="number").columns[0])
            df[col].hist(ax=ax, bins=20)
        elif kind == "pie":
            col = y or df.columns[0]
            df[col].value_counts().plot.pie(ax=ax)
        else:
            df.plot(kind=kind, **plot_kwargs)

        ax.set_title(title or f"{kind.capitalize()} Plot")
        plt.tight_layout()

        if save_to:
            plt.savefig(save_to, dpi=150)
            plt.close(fig)
            return os.path.abspath(save_to)

        plt.show()
        plt.close(fig)
        return None

    # =========================================================================
    # Model training & evaluation
    # =========================================================================

    def model_train(
        self,
        X           : Any,
        y           : Any,
        kind        : str = "random_forest",
        test_size   : float = 0.2,
        random_state: int = 42,
        **kwargs    : Any,
    ) -> Any:
        """
        Train a machine learning model.

        Args:
            X           : Features (DataFrame, ndarray, or list).
            y           : Labels / target (Series, ndarray, or list).
            kind        : Model type. Options:
                          "logistic", "random_forest", "svm",
                          "decision_tree", "knn", "gradient_boost",
                          "linear", "linear_regression",
                          "random_forest_reg", "svr".
            test_size   : Fraction of data held out for validation.
            random_state: Reproducibility seed.
            **kwargs    : Passed to the estimator constructor.

        Returns:
            Trained sklearn estimator.

        Example:
            df    = data_chudu("iris.csv")
            X     = df[["sepal_length","sepal_width"]]
            y     = df["species"]
            model = model_train(X, y, "random_forest")
        """
        _require_sk()
        _require_np()

        kind_lower = kind.lower()
        if kind_lower not in _MODEL_REGISTRY:
            raise ValueError(
                f"model_train: Unknown model kind '{kind}'. "
                f"Available: {list(_MODEL_REGISTRY.keys())}"
            )

        estimator = _MODEL_REGISTRY[kind_lower]()
        if estimator is None:
            raise RuntimeError("model_train: scikit-learn is not installed.")

        # Apply any extra kwargs (e.g., n_estimators=200)
        for k, v in kwargs.items():
            setattr(estimator, k, v)

        # Convert to numpy
        X_arr = np.asarray(X) if _NP else list(X)
        y_arr = np.asarray(y) if _NP else list(y)

        # Train / validation split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_arr, y_arr, test_size=test_size, random_state=random_state
        )

        estimator.fit(X_tr, y_tr)

        # Print quick validation metrics
        y_pred = estimator.predict(X_val)
        try:
            score = accuracy_score(y_val, y_pred)
            print(f"[NithinLang ML] Validation accuracy: {score*100:.2f}%")
        except Exception:
            try:
                score = r2_score(y_val, y_pred)
                print(f"[NithinLang ML] Validation R²: {score:.4f}")
            except Exception:
                pass

        return estimator

    def model_test(
        self,
        model : Any,
        X     : Any,
        y     : Any,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        Returns:
            Dict with keys: "accuracy" (or "r2"), "mse" (regression),
            "report" (classification), "confusion_matrix".

        Example:
            metrics = model_test(model, X_test, y_test)
            print(metrics["accuracy"])
        """
        _require_sk()
        X_arr  = np.asarray(X)  if _NP else list(X)
        y_arr  = np.asarray(y)  if _NP else list(y)
        y_pred = model.predict(X_arr)

        result: Dict[str, Any] = {}

        try:
            acc = accuracy_score(y_arr, y_pred)
            result["accuracy"] = acc
            if verbose:
                print(f"[NithinLang ML] Test Accuracy: {acc*100:.2f}%")
                print("[NithinLang ML] Classification Report:")
                print(classification_report(y_arr, y_pred))
                print("[NithinLang ML] Confusion Matrix:")
                print(confusion_matrix(y_arr, y_pred))
            result["report"]           = classification_report(y_arr, y_pred)
            result["confusion_matrix"] = confusion_matrix(y_arr, y_pred).tolist()
        except Exception:
            try:
                r2  = r2_score(y_arr, y_pred)
                mse = mean_squared_error(y_arr, y_pred)
                result["r2"]  = r2
                result["mse"] = mse
                if verbose:
                    print(f"[NithinLang ML] R²:  {r2:.4f}")
                    print(f"[NithinLang ML] MSE: {mse:.4f}")
            except Exception:
                result["error"] = "Could not compute metrics."

        return result

    def model_predict(
        self,
        model : Any,
        X     : Any,
    ) -> Any:
        """
        Generate predictions from a trained model.

        Example:
            preds = model_predict(model, [[5.1, 3.5], [4.9, 3.0]])
        """
        _require_sk()
        X_arr = np.asarray(X) if _NP else list(X)
        return model.predict(X_arr)

    def model_save(self, model: Any, path: str) -> str:
        """
        Save a trained model to disk using pickle.

        Returns:
            Absolute path of saved file.
        """
        with open(path, "wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return os.path.abspath(path)

    def model_load(self, path: str) -> Any:
        """
        Load a trained model from disk.

        Returns:
            Unpickled sklearn estimator.
        """
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # =========================================================================
    # Clustering
    # =========================================================================

    def cluster_fit(
        self,
        X           : Any,
        k           : int = 3,
        random_state: int = 42,
        max_iter    : int = 300,
    ) -> Dict[str, Any]:
        """
        Fit KMeans clustering.

        Args:
            X           : Feature matrix.
            k           : Number of clusters.
            random_state: Seed.
            max_iter    : Maximum iterations.

        Returns:
            Dict with keys:
              "model"    → fitted KMeans object
              "labels"   → cluster label for each sample
              "centers"  → cluster centroid coordinates
              "inertia"  → within-cluster sum of squares

        Example:
            result  = cluster_fit(X, k=3)
            labels  = result["labels"]
            centers = result["centers"]
        """
        _require_sk()
        _require_np()

        X_arr = np.asarray(X, dtype=np.float64)
        km    = KMeans(n_clusters=k, random_state=random_state, max_iter=max_iter, n_init=10)
        km.fit(X_arr)

        return {
            "model"   : km,
            "labels"  : km.labels_.tolist(),
            "centers" : km.cluster_centers_.tolist(),
            "inertia" : float(km.inertia_),
        }

    # =========================================================================
    # NumPy wrappers
    # =========================================================================

    def np_array(self, data: Any, dtype: str = "float64") -> "np.ndarray":
        """Create a NumPy array from a Python list / nested list."""
        _require_np()
        return np.array(data, dtype=dtype)

    def np_zeros(self, shape: Union[int, tuple], dtype: str = "float64") -> "np.ndarray":
        """Array of zeros."""
        _require_np()
        return np.zeros(shape, dtype=dtype)

    def np_ones(self, shape: Union[int, tuple], dtype: str = "float64") -> "np.ndarray":
        """Array of ones."""
        _require_np()
        return np.ones(shape, dtype=dtype)

    def np_linspace(self, start: float, stop: float, num: int = 50) -> "np.ndarray":
        """Linearly spaced values."""
        _require_np()
        return np.linspace(start, stop, num)

    def np_mean(self, arr: Any, axis: Optional[int] = None) -> Any:
        """Mean of array."""
        _require_np()
        return np.mean(np.asarray(arr), axis=axis)

    def np_std(self, arr: Any, axis: Optional[int] = None) -> Any:
        """Standard deviation."""
        _require_np()
        return np.std(np.asarray(arr), axis=axis)

    def np_sum(self, arr: Any, axis: Optional[int] = None) -> Any:
        """Sum of array."""
        _require_np()
        return np.sum(np.asarray(arr), axis=axis)

    def np_max(self, arr: Any, axis: Optional[int] = None) -> Any:
        """Maximum value."""
        _require_np()
        return np.max(np.asarray(arr), axis=axis)

    def np_min(self, arr: Any, axis: Optional[int] = None) -> Any:
        """Minimum value."""
        _require_np()
        return np.min(np.asarray(arr), axis=axis)