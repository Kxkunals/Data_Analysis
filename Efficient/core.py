import argparse
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from datetime import datetime
import logging
from IPython.display import display, HTML
# --- Scikit-learn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, roc_curve, auc, RocCurveDisplay,
                             ConfusionMatrixDisplay, r2_score, mean_squared_error)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# --- Model Imports ---
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

# --- Parallel Processing ---
from joblib import Parallel, delayed

# --- Rich CLI Library ---
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax

# Ignore common warnings for a cleaner output
warnings.filterwarnings('ignore')

# Initialize Rich Console
console = Console()

# --- Core Functions ---

def detect_problem_type(target_series: pd.Series) -> str:
    """
    Automatically detects if the problem is classification or regression.
    """
    if target_series.dtype in ['object', 'category', 'bool']:
        return 'classification'

    unique_values = target_series.nunique()
    if unique_values < 2:
         raise ValueError("Target column has less than 2 unique values. Cannot perform modeling.")
    # Heuristic: If low cardinality integer feature, treat as classification
    if pd.api.types.is_integer_dtype(target_series) and unique_values < 50 and (unique_values / len(target_series)) < 0.05:
        return 'classification'

    return 'regression'

def get_models(problem_type: str) -> dict:
    """Returns a dictionary of diverse models suitable for the problem type."""
    n_jobs = -1  # Use all available cores for models that support it
    random_state = 42

    if problem_type == 'classification':
        # Base estimators for Stacking
        estimators_clf = [
            ('rf', RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)),
            ('gb', GradientBoostingClassifier(random_state=random_state))
        ]
        models = {
            "Logistic Regression": LogisticRegression(random_state=random_state, n_jobs=n_jobs, max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "AdaBoost": AdaBoostClassifier(random_state=random_state),
            "Support Vector Machine": SVC(random_state=random_state, probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=n_jobs),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=n_jobs),
            "Stacking": StackingClassifier(estimators=estimators_clf, final_estimator=LogisticRegression(), n_jobs=n_jobs)
        }
    else:  # regression
        # Base estimators for Stacking
        estimators_reg = [
            ('rf', RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)),
            ('ridge', Ridge(random_state=random_state))
        ]
        models = {
            "Linear Regression": LinearRegression(n_jobs=n_jobs),
            "Ridge": Ridge(random_state=random_state),
            "Lasso": Lasso(random_state=random_state),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Random Forest": RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
            "Support Vector Machine": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(n_jobs=n_jobs),
            "XGBoost": XGBRegressor(random_state=random_state, n_jobs=n_jobs),
            "Stacking": StackingRegressor(estimators=estimators_reg, final_estimator=Ridge(), n_jobs=n_jobs)
        }
    return models

# -*- coding: utf-8 -*-
"""
AutoFeatSelect: A Lightweight Python Library for Automatic Feature Selection.

This library provides a single class, AutoFeatSelect, to automatically identify and
drop irrelevant, redundant, or low-information features from a pandas DataFrame.
It uses a combination of statistical, mathematical, and model-based techniques
to clean a dataset, making it ready for machine learning.

Core Features:
- Handles both numerical and categorical data.
- Provides a detailed report explaining why each feature was dropped.
- Highly customizable with thresholds for various checks.
- Lightweight, with dependencies only on pandas, numpy, scikit-learn, and statsmodels.

"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

class AutoFeatSelect(BaseEstimator, TransformerMixin):
    """
    Automated feature selection tool to drop irrelevant or redundant features.

    This transformer applies a series of selection methods in a specific order
    to efficiently prune the feature space.

    Parameters
    ----------
    target_col : str, optional (default=None)
        Name of the target column. If provided, it will be used for supervised
        selection methods.
    
    missing_threshold : float, optional (default=0.95)
        Drop columns with a ratio of missing values higher than this threshold.
        
    id_threshold : float, optional (default=0.99)
        Drop columns where the ratio of unique values to rows is higher than this.
        Useful for dropping ID-like columns.
        
    variance_threshold : float, optional (default=0.01)
        Threshold for the VarianceThreshold selector to drop low-variance features.
        
    correlation_threshold : float, optional (default=0.90)
        Drop one of a pair of features with a Pearson correlation higher than this.
        
    use_correlation_clustering : bool, optional (default=True)
        If True, uses hierarchical clustering on the correlation matrix to drop
        redundant features, which can be more robust than pairwise checks. If False,
        uses a simpler pairwise correlation check.

    multicollinearity_threshold : float, optional (default=10.0)
        Variance Inflation Factor (VIF) threshold for dropping collinear features.
        
    feature_importance_threshold : float, optional (default=0.001)
        Threshold for tree-based feature importance. Features with importance
        below this will be dropped. Requires a target column.
        
    mutual_info_threshold : float, optional (default=0.01)
        Threshold for mutual information. Features with MI below this will be dropped.
        Requires a target column.
        
    verbose : bool, optional (default=False)
        If True, prints progress during the fitting process.
    """

    def __init__(self, target_col=None, missing_threshold=0.95, id_threshold=0.99,
                 variance_threshold=0.01, correlation_threshold=0.90,
                 use_correlation_clustering=True, multicollinearity_threshold=10.0,
                 feature_importance_threshold=0.001, mutual_info_threshold=0.01,
                 verbose=False):
        self.target_col = target_col
        self.missing_threshold = missing_threshold
        self.id_threshold = id_threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.use_correlation_clustering = use_correlation_clustering
        self.multicollinearity_threshold = multicollinearity_threshold
        self.feature_importance_threshold = feature_importance_threshold
        self.mutual_info_threshold = mutual_info_threshold
        self.verbose = verbose
        
        self.dropped_features_report_ = {}
        self.initial_features_ = []
        self.final_features_ = []

    def _log(self, message):
        """Prints a message if verbose is True."""
        if self.verbose:
            print(f"[AutoFeatSelect] {message}")

    def _add_to_report(self, features, reason):
        """Adds dropped features and the reason to the report."""
        if not isinstance(features, list):
            features = [features]
        for feature in features:
            if feature not in self.dropped_features_report_:
                self.dropped_features_report_[feature] = reason

    def fit(self, X, y=None):
        """
        Fits the feature selector to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series or np.array, optional (default=None)
            The target variable. If not provided, the `target_col` in X will be used.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        df = X.copy()
        
        # Separate target if it exists
        if self.target_col and self.target_col in df.columns:
            y = df[self.target_col]
            df = df.drop(columns=[self.target_col])
            self._log(f"Target column '{self.target_col}' identified and separated.")

        self.initial_features_ = df.columns.tolist()
        
        # --- Stage 1: Basic Cleanup (Unsupervised) ---
        df = self._drop_missing_values(df)
        df = self._drop_single_value_columns(df)
        df = self._drop_id_like_columns(df)
        
        # --- Stage 2: Statistical Pruning (Unsupervised) ---
        df = self._drop_low_variance_features(df)
        if self.use_correlation_clustering:
            df = self._drop_correlated_features_clustered(df)
        else:
            df = self._drop_correlated_features_pairwise(df)
        df = self._drop_multicollinearity(df)

        # --- Stage 3: Supervised Selection (if target is available) ---
        if y is not None:
            self._log("Target variable provided. Running supervised selection methods.")
            # Ensure y is a pandas Series with the same index as df
            y = pd.Series(y, index=df.index)
            
            # Impute NaNs for supervised methods
            df_imputed, y_imputed = self._impute_for_supervised(df, y)
            
            df = self._drop_low_mutual_information(df_imputed, y_imputed)
            df = self._drop_low_importance_features(df_imputed, y_imputed)
        else:
            self._log("No target variable. Skipping supervised selection methods.")
            
        self.final_features_ = df.columns.tolist()
        self._log(f"Finished selection. Kept {len(self.final_features_)} out of {len(self.initial_features_)} features.")
        return self

    def transform(self, X, drop=True):
        """
        Transforms the data by dropping selected features.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.
        drop : bool, optional (default=True)
            If True, drops the identified features. If False, returns the original DataFrame.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with irrelevant features removed.
        """
        if not drop:
            return X
        
        if not self.final_features_:
             raise RuntimeError("You must call 'fit' before calling 'transform'.")
        
        df = X.copy()
        
        # Ensure target column is preserved if it was in the original transform input
        cols_to_keep = self.final_features_
        if self.target_col and self.target_col in df.columns:
            cols_to_keep = [self.target_col] + self.final_features_
            
        return df[cols_to_keep]

    def fit_transform(self, X, y=None, drop=True):
        """
        Fits to data, then transforms it.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series or np.array, optional (default=None)
            The target variable.
        drop : bool, optional (default=True)
            If True, drops the identified features. If False, returns the original DataFrame.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.
        """
        self.fit(X, y)
        return self.transform(X, drop=drop)
    
    def get_report(self, as_dataframe=True):
        """
        Returns a report of dropped features and the reasons.

        Parameters
        ----------
        as_dataframe : bool, optional (default=True)
            If True, returns the report as a pandas DataFrame. Otherwise, returns a dict.

        Returns
        -------
        pd.DataFrame or dict
            A report detailing which features were dropped and why.
        """
        if not self.dropped_features_report_:
            print("No features were dropped.")
            return pd.DataFrame(columns=['Feature', 'Reason Dropped']) if as_dataframe else {}
            
        if as_dataframe:
            report_df = pd.DataFrame(list(self.dropped_features_report_.items()),
                                     columns=['Feature', 'Reason Dropped'])
            report_df.sort_values(by='Reason Dropped', inplace=True)
            return report_df
        return self.dropped_features_report_

    # --- Private Helper Methods for each selection stage ---

    def _drop_missing_values(self, df):
        self._log("Running: Drop high missing values...")
        missing_ratio = df.isnull().sum() / len(df)
        high_missing_cols = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()
        if high_missing_cols:
            self._add_to_report(high_missing_cols, f"Missing > {self.missing_threshold*100:.0f}%")
            df = df.drop(columns=high_missing_cols)
            self._log(f"  Dropped: {high_missing_cols}")
        return df

    def _drop_single_value_columns(self, df):
        self._log("Running: Drop single value columns...")
        single_value_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        if single_value_cols:
            self._add_to_report(single_value_cols, "Single unique value")
            df = df.drop(columns=single_value_cols)
            self._log(f"  Dropped: {single_value_cols}")
        return df

    def _drop_id_like_columns(self, df):
        self._log("Running: Drop ID-like columns...")
        id_cols = []
        for col in df.columns:
            # Check for high cardinality primarily in object/int columns
            if df[col].dtype in ['object', 'int64']:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > self.id_threshold:
                    id_cols.append(col)
        
        if id_cols:
            self._add_to_report(id_cols, f"ID-like (unique > {self.id_threshold*100:.0f}%)")
            df = df.drop(columns=id_cols)
            self._log(f"  Dropped: {id_cols}")
        return df

    def _drop_low_variance_features(self, df):
        self._log("Running: Drop low variance features...")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            return df
        
        temp_df = df[numeric_cols].dropna() # VarianceThreshold can't handle NaNs
        if temp_df.empty:
            return df

        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(temp_df)
        
        low_variance_cols = [col for col, var in zip(temp_df.columns, selector.variances_) if var < self.variance_threshold]
        
        if low_variance_cols:
            self._add_to_report(low_variance_cols, f"Variance < {self.variance_threshold}")
            df = df.drop(columns=low_variance_cols)
            self._log(f"  Dropped: {low_variance_cols}")
        return df

    def _drop_correlated_features_pairwise(self, df):
        self._log("Running: Drop highly correlated features (pairwise)...")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            return df

        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        
        if to_drop:
            self._add_to_report(to_drop, f"High correlation > {self.correlation_threshold}")
            df = df.drop(columns=to_drop)
            self._log(f"  Dropped: {to_drop}")
        return df

    def _drop_correlated_features_clustered(self, df):
        self._log("Running: Drop highly correlated features (hierarchical clustering)...")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            return df

        corr = df[numeric_cols].corr().abs()
        dist = squareform(1 - corr)
        linkage_matrix = hierarchy.linkage(dist, method='average')
        
        clusters = hierarchy.fcluster(linkage_matrix, 1 - self.correlation_threshold, criterion='distance')
        
        cluster_map = pd.DataFrame({'feature': numeric_cols, 'cluster': clusters})
        
        to_drop = []
        for cluster_id in cluster_map['cluster'].unique():
            features_in_cluster = cluster_map[cluster_map['cluster'] == cluster_id]['feature'].tolist()
            if len(features_in_cluster) > 1:
                # Keep the first feature, drop the rest in the cluster
                to_drop.extend(features_in_cluster[1:])
        
        if to_drop:
            self._add_to_report(to_drop, f"Redundant (correlation cluster > {self.correlation_threshold})")
            df = df.drop(columns=to_drop)
            self._log(f"  Dropped: {to_drop}")
        return df

    def _drop_multicollinearity(self, df):
        self._log("Running: Drop multicollinear features (VIF)...")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            return df
            
        X_numeric = df[numeric_cols].dropna()
        if X_numeric.empty:
            return df

        # Iteratively drop features with high VIF
        while True:
            vif = pd.DataFrame()
            vif["feature"] = X_numeric.columns
            vif["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
            
            max_vif = vif['VIF'].max()
            if max_vif > self.multicollinearity_threshold:
                feature_to_drop = vif.sort_values('VIF', ascending=False)['feature'].iloc[0]
                self._add_to_report(feature_to_drop, f"High multicollinearity (VIF > {self.multicollinearity_threshold})")
                X_numeric = X_numeric.drop(columns=[feature_to_drop])
                self._log(f"  Dropped: {feature_to_drop} (VIF: {max_vif:.2f})")
            else:
                break
        
        final_numeric_cols = X_numeric.columns.tolist()
        cols_to_drop_from_df = [col for col in numeric_cols if col not in final_numeric_cols]
        df = df.drop(columns=cols_to_drop_from_df)
        return df

    def _impute_for_supervised(self, df, y):
        """A simple imputer for supervised methods."""
        df_imputed = df.copy()
        y_imputed = y.copy()

        # Impute y if it has missing values
        if y_imputed.isnull().any():
            if pd.api.types.is_numeric_dtype(y_imputed):
                y_imputed.fillna(y_imputed.median(), inplace=True)
            else:
                y_imputed.fillna(y_imputed.mode()[0], inplace=True)

        for col in df_imputed.columns:
            if df_imputed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_imputed[col]):
                    df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
                else:
                    df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
        return df_imputed, y_imputed


    def _drop_low_mutual_information(self, df, y):
        self._log("Running: Drop features with low mutual information...")
        df_encoded = df.copy()
        
        # Label encode categorical features for MI calculation
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            
        if pd.api.types.is_numeric_dtype(y):
            mi_scores = mutual_info_regression(df_encoded, y)
        else:
            mi_scores = mutual_info_classif(df_encoded, y)
            
        mi_series = pd.Series(mi_scores, index=df_encoded.columns)
        low_mi_features = mi_series[mi_series < self.mutual_info_threshold].index.tolist()
        
        if low_mi_features:
            self._add_to_report(low_mi_features, f"Mutual Information < {self.mutual_info_threshold}")
            df = df.drop(columns=low_mi_features)
            self._log(f"  Dropped: {low_mi_features}")
        return df

    def _drop_low_importance_features(self, df, y):
        self._log("Running: Drop low importance features (Tree-based)...")
        df_encoded = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            
        if pd.api.types.is_numeric_dtype(y):
            model = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            model = ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            
        model.fit(df_encoded, y)
        
        importances = pd.Series(model.feature_importances_, index=df_encoded.columns)
        low_importance_features = importances[importances < self.feature_importance_threshold].index.tolist()
        
        if low_importance_features:
            self._add_to_report(low_importance_features, f"Feature Importance < {self.feature_importance_threshold}")
            df = df.drop(columns=low_importance_features)
            self._log(f"  Dropped: {low_importance_features}")
        return df


def build_preprocessor(X: pd.DataFrame, problem_type: str, n_features: int = None, pca_components: int = None) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for preprocessing."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Define pipelines for numeric and categorical features
    numeric_transformer_steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
    if pca_components and pca_components > 0:
        numeric_transformer_steps.append(('pca', PCA(n_components=pca_components)))
        console.print(f"🔩 [cyan]Applying PCA with {pca_components} components.[/cyan]")

    numeric_transformer = Pipeline(steps=numeric_transformer_steps)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create the main preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough',
        n_jobs=-1 # Parallelize the transformation
    )
    return preprocessor

def train_and_evaluate_model(name, model, preprocessor, X_train, y_train, X_test, y_test, problem_type):
    """A helper function to train one model, used for parallel execution."""
    start_time = time.time()

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)

    metric1, metric2 = (0, 0)
    if problem_type == 'classification':
        metric1 = accuracy_score(y_test, y_pred)
        metric2 = f1_score(y_test, y_pred, average='weighted')
    else: # regression
        metric1 = r2_score(y_test, y_pred)
        metric2 = np.sqrt(mean_squared_error(y_test, y_pred))

    elapsed_time = time.time() - start_time

    return {
        "Model": name,
        "Metric1": metric1,
        "Metric2": metric2,
        "Time (s)": elapsed_time,
        "pipeline": pipeline # Return the trained pipeline
    }

def generate_visuals(pipeline, X_test, y_test, problem_type, model_name, output_dir):
    """Generates and saves relevant plots for the best model."""
    console.print(f"🎨 [bold]Generating visuals for {model_name}...[/bold]")
    plt.style.use('seaborn-v0_8-whitegrid')

    if problem_type == 'classification':
        # Confusion Matrix
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'Confusion Matrix: {model_name}')
            plt.tight_layout()
            cm_path = output_dir / f"confusion_matrix_{model_name}.png"
            plt.savefig(cm_path)
            plt.close()
            console.print(f"  ✅ Confusion Matrix saved to [cyan]{cm_path}[/cyan]")
        except Exception as e:
            console.print(f"  ⚠️ Could not generate Confusion Matrix: {e}")

        # ROC Curve
        try:
            if hasattr(pipeline.named_steps['model'], "predict_proba"):
                fig, ax = plt.subplots(figsize=(8, 6))
                RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
                ax.set_title(f'ROC Curve: {model_name}')
                plt.tight_layout()
                roc_path = output_dir / f"roc_curve_{model_name}.png"
                plt.savefig(roc_path)
                plt.close()
                console.print(f"  ✅ ROC Curve saved to [cyan]{roc_path}[/cyan]")
        except Exception as e:
            console.print(f"  ⚠️ Could not generate ROC Curve (possibly not a binary problem or model lacks predict_proba): {e}")

    else: # Regression
        y_pred = pipeline.predict(X_test)

        # Predicted vs. Actual Plot
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Predicted vs. Actual: {model_name}')
            plt.tight_layout()
            pvsa_path = output_dir / f"predicted_vs_actual_{model_name}.png"
            plt.savefig(pvsa_path)
            plt.close()
            console.print(f"  ✅ Predicted vs. Actual plot saved to [cyan]{pvsa_path}[/cyan]")
        except Exception as e:
            console.print(f"  ⚠️ Could not generate Predicted vs. Actual plot: {e}")

        # Residuals Plot
        try:
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.residplot(x=y_pred, y=residuals, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 2}, ax=ax)
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residuals Plot: {model_name}')
            plt.tight_layout()
            res_path = output_dir / f"residuals_{model_name}.png"
            plt.savefig(res_path)
            plt.close()
            console.print(f"  ✅ Residuals plot saved to [cyan]{res_path}[/cyan]")
        except Exception as e:
            console.print(f"  ⚠️ Could not generate Residuals plot: {e}")


def generate_shap_plot(pipeline, X_train, X_test, problem_type, model_name, output_dir):
    """Generates and saves a SHAP summary plot for any model."""
    console.print(f"🤔 [bold]Generating SHAP interpretability plot for {model_name}...[/bold]")
    try:
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']

        # Transform data and get feature names
        X_test_transformed = preprocessor.transform(X_test)

        try:
            # Works for ColumnTransformer with OneHotEncoder
            ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
                X.select_dtypes(include=['object', 'category', 'bool']).columns
            )
            all_feature_names = X.select_dtypes(include=np.number).columns.tolist() + list(ohe_feature_names)
            X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=all_feature_names)
        except Exception:
             # Fallback if feature names can't be extracted
            all_feature_names = None
            X_test_transformed_df = pd.DataFrame(X_test_transformed)


        # For complex models like Stacking, SHAP works best with a prediction function
        if problem_type == 'classification' and hasattr(model, 'predict_proba'):
            predict_fn = lambda x: pipeline.predict_proba(pd.DataFrame(x, columns=X_test.columns))
        else:
            predict_fn = lambda x: pipeline.predict(pd.DataFrame(x, columns=X_test.columns))

        # Use the appropriate explainer
        if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor, XGBClassifier, XGBRegressor)):
             explainer = shap.TreeExplainer(model, data=preprocessor.transform(X_train), feature_perturbation="interventional")
        else:
             # KernelExplainer is a model-agnostic fallback, requires a function and background data
             explainer = shap.KernelExplainer(predict_fn, shap.sample(X_train, 50))

        shap_values = explainer(X_test_transformed_df)

        # For classifiers with predict_proba, shap_values can be a list of arrays (one per class)
        if isinstance(shap_values, list):
             shap_values_to_plot = shap_values[1] # Plot for the positive class
        else:
             shap_values_to_plot = shap_values

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_to_plot, X_test_transformed_df, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance: {model_name}")
        plt.tight_layout()
        
        shap_filename = output_dir / f"shap_summary_{model_name}.png"
        plt.savefig(shap_filename)
        plt.close()
        console.print(f"  ✅ SHAP summary plot saved to [cyan]'{shap_filename}'[/cyan]")

    except Exception as e:
        console.print(f"  [bold yellow]⚠️ Warning: Could not generate SHAP plot. Reason: {e}[/bold yellow]")


def main(args, save_artifacts=False):
    """Main function to run the enhanced ML pipeline.

    Parameters
    ----------
    args : argparse.Namespace or similar
        Arguments including dataset_path, target_column, etc.
    save_artifacts : bool, optional
        Whether to save output artifacts.
    """
    # --- 1. Setup & Introduction ---
    start_run_time = time.time()
    if save_artifacts:
        output_dir = Path(args.output_dir) / f"results_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(Panel("🧠 [bold magenta]Universal ML Model Explorer Pro[/bold magenta] is Starting!", title="🚀 Launching", border_style="green"))
        console.print(f"📂 All artifacts will be saved in: [cyan]{output_dir}[/cyan]")

    # --- 2. Dataset Loading ---
    try:
        df = pd.read_csv(args.dataset_path)
    except Exception as e:
        console.print(f"[bold red]❌ Error loading dataset: {e}[/bold red]")
        return

    if args.target_column not in df.columns:
        console.print(f"[bold red]❌ Error: Target column '{args.target_column}' not found.[/bold red]")
        return

    df = df.dropna(subset=[args.target_column]) # Drop rows where target is NaN
    console.print(f"✅ Dataset loaded successfully. Shape: {df.shape}")

    # --- 3. Data Preparation & Problem Detection ---
    X = df.drop(args.target_column, axis=1)
    y = df[args.target_column]

    problem_type = detect_problem_type(y)
    console.print(Panel(f"🤖 [bold]Detected Problem Type: [cyan]{problem_type.capitalize()}[/cyan][/bold]", title="🔍 Analysis", border_style="cyan"))

    if problem_type == 'classification':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=y.name)
        console.print(f"🏷️ Target variable encoded. Classes: {list(le.classes_)}")

    # --- 4. Preprocessing & Feature Engineering ---
    preprocessor = build_preprocessor(X, problem_type, args.pca_components)

    # The full pipeline will now include feature selection after preprocessing
    # We define the model step later

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' else None)

    # --- 5. Parallel Model Training ---
    models = get_models(problem_type)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        console.print("\n[bold green]🏋️ Training all models in parallel... Please wait![/bold green]\n")

        results = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate_model)(
                name, model, preprocessor, X_train, y_train, X_test, y_test, problem_type
            )
            for name, model in models.items()
        )


    # --- 6. Model Comparison & Best Model Selection ---
    metric1_name = "Accuracy" if problem_type == 'classification' else "R-squared"
    metric2_name = "F1-Score" if problem_type == 'classification' else "RMSE"
    sort_key, reverse_sort = ("Metric1", True) if problem_type == 'classification' else ("Metric1", True)

    results.sort(key=lambda x: x[sort_key], reverse=reverse_sort)
    best_model_result = results[0]
    best_model_name = best_model_result["Model"]
    best_pipeline = best_model_result["pipeline"]

    # Display results table
    table = Table(title=f"📊 [bold]Model Performance Comparison ({problem_type.capitalize()})[/bold]")
    table.add_column("Rank", style="blue")
    table.add_column("Model", style="cyan")
    table.add_column(metric1_name, style="green")
    table.add_column(metric2_name, style="magenta")
    table.add_column("Time (s)", justify="right", style="yellow")

    for i, res in enumerate(results):
        is_best = "👑 " if i == 0 else ""
        table.add_row(f"{i+1}", f"{is_best}{res['Model']}", f"{res['Metric1']:.4f}", f"{res['Metric2']:.4f}", f"{res['Time (s)']:.2f}")

    console.print(table)
    console.print(f"🏆 [bold green]Best Model Identified: {best_model_name}[/bold green]")

    # --- 7. Generate and Save Artifacts ---
    if save_artifacts:
        console.print("\n[bold blue]--- 💾 Generating Final Artifacts ---[/bold blue]")

        # a) Save best model
        model_filename = output_dir / "best_model.pkl"
        joblib.dump(best_pipeline, model_filename)
        console.print(f"✅ Best model saved as [cyan]'{model_filename}'[/cyan]")

        # b) Generate visuals for the best model
        generate_visuals(best_pipeline, X_test, y_test, problem_type, best_model_name, output_dir)

    # c) Generate SHAP plot for the best model
    if not args.no_shap and save_artifacts:
        generate_shap_plot(best_pipeline, X_train, X_test, problem_type, best_model_name, output_dir)

    # d) Save full report
    if save_artifacts:
        report_filename = output_dir / "model_report.txt"
        with open(report_filename, "w") as f:
            f.write(f"--- Universal ML Model Explorer Report ---\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {args.dataset_path}\n")
            f.write(f"Target Variable: {args.target_column}\n")
            f.write(f"Problem Type: {problem_type.capitalize()}\n\n")
            f.write(f"--- Best Model: {best_model_name} ---\n")
            f.write(f"  - {metric1_name}: {best_model_result['Metric1']:.4f}\n")
            f.write(f"  - {metric2_name}: {best_model_result['Metric2']:.4f}\n")
            f.write(f"  - Training Time (s): {best_model_result['Time (s)']:.2f}\n\n")
            f.write("--- Full Model Comparison ---\n")
            header = f"{'Rank':<5} {'Model':<25} {metric1_name:<15} {metric2_name:<15} {'Time (s)':<10}\n"
            f.write(header + "-"*len(header) + "\n")
            for i, res in enumerate(results):
                f.write(f"{i+1:<5} {res['Model']:<25} {res['Metric1']:<15.4f} {res['Metric2']:<15.4f} {res['Time (s)']:.2f}\n")
        console.print(f"📋 Full report saved to [cyan]'{report_filename}'[/cyan]")

    # --- 8. Conclusion ---
    total_runtime = time.time() - start_run_time
    console.print(Panel(f"✅ [bold green]Pipeline finished successfully in {total_runtime:.2f} seconds![/bold green]", title="🏁 Complete", border_style="green"))

def run_pipeline_in_notebook(dataset_path: str = None, target_column: str = None, df: pd.DataFrame = None, save_artifacts: bool = False, **kwargs):
    """
    A helper to run the pipeline from a Jupyter Notebook or another script.

    Args:
        dataset_path (str, optional): Path to the dataset. If not provided, df must be given.
        target_column (str): Name of the target column.
        df (pd.DataFrame, optional): DataFrame to use directly instead of loading from dataset_path.
        save_artifacts (bool): Whether to save models, plots, reports.
        **kwargs: Additional arguments like pca_components, no_shap, etc.
    """
    class Args:
        def __init__(self, dataset_path, target_column, df, save_artifacts, **kwargs):
            self.dataset_path = dataset_path
            self.target_column = target_column
            self.df = df
            self.pca_components = kwargs.get("pca_components", None)
            self.no_shap = kwargs.get("no_shap", False)
            self.output_dir = kwargs.get("output_dir", "results") if save_artifacts else None

    args = Args(dataset_path, target_column, df, save_artifacts, **kwargs)
    main(args, save_artifacts)



class AutoEDA:
    """
    Automated Exploratory Data Analysis tool.

    Generates a comprehensive EDA report for a given DataFrame.

    Parameters
    ----------
    target_col : str, optional
        The name of the target column for bivariate and target-specific analysis.
    save_report : bool, optional (default=False)
        If True, saves the report as an HTML file. Otherwise, displays it inline.
    output_filename : str, optional
        The name of the output HTML file. If not provided, a name with a
        timestamp is generated automatically.
    enable_logging : bool, optional (default=False)
        If True, creates a log file of the operations performed.
    """

    def __init__(self, target_col=None, save_report=False, output_filename=None, enable_logging=False):
        self.target_col = target_col
        self.save_report = save_report
        self.output_filename = output_filename
        self.enable_logging = enable_logging
        self.report_html = ""
        self.start_time = datetime.now()

        if self.enable_logging:
            self._setup_logging()

    def _setup_logging(self):
        """Initializes the logging configuration."""
        log_filename = f"autoeda_log_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'
        )
        self._log_info("Logging enabled.")

    def _log_info(self, message):
        """Logs an informational message."""
        if self.enable_logging:
            logging.info(message)

    def _fig_to_html(self, fig):
        """Converts a Plotly or Matplotlib figure to an HTML string."""
        if isinstance(fig, go.Figure):
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # For Matplotlib/Seaborn figures
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f'<img src="data:image/png;base64,{data}"/>'

    def _add_section(self, title, content):
        """Adds a new section to the HTML report."""
        self.report_html += f"<h2>{title}</h2>{content}<hr>"

    def run(self, df):
        """
        Executes the full EDA process on the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to analyze.
        """
        self._log_info(f"Starting EDA for a DataFrame with shape {df.shape}.")
        self.report_html = f"<h1>EDA Report - {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</h1>"

        # --- Analysis Sections ---
        self._log_info("Analyzing basic info and data types.")
        self._analyze_basic_info(df)
        
        self._log_info("Analyzing missing values.")
        self._analyze_missing_values(df)

        self._log_info("Analyzing duplicate and constant features.")
        self._analyze_duplicates_and_constants(df)

        self._log_info("Performing univariate analysis.")
        self._univariate_analysis(df)
        
        self._log_info("Performing outlier detection.")
        self._outlier_analysis(df)

        self._log_info("Performing bivariate analysis.")
        self._bivariate_analysis(df)

        if self.target_col:
            self._log_info(f"Performing target variable analysis for '{self.target_col}'.")
            self._target_analysis(df)

        # --- Finalize Report ---
        if self.save_report:
            if not self.output_filename:
                self.output_filename = f"eda_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.html"
            with open(self.output_filename, "w", encoding="utf-8") as f:
                f.write(self.report_html)
            print(f"EDA report saved to '{self.output_filename}'")
            self._log_info(f"Report saved to {self.output_filename}.")
        else:
            display(HTML(self.report_html))
        
        end_time = datetime.now()
        self._log_info(f"EDA finished. Total time taken: {end_time - self.start_time}.")

    # --- Private Analysis Methods ---

    def _analyze_basic_info(self, df):
        """Analyzes data types and basic DataFrame info."""
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        info_html = f"""
        <p><b>DataFrame Shape:</b> {df.shape[0]} rows, {df.shape[1]} columns</p>
        <p><b>Numerical Columns ({len(num_cols)}):</b> {', '.join(num_cols)}</p>
        <p><b>Categorical Columns ({len(cat_cols)}):</b> {', '.join(cat_cols)}</p>
        """
        
        # Unique values in categorical columns
        if cat_cols:
            unique_counts = df[cat_cols].nunique().to_frame('Unique Values')
            info_html += "<h3>Unique Values in Categorical Columns</h3>" + unique_counts.to_html()
            
        self._add_section("1. DataFrame Overview & Data Types", info_html)

    def _analyze_missing_values(self, df):
        """Analyzes and visualizes missing values."""
        missing_counts = df.isnull().sum()
        missing_perc = (missing_counts / len(df) * 100).round(2)
        missing_df = pd.DataFrame({'Count': missing_counts, 'Percentage': missing_perc})
        missing_df = missing_df[missing_df['Count'] > 0].sort_values(by='Count', ascending=False)

        content_html = "<h3>Missing Value Counts & Percentage</h3>"
        if missing_df.empty:
            content_html += "<p>No missing values found.</p>"
        else:
            content_html += missing_df.to_html()
            
            # Bar plot for missing values
            fig = go.Figure(go.Bar(x=missing_df.index, y=missing_df['Count'], text=missing_df['Percentage'].astype(str) + '%'))
            fig.update_layout(title='Count of Missing Values per Column', xaxis_title='Columns', yaxis_title='Count')
            content_html += self._fig_to_html(fig)

            # Heatmap for missing values
            plt.figure(figsize=(12, 6))
            sns_plot = sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
            plt.title('Missing Value Heatmap')
            content_html += self._fig_to_html(sns_plot.get_figure())

        self._add_section("2. Missing Value Analysis", content_html)

    def _analyze_duplicates_and_constants(self, df):
        """Finds duplicate rows and constant columns."""
        # Duplicates
        num_duplicates = df.duplicated().sum()
        content_html = f"<p><b>Duplicate Rows:</b> {num_duplicates} found.</p>"
        
        # Constants
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        content_html += f"<p><b>Constant Columns (1 unique value):</b> {len(constant_cols)} found. {', '.join(constant_cols)}</p>"
        
        self._add_section("3. Duplicate & Constant Feature Detection", content_html)

    def _univariate_analysis(self, df):
        """Performs univariate analysis for numerical and categorical columns."""
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        content_html = ""

        # Numerical columns
        if num_cols:
            content_html += "<h3>Numerical Feature Distribution</h3>"
            summary_stats = df[num_cols].describe().T
            summary_stats['skew'] = df[num_cols].skew()
            summary_stats['kurtosis'] = df[num_cols].kurtosis()
            content_html += summary_stats.to_html()

            for col in num_cols:
                fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Histogram of {col}', f'KDE Plot of {col}'))
                fig.add_trace(go.Histogram(x=df[col], name='Histogram'), row=1, col=1)
                kde_fig = ff.create_distplot([df[col].dropna()], [col], show_hist=False, show_rug=False)
                fig.add_trace(kde_fig['data'][0], row=1, col=2)
                fig.update_layout(showlegend=False, height=400, title_text=f"Distribution of {col}")
                content_html += self._fig_to_html(fig)

        # Categorical columns
        if cat_cols:
            content_html += "<h3>Categorical Feature Distribution</h3>"
            for col in cat_cols:
                counts = df[col].value_counts()
                fig = go.Figure(go.Bar(x=counts.index, y=counts.values))
                fig.update_layout(title=f'Value Counts for {col}', xaxis_title=col, yaxis_title='Count')
                content_html += self._fig_to_html(fig)
        
        self._add_section("4. Univariate Analysis", content_html)

    def _outlier_analysis(self, df):
        """Detects outliers using the IQR method."""
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            self._add_section("5. Outlier Detection", "<p>No numerical columns to analyze for outliers.</p>")
            return

        outlier_counts = {}
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                outlier_counts[col] = len(outliers)
        
        content_html = "<h3>Outlier Counts (IQR Method)</h3>"
        if not outlier_counts:
            content_html += "<p>No outliers detected using the 1.5 * IQR rule.</p>"
        else:
            outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['Outlier Count'])
            content_html += outlier_df.to_html()
            
            content_html += "<h3>Boxplots for Columns with Outliers</h3>"
            fig = go.Figure()
            for col in outlier_counts.keys():
                fig.add_trace(go.Box(y=df[col], name=col))
            fig.update_layout(title="Boxplots of Features with Potential Outliers")
            content_html += self._fig_to_html(fig)
            
        self._add_section("5. Outlier Detection", content_html)

    def _bivariate_analysis(self, df):
        """Performs bivariate analysis."""
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        content_html = ""

        if len(num_cols) > 1:
            content_html += "<h3>Numerical Feature Correlation</h3>"
            corr = df[num_cols].corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='Viridis',
                zmin=-1, zmax=1
            ))
            fig.update_layout(title='Correlation Heatmap')
            content_html += self._fig_to_html(fig)

            # Pairplot for top correlated features
            if len(num_cols) > 2:
                top_corr_cols = corr.unstack().sort_values(ascending=False).drop_duplicates()
                top_5_pairs = top_corr_cols[top_corr_cols < 1].head(5).index.tolist()
                if top_5_pairs:
                    cols_to_plot = list(set([p[0] for p in top_5_pairs] + [p[1] for p in top_5_pairs]))
                    content_html += "<h3>Pairplot of Top 5 Correlated Features</h3>"
                    pairplot_fig = sns.pairplot(df[cols_to_plot], kind='reg', diag_kind='kde')
                    content_html += self._fig_to_html(pairplot_fig.fig)

        self._add_section("6. Bivariate Analysis", content_html)

    def _target_analysis(self, df):
        """Analyzes the relationship between features and the target."""
        if self.target_col not in df.columns:
            self._add_section("7. Target Variable Analysis", f"<p>Target column '{self.target_col}' not found in DataFrame.</p>")
            return

        target = df[self.target_col]
        content_html = ""

        # Target distribution
        content_html += "<h3>Target Variable Distribution</h3>"
        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 20: # Regression
            fig = ff.create_distplot([target.dropna()], [self.target_col], show_hist=False)
            fig.update_layout(title=f'Distribution of Target: {self.target_col} (Regression)')
            content_html += self._fig_to_html(fig)
        else: # Classification
            counts = target.value_counts()
            fig = go.Figure(go.Bar(x=counts.index, y=counts.values))
            fig.update_layout(title=f'Class Distribution of Target: {self.target_col} (Classification)')
            content_html += self._fig_to_html(fig)

        # Feature vs Target
        content_html += "<h3>Feature vs. Target Analysis</h3>"
        num_cols = df.select_dtypes(include=np.number).columns.drop(self.target_col, errors='ignore')
        for col in num_cols:
            fig = go.Figure()
            if pd.api.types.is_numeric_dtype(target) and target.nunique() > 20: # Regression
                 fig.add_trace(go.Scatter(x=df[col], y=target, mode='markers', marker=dict(opacity=0.6)))
                 fig.update_layout(title=f'{col} vs. {self.target_col}', xaxis_title=col, yaxis_title=self.target_col)
            else: # Classification
                for cat in target.unique():
                    fig.add_trace(go.Box(y=df[df[self.target_col] == cat][col], name=str(cat)))
                fig.update_layout(title=f'{col} by {self.target_col}', boxmode='group')
            content_html += self._fig_to_html(fig)

        self._add_section("7. Target Variable Analysis", content_html)


# autoclean.py
# An advanced, scikit-learn inspired library for end-to-end tabular data preprocessing.
# Version 2: Enhanced with rich logging, verbosity controls, and improved reliability.

import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')


# --- Optional Rich Integration for better UX ---
try:
    from rich.logging import RichHandler
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class AutoClean(BaseEstimator, TransformerMixin):
    """
    An automated preprocessing library for tabular data that provides a complete
    cleaning pipeline. It follows the sklearn fit/transform pattern for easy
    integration into ML workflows.

    Parameters
    ----------
    config : dict, optional
        A dictionary to configure the preprocessing steps for specific columns.
        If not provided, the library will use default behaviors.

    verbose : bool, optional (default=True)
        If True, enables detailed logging and progress bars using the 'rich' library.
        If False, suppresses all output.
    """
    def __init__(self, config=None, verbose=True):
        self.config = config if config else {}
        self.verbose = verbose
        self._fit_params = {}
        self.summary_ = {}
        self._logger = self._setup_logger()
        if RICH_AVAILABLE and self.verbose:
            self.console = Console()

    def _setup_logger(self):
        """Sets up the logger based on verbosity."""
        logger = logging.getLogger(f"AutoClean_{id(self)}")
        logger.propagate = False # Prevent duplicate logs in root logger
        if self.verbose and RICH_AVAILABLE:
            handler = RichHandler(show_time=False, show_path=False, rich_tracebacks=True)
            handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            handler = logging.NullHandler()
        
        logger.handlers = [handler]
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger

    def fit(self, X, y=None):
        """
        Learns the parameters required for transformation from the data.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        
        X_fit = X.copy()
        
        with self._progress_context("Fitting Pipeline") as progress:
            task = progress.add_task("Total Fit Progress", total=5)

            self._logger.info("[bold cyan]Step 1: Inferring Column Types[/bold cyan]")
            self._infer_types(X_fit)
            progress.update(task, advance=1)

            self._logger.info("[bold cyan]Step 2: Fitting Imputer[/bold cyan]")
            self._fit_imputer(X_fit, y)
            X_fit = self._transform_imputer(X_fit, is_fit_phase=True)
            progress.update(task, advance=1)

            self._logger.info("[bold cyan]Step 3: Fitting Outlier Handler[/bold cyan]")
            self._fit_outliers(X_fit)
            progress.update(task, advance=1)

            self._logger.info("[bold cyan]Step 4: Fitting Encoder[/bold cyan]")
            self._fit_encoder(X_fit)
            progress.update(task, advance=1)

            self._logger.info("[bold cyan]Step 5: Fitting Scaler[/bold cyan]")
            self._fit_scaler(X_fit)
            progress.update(task, advance=1)

        self._logger.info("[bold green]:white_check_mark: Fitting complete.[/bold green]")
        return self

    def transform(self, X):
        """
        Applies the learned preprocessing transformations to the data.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
            
        X_transformed = X.copy()
        self.summary_ = {'actions': []}

        with self._progress_context("Transforming Data") as progress:
            task = progress.add_task("Total Transform Progress", total=4)

            X_transformed = self._transform_imputer(X_transformed)
            progress.update(task, advance=1)
            
            X_transformed = self._transform_outliers(X_transformed)
            progress.update(task, advance=1)
            
            X_transformed = self._transform_encoder(X_transformed)
            progress.update(task, advance=1)
            
            X_transformed = self._transform_scaler(X_transformed)
            progress.update(task, advance=1)

        self._generate_summary()
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
        
    def _progress_context(self, description):
        """Creates a rich.progress context or a dummy context."""
        if RICH_AVAILABLE and self.verbose:
            return Progress(
                TextColumn("[bold blue]{task.description}", justify="right"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
                console=self.console,
            )
        else:
            # Dummy context manager if rich is not available or verbose is False
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def add_task(self, *args, **kwargs): return 0
                def update(self, *args, **kwargs): pass
            return DummyContext()

    # --- Private Helper Methods ---

    def _infer_types(self, df):
        self._fit_params['types'] = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self._fit_params['types'][col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                self._fit_params['types'][col] = 'datetime'
            else:
                self._fit_params['types'][col] = 'categorical'

    def _fit_imputer(self, df, y):
        self._fit_params['imputation'] = {}
        impute_config = self.config.get('impute', {})
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                strategy = impute_config.get(col, 'mean' if self._fit_params['types'][col] == 'numeric' else 'mode')
                params = {'strategy': strategy}
                if strategy in ['mean', 'median']:
                    params['value'] = df[col].agg(strategy)
                elif strategy == 'mode':
                    params['value'] = df[col].mode()[0]
                elif strategy == 'constant':
                    params['value'] = self.config.get('impute', {}).get(col, {}).get('fill_value', 0)
                elif strategy == 'predictive' and self._fit_params['types'][col] == 'numeric':
                    # Use only other complete numeric columns for prediction to ensure reliability
                    predictor_cols = [c for c, t in self._fit_params['types'].items() if t == 'numeric' and c != col and df[c].isnull().sum() == 0]
                    if predictor_cols:
                        imputer_model = IterativeImputer(RandomForestRegressor(n_estimators=10, random_state=0), max_iter=5, random_state=0)
                        imputer_model.fit(df[predictor_cols], df[col])
                        params['model'] = imputer_model
                        params['features'] = predictor_cols
                    else:
                        # Fallback if no suitable predictors are found
                        params['strategy'] = 'mean'
                        params['value'] = df[col].mean()
                        self._logger.warning(f"Predictive imputation for '{col}' failed (no suitable predictors). Falling back to 'mean'.")
                self._fit_params['imputation'][col] = params

    def _transform_imputer(self, df, is_fit_phase=False):
        """
        Transforms missing values using the strategies learned in _fit_imputer.
        Parameters
        ----------
        df : DataFrame
            The data to be transformed.
        is_fit_phase : bool, optional
            Whether this is being called from the fit phase. Defaults to False.
        Returns
        -------
        DataFrame
            Transformed DataFrame with imputed values.
        """
        df_copy = df.copy()

        # Loop through all columns that need imputation
        for col, params in self._fit_params.get('imputation', {}).items():
            # Only proceed if column has missing values
            if df_copy[col].isnull().sum() > 0:
                strategy = params['strategy']

                # Handle simple strategies
                if strategy in ['mean', 'median', 'mode', 'constant']:
                    fill_value = params['value']

                    # Fill missing values directly
                    df_copy[col].fillna(fill_value, inplace=True)

                    # Format the value string for numeric or non-numeric
                    val_str = f"{fill_value:.2f}" if isinstance(fill_value, (int, float)) else f"'{fill_value}'"

                    # Add to transformation summary only during transform phase
                    if not is_fit_phase:
                        self.summary_['actions'].append(
                            f"Imputed {df[col].isnull().sum()} missing values in '{col}' with {strategy} value ({val_str})."
                        )

                # Handle predictive model-based imputation
                elif strategy == 'predictive' and 'model' in params:
                    missing_mask = df_copy[col].isnull()

                    # Check if there are still missing values
                    if missing_mask.sum() > 0:
                        pred_features = df_copy.loc[missing_mask, params['features']]

                        # Predict and fill only if we have valid features
                        if not pred_features.empty:
                            df_copy.loc[missing_mask, col] = params['model'].transform(pred_features)

                            # Log only if not in fit phase
                            if not is_fit_phase:
                                self.summary_['actions'].append(
                                    f"Imputed {missing_mask.sum()} missing values in '{col}' using a predictive model."
                                )

        return df_copy


    def _fit_outliers(self, df):
        self._fit_params['outliers'] = {}
        outlier_config = self.config.get('outliers', {})
        for col, type in self._fit_params['types'].items():
            if type == 'numeric':
                col_config = outlier_config.get(col, {})
                method = col_config.get('method', 'iqr')
                capping = col_config.get('capping', True)
                lower, upper = None, None
                if method == 'iqr':
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                elif method == 'zscore':
                    mean, std = df[col].mean(), df[col].std()
                    thresh = col_config.get('threshold', 3)
                    lower, upper = mean - thresh * std, mean + thresh * std
                if lower is not None:
                    self._fit_params['outliers'][col] = {'method': method, 'bounds': (lower, upper), 'capping': capping}

    def _transform_outliers(self, df):
        df_copy = df.copy()
        for col, params in self._fit_params.get('outliers', {}).items():
            if params.get('capping'):
                lower, upper = params['bounds']
                original = df_copy[col].copy()
                df_copy[col] = df_copy[col].clip(lower, upper)
                outliers_count = (original < lower).sum() + (original > upper).sum()
                if outliers_count > 0:
                    self.summary_['actions'].append(f"Capped {outliers_count} outliers in [bold magenta]'{col}'[/bold magenta] using {params['method']} method.")
        return df_copy

    def _fit_encoder(self, df):
        self._fit_params['encoding'] = {}
        encode_config = self.config.get('encode', {})
        for col, type in self._fit_params['types'].items():
            if type == 'categorical':
                strategy = encode_config.get(col, 'ohe')
                if strategy == 'ohe':
                    # NaN is handled by pandas, we just need to know the categories
                    categories = df[col].dropna().unique().tolist()
                    self._fit_params['encoding'][col] = {'strategy': 'ohe', 'categories': categories}
                elif strategy == 'ordinal':
                    categories = df[col].value_counts().index.tolist()
                    mapping = {cat: i for i, cat in enumerate(categories)}
                    self._fit_params['encoding'][col] = {'strategy': 'ordinal', 'mapping': mapping}

    def _transform_encoder(self, df):
        df_copy = df.copy()
        for col, params in self._fit_params.get('encoding', {}).items():
            strategy = params['strategy']
            if col in df_copy.columns:
                if strategy == 'ohe':
                    # Use pd.Categorical to ensure consistency with fitted categories
                    df_copy[col] = pd.Categorical(df_copy[col], categories=params['categories'])
                    dummies = pd.get_dummies(df_copy[col], prefix=col, prefix_sep='=', dummy_na=False) # dummy_na=False is more reliable
                    df_copy = pd.concat([df_copy.drop(col, axis=1), dummies], axis=1)
                    self.summary_['actions'].append(f"One-hot encoded [bold magenta]'{col}'[/bold magenta] into {len(dummies.columns)} columns.")
                elif strategy == 'ordinal':
                    mapping = params['mapping']
                    df_copy[col] = df_copy[col].map(mapping).fillna(-1).astype(int) # -1 for unseen categories
                    self.summary_['actions'].append(f"Ordinal encoded [bold magenta]'{col}'[/bold magenta] using a learned frequency-based mapping.")
        return df_copy

    def _fit_scaler(self, df):
        self._fit_params['scaling'] = {}
        scale_config = self.config.get('scale', {})
        for col, type in self._fit_params['types'].items():
            if type == 'numeric' and col in df.columns: # Check if col still exists after encoding
                strategy = scale_config.get(col, 'StandardScaler')
                scaler_map = {'MinMaxScaler': MinMaxScaler, 'RobustScaler': RobustScaler, 'StandardScaler': StandardScaler}
                scaler = scaler_map.get(strategy, StandardScaler)()
                scaler.fit(df[[col]])
                self._fit_params['scaling'][col] = scaler

    def _transform_scaler(self, df):
        df_copy = df.copy()
        for col, scaler in self._fit_params.get('scaling', {}).items():
            if col in df_copy.columns:
                df_copy[col] = scaler.transform(df_copy[[col]])
                self.summary_['actions'].append(f"Scaled [bold magenta]'{col}'[/bold magenta] using {scaler.__class__.__name__}.")
        return df_copy

    def _generate_summary(self):
        if not self.verbose or not RICH_AVAILABLE:
            return
        
        summary_text = "\n".join([f"• {action}" for action in self.summary_['actions']])
        if not summary_text:
            summary_text = "No transformations were applied based on the configuration."
            
        panel = Panel(
            summary_text,
            title="[bold green]AutoClean Transformation Summary[/bold green]",
            border_style="blue"
        )
        self.console.print(panel)

