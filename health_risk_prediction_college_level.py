
"""
Health Risk Prediction Pipeline (College-Level)
------------------------------------------------
Author: ChatGPT
Date: 2025-05-31

Description
-----------
This script implements an end‑to‑end machine‑learning pipeline to predict
lifestyle‑disease risk levels (Low / Medium / High) from synthetic health
metrics.  The pipeline includes:

1.  Exploratory Data Analysis (EDA)
2.  Data Cleaning & Feature Engineering
3.  Train/Test Split with Stratified Sampling
4.  Pre‑processing using ColumnTransformer & Pipelines
5.  Baseline Models (Logistic Regression, RandomForest, XGBoost)
6.  Hyper‑parameter Tuning via GridSearchCV
7.  Model Evaluation with Classification & ROC/PR metrics
8.  Explainability via SHAP values
9.  Model Persistence (joblib)
10. Re‑usable Predict Function
11. Optional interactive section for Google Colab / Jupyter users.

Requirements
------------
pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap joblib

Dataset
-------
Pass the csv path as the --data argument (default: health_data.csv).

Usage
-----
python health_risk_prediction_college_level.py --data health_data.csv

In Google Colab:
    !python health_risk_prediction_college_level.py --data health_data.csv

"""

# ----------------------- Imports -----------------------------------------

import argparse
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, RocCurveDisplay,
                             PrecisionRecallDisplay)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

import shap


# ----------------------- Utility Functions -------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from '{path}'\n")
    return df


def eda(df: pd.DataFrame, target: str):
    """Perform basic EDA with plots."""
    print("=== Data Overview ===")
    print(df.describe(include='all'))
    print("\n=== Missing Values ===")
    print(df.isna().sum())

    # Histograms
    df.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="viridis")
    plt.title("Correlation Heatmap")
    plt.show()

    # Target distribution
    sns.countplot(x=target, data=df)
    plt.title("Target Distribution")
    plt.show()


def build_preprocessor(numeric_features, categorical_features):
    """Create a ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor


def evaluate(model, X_test, y_test, class_names, prefix="Model"):
    """Print metrics and plot curves."""
    y_pred = model.predict(X_test)
    print(f"\n=== {prefix} Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{prefix} Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # One-vs-Rest ROC AUC (requires label binarization)
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_test, classes=class_names)
    y_score = model.predict_proba(X_test)
    if y_bin.shape[1] == y_score.shape[1]:
        roc_auc = roc_auc_score(y_bin, y_score, average='macro', multi_class='ovr')
        print(f"Macro ROC AUC: {roc_auc:.3f}")

        RocCurveDisplay.from_predictions(
            y_bin.ravel(), y_score.ravel(), name=prefix)
        plt.show()

        PrecisionRecallDisplay.from_predictions(
            y_bin.ravel(), y_score.ravel(), name=prefix)
        plt.show()


def explain(model, X_train):
    """Explain model predictions with SHAP."""
    print("\n=== SHAP Explainability ===")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=True)


# ----------------------- Main Routine ------------------------------------

def main(data_path: str, perform_eda: bool = True, explain_flag: bool = False):
    df = load_data(data_path)

    target = 'risk_level'
    y = df[target]
    X = df.drop(columns=[target])

    # Identify numeric & categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if perform_eda:
        eda(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # ------------------- Baseline Models ---------------------------------
    models = {
        'LogReg': LogisticRegression(max_iter=1000, multi_class='ovr'),
        'RF': RandomForestClassifier(),
    }
    if XGBClassifier is not None:
        models['XGB'] = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')

    results = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', clf)])
        pipe.fit(X_train, y_train)
        evaluate(pipe, X_test, y_test, sorted(y.unique()), prefix=name)
        results[name] = pipe

    # ------------------- Hyper‑parameter Tuning ---------------------------
    param_grid = {
        'classifier__n_estimators': [100, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
    }
    grid_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    grid = GridSearchCV(grid_pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("\n=== Best Params (GridSearchCV) ===")
    print(grid.best_params_)

    best_model = grid.best_estimator_
    evaluate(best_model, X_test, y_test, sorted(y.unique()), prefix='Tuned RF')

    if explain_flag:
        # SHAP kernel explainer may be needed if not tree-based
        explain(best_model['classifier'], best_model['preprocessor'].transform(X_train)[:200])

    # ------------------- Save Best Model ----------------------------------
    output_dir = Path('artifacts')
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / 'best_model.joblib'
    joblib.dump(best_model, model_path)
    print(f"\nSaved best model to {model_path}")

    # ------------------- Predict Function --------------------------------
    def predict_single(sample_dict):
        """Predict risk level for a single sample (dict)."""
        sample_df = pd.DataFrame([sample_dict])
        pred = best_model.predict(sample_df)[0]
        proba = best_model.predict_proba(sample_df)[0]
        return pred, proba

    # Example usage
    example = {col: X.iloc[0][col] for col in X.columns}
    pred, prob = predict_single(example)
    print(f"\nExample prediction: {pred} (proba = {prob})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Health Risk Prediction Pipeline')
    parser.add_argument('--data', type=str, default='health_data.csv',
                        help='Path to CSV dataset')
    parser.add_argument('--no-eda', action='store_true',
                        help='Skip exploratory data analysis')
    parser.add_argument('--explain', action='store_true',
                        help='Run SHAP explainability (slow)')
    args = parser.parse_args()

    main(args.data, perform_eda=not args.no_eda, explain_flag=args.explain)
