"""
Training pipeline for credit card fraud detection.

Run:
    python src/train.py --data_path <kagglehub_path>

Outputs:
    models/xgb_model.joblib
    models/scaler.joblib
    models/shap_explainer.joblib
    Printed PR-AUC, ROC-AUC, classification report for LR baseline and XGBoost
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

from dataload import load_data


def preprocess(df: pd.DataFrame):
    # Time isn't very predictive raw; Amount needs scaling. V1-V28 are already PCA'd.
    df = df.copy()
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Hour"] = (df["Time"] // 3600) % 24  # cyclical hour of day
    df = df.drop(columns=["Time", "Amount"])

    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y, scaler


def evaluate(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"\n--- {name} ---")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    return pr_auc, roc_auc


def main(data_path):
    print("Loading data...")
    df = load_data(data_path)
    X, y, scaler = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Baseline: Logistic Regression with class_weight balanced ---
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    evaluate("Logistic Regression (baseline)", y_test, lr_prob)

    # --- SMOTE + XGBoost ---
    print("\nApplying SMOTE to training set...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"Resampled train shape: {X_train_res.shape}, fraud ratio: {y_train_res.mean():.4f}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_res, y_train_res)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    pr_auc, roc_auc = evaluate("XGBoost (SMOTE)", y_test, xgb_prob)

    # Find threshold that maximizes F1 for a sane operating point
    precisions, recalls, thresholds = precision_recall_curve(y_test, xgb_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"\nBest F1 threshold: {best_threshold:.4f} "
          f"(precision={precisions[best_idx]:.4f}, recall={recalls[best_idx]:.4f})")

    # --- SHAP explainability ---
    print("\nComputing SHAP values (this can take a minute)...")
    explainer = shap.TreeExplainer(xgb_model)
    sample = X_test.sample(min(1000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    # --- Save artifacts ---
    joblib.dump(xgb_model, "models/xgb_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(explainer, "models/shap_explainer.joblib")
    joblib.dump(best_threshold, "models/best_threshold.joblib")
    joblib.dump(list(X.columns), "models/feature_columns.joblib")
    print("\nSaved model, scaler, SHAP explainer, threshold, and feature columns to models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path returned by kagglehub.dataset_download(...)"
    )
    args = parser.parse_args()
    main(args.data_path)