"""
Diabetic Retinopathy — Clinical Feature-Based Prediction Model
================================================================
Trains on the patient dataset (Diabetic_Retinopathy_Dataset.xlsx)
using tabular clinical features to predict DR Grade (0–4).

Features used:
  - Demographics: Age, Gender, Ethnicity
  - Diabetes:    Type, Duration, HbA1c%
  - Vitals:      Systolic BP, Diastolic BP, BMI, Cholesterol
  - Comorbid:    Smoking, Kidney Disease, Neuropathy
  - Ophthalmic:  Eye Side, VA Right, VA Left, IOP mmHg

Target: Grade (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative)

Models:  Random Forest, Gradient Boosting (XGBoost-style), Neural Network
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, cohen_kappa_score, roc_auc_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, VotingClassifier
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
DATASET_PATH = r"c:\Users\T M lakshmi narasimh\Downloads\Diabetic_Retinopathy_Dataset.xlsx"
OUTPUT_DIR   = Path("./outputs")
MODEL_DIR    = Path("./models")
SEED         = 42

CLASS_NAMES  = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    """Load the Excel dataset and clean column names."""
    df = pd.read_excel(path, header=1)
    print(f"[✓] Loaded dataset: {df.shape[0]} patients × {df.shape[1]} features")
    print(f"    Grade distribution:\n{df['Grade'].value_counts().sort_index().to_string()}")
    return df


# ─────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Select and transform clinical features for training.

    Returns: X (DataFrame), y (Series), feature_names (list), encoders (dict)
    """
    # --- Target ---
    y = df["Grade"].astype(int)

    # --- Numeric features ---
    numeric_cols = [
        "Age", "Diabetes Duration", "HbA1c %",
        "Systolic BP", "Diastolic BP", "BMI",
        "Cholesterol mg dL", "IOP mmHg"
    ]

    # --- Categorical features ---
    cat_cols = [
        "Gender", "Ethnicity", "Diabetes Type",
        "Smoking", "Kidney Disease", "Neuropathy", "Eye Side"
    ]

    # --- Visual Acuity encoding (ordinal scale) ---
    va_mapping = {
        "20/20": 10, "20/25": 9, "20/30": 8, "20/40": 7,
        "20/50": 6, "20/60": 5, "20/80": 4, "20/100": 3,
        "20/200": 2, "CF": 1, "HM": 0, "LP": 0, "NLP": 0
    }

    df_feat = df[numeric_cols].copy()

    # Encode VA
    df_feat["VA_Right_Score"] = df["VA Right"].map(va_mapping).fillna(5)
    df_feat["VA_Left_Score"]  = df["VA Left"].map(va_mapping).fillna(5)
    df_feat["VA_Avg_Score"]   = (df_feat["VA_Right_Score"] + df_feat["VA_Left_Score"]) / 2
    df_feat["VA_Diff"]        = abs(df_feat["VA_Right_Score"] - df_feat["VA_Left_Score"])

    # Derived features
    df_feat["MAP"] = df["Diastolic BP"] + (df["Systolic BP"] - df["Diastolic BP"]) / 3
    df_feat["Pulse_Pressure"] = df["Systolic BP"] - df["Diastolic BP"]
    df_feat["HbA1c_Duration"] = df["HbA1c %"] * df["Diabetes Duration"]  # interaction
    df_feat["Age_Duration_Ratio"] = df["Age"] / (df["Diabetes Duration"] + 1)

    # Encode categoricals
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_feat[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))
        encoders[col] = le

    # Symptom presence (binary)
    df_feat["Has_Symptom"] = (~df["Symptom"].isna()).astype(int)

    feature_names = list(df_feat.columns)

    # Fill remaining NaN
    df_feat = df_feat.fillna(df_feat.median())

    print(f"[✓] Engineered {len(feature_names)} features")
    return df_feat, y, feature_names, encoders


# ─────────────────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────────────────
def train_models(X_train, y_train, X_test, y_test, class_weights_dict):
    """Train multiple models and return the best one."""

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # ── Model 1: Random Forest ──────────────────────────
    print("\n━━━ Training Random Forest ━━━")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train)
    rf_pred = rf.predict(X_test_sc)
    rf_acc  = accuracy_score(y_test, rf_pred)
    rf_f1   = f1_score(y_test, rf_pred, average="weighted")
    rf_kappa = cohen_kappa_score(y_test, rf_pred, weights="quadratic")
    results["Random Forest"] = {
        "model": rf, "acc": rf_acc, "f1": rf_f1, "kappa": rf_kappa,
        "pred": rf_pred, "prob": rf.predict_proba(X_test_sc)
    }
    print(f"  Accuracy: {rf_acc:.4f}  |  F1: {rf_f1:.4f}  |  QWK: {rf_kappa:.4f}")

    # ── Model 2: Gradient Boosting ──────────────────────
    print("\n━━━ Training Gradient Boosting ━━━")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=4,
        random_state=SEED,
    )
    gb.fit(X_train_sc, y_train)
    gb_pred = gb.predict(X_test_sc)
    gb_acc  = accuracy_score(y_test, gb_pred)
    gb_f1   = f1_score(y_test, gb_pred, average="weighted")
    gb_kappa = cohen_kappa_score(y_test, gb_pred, weights="quadratic")
    results["Gradient Boosting"] = {
        "model": gb, "acc": gb_acc, "f1": gb_f1, "kappa": gb_kappa,
        "pred": gb_pred, "prob": gb.predict_proba(X_test_sc)
    }
    print(f"  Accuracy: {gb_acc:.4f}  |  F1: {gb_f1:.4f}  |  QWK: {gb_kappa:.4f}")

    # ── Model 3: HistGradientBoosting ───────────────────
    print("\n━━━ Training HistGradientBoosting ━━━")
    hgb = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
    )
    hgb.fit(X_train_sc, y_train)
    hgb_pred = hgb.predict(X_test_sc)
    hgb_acc  = accuracy_score(y_test, hgb_pred)
    hgb_f1   = f1_score(y_test, hgb_pred, average="weighted")
    hgb_kappa = cohen_kappa_score(y_test, hgb_pred, weights="quadratic")
    results["HistGradientBoosting"] = {
        "model": hgb, "acc": hgb_acc, "f1": hgb_f1, "kappa": hgb_kappa,
        "pred": hgb_pred, "prob": hgb.predict_proba(X_test_sc)
    }
    print(f"  Accuracy: {hgb_acc:.4f}  |  F1: {hgb_f1:.4f}  |  QWK: {hgb_kappa:.4f}")

    # ── Model 4: Ensemble (Voting) ─────────────────────
    print("\n━━━ Training Ensemble (Soft Voting) ━━━")
    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf), ("gb", gb), ("hgb", hgb)
        ],
        voting="soft",
    )
    ensemble.fit(X_train_sc, y_train)
    ens_pred = ensemble.predict(X_test_sc)
    ens_acc  = accuracy_score(y_test, ens_pred)
    ens_f1   = f1_score(y_test, ens_pred, average="weighted")
    ens_kappa = cohen_kappa_score(y_test, ens_pred, weights="quadratic")
    results["Ensemble"] = {
        "model": ensemble, "acc": ens_acc, "f1": ens_f1, "kappa": ens_kappa,
        "pred": ens_pred, "prob": ensemble.predict_proba(X_test_sc)
    }
    print(f"  Accuracy: {ens_acc:.4f}  |  F1: {ens_f1:.4f}  |  QWK: {ens_kappa:.4f}")

    return results, scaler


# ─────────────────────────────────────────────────────────
# 4. CROSS VALIDATION
# ─────────────────────────────────────────────────────────
def cross_validate_best(best_model, X, y, scaler):
    """Run 5-fold stratified cross-validation on the best model."""
    X_sc = scaler.transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(best_model, X_sc, y, cv=cv, scoring="accuracy")
    print(f"\n[Cross-Validation] 5-Fold Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Per-fold: {[round(s, 4) for s in scores]}")
    return scores


# ─────────────────────────────────────────────────────────
# 5. PLOTTING
# ─────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {filename}")


def plot_feature_importance(model, feature_names, filename):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return
    idx = np.argsort(imp)[::-1][:15]  # top 15
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(idx)))
    ax.barh(range(len(idx)), imp[idx][::-1], color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 15 Feature Importances (Best Model)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {filename}")


def plot_model_comparison(results, filename):
    names = list(results.keys())
    metrics = {
        "Accuracy": [results[n]["acc"] for n in names],
        "F1 Score": [results[n]["f1"] for n in names],
        "QW Kappa": [results[n]["kappa"] for n in names],
    }
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, vals, width, label=metric, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {filename}")


def plot_grade_distribution(y, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = pd.Series(y).value_counts().sort_index()
    colors = ["#22c55e", "#84cc16", "#f59e0b", "#ef4444", "#dc2626"]
    bars = ax.bar(range(5), counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
    ax.set_ylabel("Number of Patients")
    ax.set_title("DR Grade Distribution (300 Patients)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {filename}")


def plot_correlation_heatmap(X, feature_names, filename):
    fig, ax = plt.subplots(figsize=(14, 11))
    corr = pd.DataFrame(X, columns=feature_names).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {filename}")


# ─────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("  DIABETIC RETINOPATHY — CLINICAL PREDICTION PIPELINE")
    print("=" * 65)

    # 1. Load
    df = load_dataset(DATASET_PATH)

    # 2. Engineer features
    X, y, feature_names, encoders = engineer_features(df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"\n[✓] Train/Test split: {len(X_train)} train / {len(X_test)} test")

    # 4. Class weights
    cw = compute_class_weight("balanced", classes=np.arange(5), y=y_train)
    class_weights = dict(enumerate(cw))
    print(f"[✓] Class weights: {class_weights}")

    # 5. Train models
    results, scaler = train_models(X_train, y_train, X_test, y_test, class_weights)

    # 6. Select best model
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]
    print(f"\n{'='*65}")
    print(f"  🏆 BEST MODEL: {best_name}")
    print(f"     Accuracy:   {best['acc']:.4f}")
    print(f"     F1 Score:   {best['f1']:.4f}")
    print(f"     QW Kappa:   {best['kappa']:.4f}")
    print(f"{'='*65}")

    # 7. Classification report
    print(f"\n{' CLASSIFICATION REPORT ':=^65}")
    report_str = classification_report(y_test, best["pred"], target_names=CLASS_NAMES)
    print(report_str)

    # 8. Cross-validation
    cv_scores = cross_validate_best(best["model"], X, y, scaler)

    # 9. Plots
    print("\n[📊] Generating plots...")
    plot_grade_distribution(y, "grade_distribution.png")
    plot_confusion_matrix(y_test, best["pred"],
                         f"Confusion Matrix — {best_name}", "confusion_matrix.png")
    plot_feature_importance(best["model"], feature_names, "feature_importance.png")
    plot_model_comparison(results, "model_comparison.png")
    plot_correlation_heatmap(X.values, feature_names, "correlation_heatmap.png")

    # 10. Save model + artifacts
    print("\n[💾] Saving model artifacts...")
    model_path = MODEL_DIR / "dr_clinical_model.joblib"
    scaler_path = MODEL_DIR / "dr_scaler.joblib"
    meta_path = MODEL_DIR / "dr_model_metadata.json"

    joblib.dump(best["model"], model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_name": best_name,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "class_names": CLASS_NAMES,
        "test_accuracy": round(best["acc"], 4),
        "test_f1": round(best["f1"], 4),
        "test_qw_kappa": round(best["kappa"], 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "dataset_total": len(df),
        "trained_at": datetime.now().isoformat(),
        "all_results": {
            name: {
                "accuracy": round(r["acc"], 4),
                "f1": round(r["f1"], 4),
                "qw_kappa": round(r["kappa"], 4),
            }
            for name, r in results.items()
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [Saved] {model_path}")
    print(f"  [Saved] {scaler_path}")
    print(f"  [Saved] {meta_path}")

    # 11. Save classification report to file
    report_path = OUTPUT_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Best Model: {best_name}\n")
        f.write(f"Test Accuracy: {best['acc']:.4f}\n")
        f.write(f"Test F1 Score: {best['f1']:.4f}\n")
        f.write(f"Quadratic Weighted Kappa: {best['kappa']:.4f}\n")
        f.write(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        f.write(report_str)
    print(f"  [Saved] {report_path}")

    print("\n" + "=" * 65)
    print("  ✅ PIPELINE COMPLETE!")
    print("=" * 65)

    return metadata


if __name__ == "__main__":
    meta = main()
