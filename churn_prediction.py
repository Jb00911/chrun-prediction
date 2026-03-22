"""
========================================================
  Customer Churn Prediction — Complete ML Pipeline
  Author : Jibran Shahid
  Dataset: IBM Telco Customer Churn (Kaggle)
  Run    : python churn_prediction.py
========================================================

SETUP (run once in CMD):
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost kagglehub

HOW TO GET THE DATASET:
    Option A (automatic) : script auto-downloads via kagglehub if installed
    Option B (manual)    : download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
                           place  WA_Fn-UseC_-Telco-Customer-Churn.csv  in the same folder as this script
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — saves PNGs, never opens GUI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.pipeline         import Pipeline
from sklearn.inspection       import permutation_importance

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] xgboost not installed — skipping XGBoost model. Run: pip install xgboost")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CSV_FILENAME  = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR    = "churn_output"           # all charts + report saved here
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5

# Seaborn theme
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {"No": "#4C9BE8", "Yes": "#E8604C"}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def banner(title: str) -> None:
    """Print a formatted section banner."""
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def save_figure(name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    banner("STEP 1 · Loading Data")

    # Try local file first
    if os.path.isfile(CSV_FILENAME):
        print(f"  Found local file: {CSV_FILENAME}")
        df = pd.read_csv(CSV_FILENAME)
        print(f"  Shape: {df.shape}")
        return df

    # Try kagglehub auto-download
    try:
        import kagglehub
        print("  Downloading dataset via kagglehub …")
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        # find the CSV inside the downloaded folder
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    csv_path = os.path.join(root, f)
                    df = pd.read_csv(csv_path)
                    print(f"  Downloaded → {csv_path}  Shape: {df.shape}")
                    return df
    except Exception as exc:
        pass

    # Neither worked
    print("\n  [ERROR] Dataset not found.")
    print(f"  Place '{CSV_FILENAME}' in the same folder as this script, then re-run.")
    print("  Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame) -> None:
    banner("STEP 2 · Exploratory Data Analysis")

    print(f"\n  Rows      : {df.shape[0]:,}")
    print(f"  Columns   : {df.shape[1]}")
    print(f"  Churn rate: {(df['Churn']=='Yes').mean()*100:.1f}%")
    print("\n  Missing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  None")
    print("\n  Data types:")
    print(df.dtypes.value_counts().to_string())

    # ── Chart 1: Churn Distribution ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Churn Overview", fontsize=14, fontweight="bold")

    counts = df["Churn"].value_counts()
    axes[0].pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=[PALETTE["No"], PALETTE["Yes"]],
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    axes[0].set_title("Overall churn split")

    sns.countplot(data=df, x="Churn", palette=PALETTE, ax=axes[1])
    axes[1].set_title("Count by churn label")
    axes[1].bar_label(axes[1].containers[0])
    save_figure("01_churn_distribution.png")

    # ── Chart 2: Numeric distributions by churn ──────────────────────────────
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Numeric Features vs Churn", fontsize=14, fontweight="bold")
    for ax, col in zip(axes, num_cols):
        for label, color in PALETTE.items():
            subset = df[df["Churn"] == label][col].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(col)
        ax.legend()
    save_figure("02_numeric_distributions.png")

    # ── Chart 3: Contract & payment type ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Categorical Features vs Churn", fontsize=14, fontweight="bold")
    for ax, col in zip(axes, ["Contract", "PaymentMethod"]):
        ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
        ct.plot(kind="bar", ax=ax, color=[PALETTE["No"], PALETTE["Yes"]],
                edgecolor="white", rot=15)
        ax.set_title(f"Churn % by {col}")
        ax.set_ylabel("Churn rate (%)")
        ax.legend(title="Churn")
    plt.tight_layout()
    save_figure("03_categorical_churn.png")

    # ── Chart 4: Correlation heatmap (numeric only) ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    df_num = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df_num["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
    corr = df_num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix")
    save_figure("04_correlation_heatmap.png")

    print("  EDA charts saved to:", OUTPUT_DIR)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    banner("STEP 3 · Preprocessing")

    df = df.copy()

    # Drop ID column
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Fix TotalCharges (whitespace → NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Feature engineering
    df["AvgMonthlySpend"]     = df["TotalCharges"] / (df["tenure"] + 1)
    df["IsNewCustomer"]       = (df["tenure"] <= 3).astype(int)
    df["IsLongTermCustomer"]  = (df["tenure"] >= 48).astype(int)

    # One-hot encode all remaining object columns
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"  Final shape after preprocessing: {df.shape}")
    print(f"  Churn balance: {df['Churn'].value_counts().to_dict()}")

    # Verify — no more string columns
    remaining_obj = df.select_dtypes(include="object").columns.tolist()
    if remaining_obj:
        print(f"  [WARN] Non-numeric columns still present: {remaining_obj}")
    else:
        print("  All columns are numeric ✓")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def build_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(
                                   n_estimators=200, max_depth=10,
                                   min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(
                                   n_estimators=150, learning_rate=0.08,
                                   max_depth=4, random_state=RANDOM_STATE),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE,
            verbosity=0, use_label_encoder=False
        )
    return models


def train_and_evaluate(X, y):
    banner("STEP 4 · Training & Evaluation")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models   = build_models()
    results  = {}
    cv       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n  Training: {name} …")

        # Cross-validation AUC
        cv_aucs = cross_val_score(model, X_train_sc, y_train,
                                  cv=cv, scoring="roc_auc", n_jobs=-1)

        # Fit on full train set
        model.fit(X_train_sc, y_train)
        preds      = model.predict(X_test_sc)
        proba      = model.predict_proba(X_test_sc)[:, 1]

        acc  = accuracy_score(y_test, preds)
        auc  = roc_auc_score(y_test, proba)
        f1   = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec  = recall_score(y_test, preds)

        results[name] = {
            "model": model, "scaler": scaler,
            "preds": preds, "proba": proba,
            "accuracy": acc, "auc": auc, "f1": f1,
            "precision": prec, "recall": rec,
            "cv_auc_mean": cv_aucs.mean(), "cv_auc_std": cv_aucs.std(),
            "X_test": X_test_sc, "y_test": y_test,
            "X_train": X_train_sc, "y_train": y_train,
        }

        print(f"    Accuracy  : {acc*100:.2f}%")
        print(f"    AUC       : {auc:.4f}")
        print(f"    F1 Score  : {f1:.4f}")
        print(f"    CV AUC    : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    return results, X.columns.tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — VISUALISE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def visualise_results(results: dict, feature_names: list) -> str:
    banner("STEP 5 · Visualising Results")

    # ── Chart 5: Model comparison bar chart ──────────────────────────────────
    names    = list(results.keys())
    metrics  = ["accuracy", "auc", "f1", "precision", "recall"]
    metric_labels = ["Accuracy", "AUC-ROC", "F1 Score", "Precision", "Recall"]

    fig, ax = plt.subplots(figsize=(13, 5))
    x    = np.arange(len(names))
    w    = 0.15
    colors = ["#4C9BE8", "#E8604C", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, (m, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + i * w, vals, w, label=label, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure("05_model_comparison.png")

    # ── Chart 6: ROC curves ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["proba"])
        ax.plot(fpr, tpr, lw=2, label=f"{name}  (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — All Models")
    ax.legend(loc="lower right")
    save_figure("06_roc_curves.png")

    # ── Chart 7: Confusion matrices ───────────────────────────────────────────
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_test"], res["preds"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=11)
    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure("07_confusion_matrices.png")

    # ── Chart 8: Feature importance (best model) ──────────────────────────────
    best_name = max(results, key=lambda n: results[n]["auc"])
    best      = results[best_name]
    model     = best["model"]

    has_native = hasattr(model, "feature_importances_")
    if has_native:
        importances = model.feature_importances_
    else:
        # Use permutation importance for models without native support
        perm = permutation_importance(model, best["X_test"], best["y_test"],
                                      n_repeats=10, random_state=RANDOM_STATE)
        importances = perm.importances_mean

    feat_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=feat_df, x="importance", y="feature",
                palette="Blues_r", ax=ax)
    ax.set_title(f"Top 20 Feature Importances — {best_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance score")
    plt.tight_layout()
    save_figure("08_feature_importance.png")

    return best_name


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict, best_name: str) -> None:
    banner("STEP 6 · Final Summary Report")

    best = results[best_name]

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │         CUSTOMER CHURN PREDICTION RESULTS       │
  ├─────────────────────────────────────────────────┤
  │  Best Model  : {best_name:<32}│
  │  Accuracy    : {best['accuracy']*100:>6.2f}%                          │
  │  AUC-ROC     : {best['auc']:>6.4f}                           │
  │  F1 Score    : {best['f1']:>6.4f}                           │
  │  Precision   : {best['precision']:>6.4f}                           │
  │  Recall      : {best['recall']:>6.4f}                           │
  │  CV AUC      : {best['cv_auc_mean']:.4f} ± {best['cv_auc_std']:.4f}                  │
  └─────────────────────────────────────────────────┘
""")

    print("  All Models:\n")
    print(f"  {'Model':<25} {'Accuracy':>9} {'AUC':>8} {'F1':>8}")
    print("  " + "-" * 55)
    for name, res in sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True):
        marker = " ◄ best" if name == best_name else ""
        print(f"  {name:<25} {res['accuracy']*100:>8.2f}% {res['auc']:>8.4f} {res['f1']:>8.4f}{marker}")

    print(f"\n  Detailed classification report — {best_name}:")
    print(classification_report(best["y_test"], best["preds"],
                                target_names=["No Churn", "Churn"]))

    print(f"\n  Output files saved to: ./{OUTPUT_DIR}/")
    print("  ├── 01_churn_distribution.png")
    print("  ├── 02_numeric_distributions.png")
    print("  ├── 03_categorical_churn.png")
    print("  ├── 04_correlation_heatmap.png")
    print("  ├── 05_model_comparison.png")
    print("  ├── 06_roc_curves.png")
    print("  ├── 07_confusion_matrices.png")
    print("  └── 08_feature_importance.png")
    print("\n  Add these charts to your GitHub README and portfolio. Done!\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  CUSTOMER CHURN PREDICTION — ML PIPELINE")
    print("  Jibran Shahid · Data Science Portfolio")
    print("═" * 60)

    ensure_output_dir()

    df            = load_data()
    run_eda(df)
    X, y          = preprocess(df)
    results, cols = train_and_evaluate(X, y)
    best_name     = visualise_results(results, cols)
    print_summary(results, best_name)


if __name__ == "__main__":
    main()
