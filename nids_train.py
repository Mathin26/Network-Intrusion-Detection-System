"""
╔══════════════════════════════════════════════════════════════╗
║       NETWORK INTRUSION DETECTION SYSTEM (NIDS)              ║
║       Train · Evaluate · Detect  |  KDD Cup '99 Dataset      ║
╚══════════════════════════════════════════════════════════════╝

STEP 1 ─ Install dependencies:
    pip install pandas scikit-learn matplotlib seaborn joblib

STEP 2 ─ Download dataset (choose one):
    • KDD Cup '99 (built-in via sklearn):  no download needed!
    • UNSW-NB15: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

Run:
    python nids_train.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import time

from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
N_ESTIMATORS   = 100
MAX_DEPTH      = 20
SAMPLE_SIZE    = 50_000   # Subsample for speed; set None for full dataset
MODEL_PATH     = "nids_model.joblib"
SCALER_PATH    = "nids_scaler.joblib"
ENCODER_PATH   = "nids_encoders.joblib"


# ─────────────────────────────────────────────
# 1. LOAD & PREVIEW DATA
# ─────────────────────────────────────────────
def load_kdd_data(sample_size=SAMPLE_SIZE):
    print("\n" + "═"*60)
    print("  [1/5]  Loading KDD Cup '99 Dataset …")
    print("═"*60)

    kdd = fetch_kddcup99(subset=None, shuffle=True, random_state=RANDOM_STATE,
                         percent10=True)      # 10% version ≈ 490k rows

    col_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login",
        "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
        "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]

    df = pd.DataFrame(kdd.data, columns=col_names)
    df["label"] = kdd.target

    # Decode bytes → strings
    for col in df.select_dtypes(include=object).columns:
        df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE)

    print(f"  ✔  Rows loaded  : {len(df):,}")
    print(f"  ✔  Features     : {df.shape[1] - 1}")
    print(f"  ✔  Attack types : {df['label'].nunique()}")
    print(f"\n  Top labels:\n{df['label'].value_counts().head(8).to_string()}")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    print("\n" + "═"*60)
    print("  [2/5]  Preprocessing …")
    print("═"*60)

    df = df.copy()

    # Binary label: 'normal.' → 0, everything else → 1
    df["binary_label"] = (df["label"] != "normal.").astype(int)
    class_counts = df["binary_label"].value_counts()
    print(f"  Normal  (0) : {class_counts.get(0, 0):,}")
    print(f"  Attack  (1) : {class_counts.get(1, 0):,}")

    # Encode categorical columns
    cat_cols = ["protocol_type", "service", "flag"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  ✔  Encoded '{col}' → {le.classes_.tolist()}")

    feature_cols = [c for c in df.columns if c not in ("label", "binary_label")]
    X = df[feature_cols].astype(float)
    y = df["binary_label"]

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    print(f"\n  Feature matrix shape : {X_scaled.shape}")
    return X_scaled, y, scaler, encoders, feature_cols


# ─────────────────────────────────────────────
# 3. TRAIN
# ─────────────────────────────────────────────
def train(X_train, y_train):
    print("\n" + "═"*60)
    print("  [3/5]  Training Random Forest …")
    print("═"*60)

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  ✔  Trees      : {N_ESTIMATORS}")
    print(f"  ✔  Max depth  : {MAX_DEPTH}")
    print(f"  ✔  Train time : {elapsed:.2f}s")
    return clf


# ─────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────
def evaluate(clf, X_test, y_test, feature_cols):
    print("\n" + "═"*60)
    print("  [4/5]  Evaluation …")
    print("═"*60)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Accuracy  : {acc*100:.2f}%")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives  (Normal→Normal) : {tn:,}")
    print(f"    False Positives (Normal→Attack) : {fp:,}  ← False Alarms")
    print(f"    False Negatives (Attack→Normal) : {fn:,}  ← Missed Attacks!")
    print(f"    True Positives  (Attack→Attack) : {tp:,}  ← Caught!")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Attack'])}")

    # ── PLOTS ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")

    # -- Confusion Matrix Heatmap
    ax = axes[0]
    ax.set_facecolor("#0d1117")
    labels = np.array([[f"TN\n{tn:,}", f"FP\n{fp:,}"],
                        [f"FN\n{fn:,}", f"TP\n{tp:,}"]])
    sns.heatmap(cm, annot=labels, fmt="", cmap="RdYlGn", linewidths=2,
                linecolor="#0d1117", ax=ax, cbar=False,
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"])
    ax.set_title("Confusion Matrix", color="#e6edf3", fontsize=14, pad=12)
    ax.set_xlabel("Predicted", color="#8b949e")
    ax.set_ylabel("Actual",    color="#8b949e")
    ax.tick_params(colors="#8b949e")

    # -- Feature Importance (top 15)
    ax = axes[1]
    ax.set_facecolor("#0d1117")
    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    top15 = importances.nlargest(15).sort_values()
    colors_bar = ["#238636" if v > top15.median() else "#388bfd" for v in top15]
    top15.plot(kind="barh", ax=ax, color=colors_bar, edgecolor="none")
    ax.set_title("Top 15 Feature Importances", color="#e6edf3", fontsize=14, pad=12)
    ax.set_xlabel("Importance Score", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.spines[["top","right","bottom","left"]].set_color("#30363d")
    ax.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # -- Attack vs Normal Dist
    ax = axes[2]
    ax.set_facecolor("#0d1117")
    normal_scores  = y_proba[y_test == 0]
    attack_scores  = y_proba[y_test == 1]
    ax.hist(normal_scores, bins=50, alpha=0.7, color="#388bfd", label="Normal",  density=True)
    ax.hist(attack_scores, bins=50, alpha=0.7, color="#f85149", label="Attack",  density=True)
    ax.axvline(x=0.5, color="#e3b341", linestyle="--", linewidth=1.5, label="Threshold 0.5")
    ax.set_title("Score Distribution", color="#e6edf3", fontsize=14, pad=12)
    ax.set_xlabel("Attack Probability", color="#8b949e")
    ax.set_ylabel("Density", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.legend(labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d")

    plt.suptitle("NIDS  ·  Random Forest Evaluation", color="#e6edf3",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("nids_evaluation.png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    print("\n  ✔  Plot saved → nids_evaluation.png")
    plt.show()

    return acc, auc


# ─────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────
def save_artifacts(clf, scaler, encoders):
    print("\n" + "═"*60)
    print("  [5/5]  Saving model artifacts …")
    print("═"*60)
    joblib.dump(clf,      MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    print(f"  ✔  Model   → {MODEL_PATH}")
    print(f"  ✔  Scaler  → {SCALER_PATH}")
    print(f"  ✔  Encoders→ {ENCODER_PATH}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║       NETWORK INTRUSION DETECTION SYSTEM  v1.0               ║
║       Powered by Random Forest + KDD Cup '99                 ║
╚══════════════════════════════════════════════════════════════╝
""")
    df                          = load_kdd_data()
    X, y, scaler, encoders, fc  = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    clf            = train(X_train, y_train)
    acc, auc       = evaluate(clf, X_test, y_test, fc)
    save_artifacts(clf, scaler, encoders)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  DONE  ✔                                                     ║
║  Accuracy : {acc*100:.2f}%                                        ║
║  ROC-AUC  : {auc:.4f}                                          ║
║  Next     : run  python nids_live.py  for live detection     ║
╚══════════════════════════════════════════════════════════════╝
""")
