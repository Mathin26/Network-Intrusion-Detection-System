# 🛡️ Network Intrusion Detection System (NIDS)
### Machine Learning · Random Forest · KDD Cup '99 · Scapy Live Detection

---

## 📁 Project Structure

```
nids_project/
├── nids_train.py      ← Main script: load, preprocess, train, evaluate
├── nids_live.py       ← Live packet sniffing + real-time classification
├── nids_unsw.py       ← Alternate script for UNSW-NB15 dataset
├── requirements.txt   ← Python dependencies
└── README.md          ← You are here
```

After training, three artifact files are also created:
```
nids_model.joblib      ← Trained Random Forest
nids_scaler.joblib     ← StandardScaler
nids_encoders.joblib   ← LabelEncoders for categorical features
nids_evaluation.png    ← Confusion matrix + feature importance plots
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (KDD Cup '99 — no download needed)
```bash
python nids_train.py
```
The script automatically downloads the KDD Cup '99 dataset via `sklearn.datasets.fetch_kddcup99`.

### 3. Live Detection (Scapy)
```bash
# Linux/macOS — requires root
sudo python nids_live.py

# Specify interface
sudo python nids_live.py --iface eth0 --count 200

# Simulation mode (no root needed, great for demos)
python nids_live.py --sim
```

### 4. Use UNSW-NB15 Instead
1. Download from: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
2. Place `UNSW_NB15_training-set.csv` in this folder
3. Run: `python nids_unsw.py`

---

## 🔬 How It Works

### The Dataset — KDD Cup '99
- ~5 million connection records labeled as **normal** or one of 22 attack types
- 41 features per record (network stats, protocol info, connection behavior)
- The script uses the built-in 10% subset (~490k rows), subsampled to 50k for speed

### Preprocessing
| Step | Detail |
|------|--------|
| Binary label | `normal.` → 0, all attacks → 1 |
| Categorical encoding | `protocol_type`, `service`, `flag` → `LabelEncoder` |
| Feature scaling | `StandardScaler` → zero mean, unit variance |
| Train/test split | 80% / 20%, stratified |

### Model — Random Forest
| Parameter | Value |
|-----------|-------|
| `n_estimators` | 100 trees |
| `max_depth` | 20 |
| `class_weight` | balanced (handles imbalanced data) |
| `n_jobs` | -1 (all CPU cores) |

### Evaluation Outputs
- **Accuracy** — overall correct classifications
- **ROC-AUC** — discrimination ability (1.0 = perfect)
- **Confusion Matrix** — TP, TN, FP, FN breakdown
- **Classification Report** — precision, recall, F1 per class
- **Feature Importances** — top 15 most predictive features
- **Score Distribution** — probability histogram for Normal vs Attack

### Live Detection — Scapy
The `nids_live.py` script:
1. Loads the pre-trained model + scaler
2. Opens a raw socket on a network interface
3. For each captured packet, extracts 41 KDD-like features
4. Feeds the feature vector to the Random Forest
5. Prints a colored real-time verdict: 🟢 Normal / 🔴 ATTACK

> **Note:** Single-packet feature extraction is a simplified approximation.
> Production NIDS (like Suricata/Snort) maintain per-flow statistics.
> For a production system, track 5-tuple flows and compute statistics over
> sliding time windows before classifying.

---

## 📊 Expected Results

On the KDD Cup '99 10% dataset with 50k sample:

| Metric | Value |
|--------|-------|
| Accuracy | ~99.5% |
| ROC-AUC | ~0.999 |
| Attack Recall | ~99% |
| False Alarm Rate | <1% |

> KDD '99 is a somewhat "easy" dataset for ML — very high accuracy is expected.
> UNSW-NB15 is more realistic/challenging (expect ~92–96% accuracy).

---

## 🔧 Extending the Project

### Try Multi-class Classification
Change the label to use attack categories instead of binary:
```python
df["label"] = df["label"].apply(lambda x: "normal" if x == "normal." else x)
```

### Try Other Models
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier  # pip install xgboost
```

### Add Real Flow Statistics (Scapy)
Track per-flow (src_ip, dst_ip, src_port, dst_port, proto) stats using a dict:
```python
flow_stats = {}   # key = 5-tuple, value = running counters
```

---

## ⚠️ Legal & Ethical Notice

This tool is for **educational and authorized testing only**.  
Never sniff network traffic without explicit permission from the network owner.  
Running packet capture on networks you do not own may violate laws.

---

*Built with Python · Scikit-Learn · Pandas · Scapy*


pip install -r requirements.txt
python nids_train.py        # trains the model, saves plots
python nids_live.py --sim   # demo live detection (no root needed)