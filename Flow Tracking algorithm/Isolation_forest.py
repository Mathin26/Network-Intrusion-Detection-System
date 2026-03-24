from sklearn.ensemble import IsolationForest

# Train on NORMAL traffic only
iso = IsolationForest(contamination=0.01, n_jobs=-1)
iso.fit(X_train[y_train == 0])  # only normal samples

# At inference: -1 = anomaly, +1 = normal
anomaly_score = iso.predict(features.reshape(1,-1))[0]
rf_pred       = model.predict(features.reshape(1,-1))[0]

if rf_pred == 1:
    alert("HIGH confidence known attack", src_ip)
elif anomaly_score == -1:
    alert("LOW confidence anomaly — investigate", src_ip)
```

---

## The Realistic Deployment Stack

For a small office/lab network (the most practical next step for this project):
```
NIC 1 (monitor) ──→ Python process
                     ├── Scapy sniffer thread
                     ├── Flow tracker (dict + timeout GC)
                     ├── Feature extractor
                     ├── RF model inference
                     └── Alert dispatcher
                           ├── iptables (block)
                           ├── SQLite log (store)
                           └── Email/Slack (notify)