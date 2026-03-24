"""
╔══════════════════════════════════════════════════════════════╗
║     NIDS  ·  LIVE DETECTION ENGINE  (Scapy)                  ║
║     Sniffs live packets → feeds pre-trained RF model         ║
╚══════════════════════════════════════════════════════════════╝

Prerequisites:
    pip install scapy joblib scikit-learn

Run (requires root/admin for raw-socket access):
    sudo python nids_live.py
    sudo python nids_live.py --iface eth0 --count 100

⚠  Train the model first:  python nids_train.py
"""

import argparse
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

# ── Scapy ──────────────────────────────────────────────────
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False
    print("[!] Scapy not installed. Running in SIMULATION mode.")
    print("    Install: pip install scapy\n")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH   = "nids_model.joblib"
SCALER_PATH  = "nids_scaler.joblib"

# KDD-style feature order (41 features used during training)
KDD_FEATURES = [
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

PROTO_MAP = {"tcp": 1, "udp": 2, "icmp": 0}  # simplified encoding
SERVICE_MAP = {"http": 20, "ftp": 10, "smtp": 35, "other": 0}
FLAG_MAP    = {"SF": 10, "S0": 5, "REJ": 3, "RSTO": 4, "other": 0}


# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
def load_model():
    try:
        clf    = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"[✔] Model loaded  → {MODEL_PATH}")
        print(f"[✔] Scaler loaded → {SCALER_PATH}")
        return clf, scaler
    except FileNotFoundError:
        print("[✘] Model not found. Run 'python nids_train.py' first.")
        sys.exit(1)


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# Extract KDD-like features from a raw Scapy packet.
# Real-world NIDS would track flow-level stats; this is
# a single-packet approximation for demonstration.
# ─────────────────────────────────────────────
def packet_to_features(pkt):
    """Convert a Scapy packet to a KDD-style feature vector (41 dims)."""
    feats = {k: 0.0 for k in KDD_FEATURES}

    if not pkt.haslayer(IP):
        return None

    ip = pkt[IP]
    feats["src_bytes"] = len(ip)
    feats["dst_bytes"] = 0
    feats["duration"]  = 0

    # Protocol
    if pkt.haslayer(TCP):
        feats["protocol_type"] = PROTO_MAP["tcp"]
        tcp = pkt[TCP]
        feats["src_bytes"]  = len(tcp.payload) if tcp.payload else 0
        feats["count"]      = tcp.dport
        feats["srv_count"]  = tcp.sport
        flags = str(tcp.flags)
        if "S" in flags and "A" in flags:
            feats["flag"] = FLAG_MAP["SF"]
        elif "S" in flags:
            feats["flag"] = FLAG_MAP["S0"]
        elif "R" in flags:
            feats["flag"] = FLAG_MAP["RSTO"]
        # Heuristic: many connections to same port → suspicious
        feats["same_srv_rate"]    = 1.0 if tcp.dport in (80, 443, 22, 21) else 0.3
        feats["dst_host_count"]   = 255
        feats["dst_host_srv_count"] = 255

    elif pkt.haslayer(UDP):
        feats["protocol_type"] = PROTO_MAP["udp"]
        udp = pkt[UDP]
        feats["src_bytes"] = len(udp.payload) if udp.payload else 0
        feats["count"]     = udp.dport

    elif pkt.haslayer(ICMP):
        feats["protocol_type"] = PROTO_MAP["icmp"]
        feats["count"]         = 10   # ICMP flood heuristic
        feats["serror_rate"]   = 0.9

    # Service guess from port
    if pkt.haslayer(TCP) or pkt.haslayer(UDP):
        dport = pkt[TCP].dport if pkt.haslayer(TCP) else pkt[UDP].dport
        feats["service"] = (
            SERVICE_MAP["http"]  if dport in (80, 8080, 443) else
            SERVICE_MAP["ftp"]   if dport == 21 else
            SERVICE_MAP["smtp"]  if dport == 25 else
            SERVICE_MAP["other"]
        )

    return np.array([feats[f] for f in KDD_FEATURES], dtype=float)


# ─────────────────────────────────────────────
# CLASSIFY
# ─────────────────────────────────────────────
def classify_packet(pkt, clf, scaler, verbose=True):
    vec = packet_to_features(pkt)
    if vec is None:
        return

    vec_scaled = scaler.transform(vec.reshape(1, -1))
    pred       = clf.predict(vec_scaled)[0]
    prob       = clf.predict_proba(vec_scaled)[0][1]

    if not pkt.haslayer(IP):
        return

    src = pkt[IP].src
    dst = pkt[IP].dst
    proto = (
        "TCP"  if pkt.haslayer(TCP)  else
        "UDP"  if pkt.haslayer(UDP)  else
        "ICMP" if pkt.haslayer(ICMP) else "???"
    )

    if pred == 1:
        tag    = "🔴 ATTACK"
        colour = "\033[91m"   # red
    else:
        tag    = "🟢 Normal"
        colour = "\033[92m"   # green

    reset = "\033[0m"
    ts    = time.strftime("%H:%M:%S")

    print(f"  [{ts}] {colour}{tag}{reset}  "
          f"({prob*100:5.1f}%)  "
          f"{proto:4s}  {src:>15} → {dst}")


# ─────────────────────────────────────────────
# SIMULATION MODE (no Scapy / no root)
# ─────────────────────────────────────────────
def simulate(clf, scaler, n=30):
    print("\n[SIM] Generating synthetic packets …\n")
    rng = np.random.default_rng(42)

    for i in range(n):
        is_attack = rng.random() > 0.6
        if is_attack:
            # Mimic attack features
            vec = rng.uniform(0, 1, len(KDD_FEATURES))
            vec[0]  = 0          # duration=0
            vec[4]  = rng.integers(1000, 65535)   # large src_bytes
            vec[23] = rng.uniform(0.8, 1.0)        # high serror_rate
        else:
            vec = rng.uniform(0, 0.3, len(KDD_FEATURES))

        vec_scaled = scaler.transform(vec.reshape(1, -1))
        pred  = clf.predict(vec_scaled)[0]
        prob  = clf.predict_proba(vec_scaled)[0][1]
        tag   = "🔴 ATTACK" if pred == 1 else "🟢 Normal"
        src   = f"192.168.{rng.integers(0,255)}.{rng.integers(1,254)}"
        dst   = f"10.0.0.{rng.integers(1,254)}"
        ts    = time.strftime("%H:%M:%S")
        colour = "\033[91m" if pred == 1 else "\033[92m"
        print(f"  [{ts}] {colour}{tag}\033[0m  ({prob*100:5.1f}%)  PKT  {src:>15} → {dst}")
        time.sleep(0.15)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NIDS Live Detection")
    parser.add_argument("--iface", default=None,
                        help="Network interface (e.g. eth0, en0)")
    parser.add_argument("--count", type=int, default=0,
                        help="Packets to capture (0=infinite)")
    parser.add_argument("--filter", default="ip",
                        help="BPF filter (default: 'ip')")
    parser.add_argument("--sim", action="store_true",
                        help="Force simulation mode")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║     NIDS  ·  LIVE DETECTION ENGINE  v1.0                     ║
╚══════════════════════════════════════════════════════════════╝
""")
    clf, scaler = load_model()

    if args.sim or not SCAPY_OK:
        simulate(clf, scaler)
        return

    # List interfaces if none specified
    if args.iface is None:
        ifaces = get_if_list()
        print(f"[i] Available interfaces: {ifaces}")
        args.iface = ifaces[0]
        print(f"[i] Using interface: {args.iface}\n")

    print(f"[►] Sniffing on {args.iface}  |  filter='{args.filter}'  "
          f"|  count={'∞' if args.count==0 else args.count}")
    print("    Press Ctrl+C to stop.\n")
    print(f"  {'TIMESTAMP':9s} {'STATUS':16s} {'CONF':7s} {'PROTO':4s} {'SRC':>15} → DST")
    print("  " + "─"*65)

    sniff(
        iface=args.iface,
        filter=args.filter,
        count=args.count,
        prn=lambda pkt: classify_packet(pkt, clf, scaler),
        store=False
    )


if __name__ == "__main__":
    main()
