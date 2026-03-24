"""The practical recommendation for a real deployment: use your Random Forest 
as the primary classifier, and add an Isolation Forest running in parallel 
as an anomaly detector. The RF catches known attacks with high confidence; 
the Isolation Forest raises lower-priority alerts for anything that looks
 statistically unusual even if it doesn't match a known pattern.
"""



import time
from collections import defaultdict
import joblib
import numpy as np

model  = joblib.load("nids_model.joblib")
scaler = joblib.load("nids_scaler.joblib")

flow_table = {}   # key=(src_ip,dst_ip,sport,dport,proto) → stats dict
FLOW_TIMEOUT = 120  # seconds

def get_or_create_flow(key, timestamp):
    if key not in flow_table:
        flow_table[key] = {
            "start_time": timestamp, "last_seen": timestamp,
            "src_bytes": 0, "dst_bytes": 0,
            "pkt_count": 0, "syn_count": 0, "rst_count": 0,
            "fin_count": 0, "serror_count": 0,
        }
    return flow_table[key]

def update_flow(pkt):
    from scapy.all import IP, TCP, UDP
    if not pkt.haslayer(IP): return

    ip    = pkt[IP]
    proto = "tcp" if pkt.haslayer(TCP) else "udp"
    sport = pkt[TCP].sport if pkt.haslayer(TCP) else 0
    dport = pkt[TCP].dport if pkt.haslayer(TCP) else 0
    key   = (ip.src, ip.dst, sport, dport, proto)

    flow  = get_or_create_flow(key, time.time())
    flow["src_bytes"]  += len(ip.payload)
    flow["pkt_count"]  += 1
    flow["last_seen"]   = time.time()

    if pkt.haslayer(TCP):
        flags = pkt[TCP].flags
        if flags & 0x02: flow["syn_count"]    += 1   # SYN
        if flags & 0x04: flow["rst_count"]    += 1   # RST
        if flags & 0x01: flow["fin_count"]    += 1   # FIN
        # Flow complete on FIN or RST — classify it
        if flags & 0x01 or flags & 0x04:
            classify_flow(key, flow)
            del flow_table[key]

def classify_flow(key, flow):
    duration = flow["last_seen"] - flow["start_time"]
    src_ip, dst_ip, sport, dport, proto = key

    # Build KDD-compatible feature vector
    features = np.array([
        duration,
        {"tcp":1,"udp":2,"icmp":0}.get(proto, 0),   # protocol_type
        get_service_code(dport),                      # service
        get_flag_code(flow),                          # flag
        flow["src_bytes"],
        flow["dst_bytes"],
        0,  # land (src==dst?)
        0,  # wrong_fragment
        0,  # urgent
        # ... fill remaining 32 features from flow counters ...
    ], dtype=float)

    scaled = scaler.transform(features.reshape(1, -1))
    pred   = model.predict(scaled)[0]
    prob   = model.predict_proba(scaled)[0][1]

    if pred == 1:
        trigger_alert(key, prob, flow)

def trigger_alert(key, confidence, flow):
    src_ip = key[0]
    print(f"[ATTACK] {src_ip} → conf={confidence:.2%}")
    # Block with iptables, send to SIEM, etc.