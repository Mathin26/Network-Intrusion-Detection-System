# Zeek writes conn.log in TSV format — tail it and classify
import tailer   # pip install tailer

for line in tailer.follow(open("/usr/local/zeek/logs/current/conn.log")):
    if line.startswith("#"): continue
    fields   = line.split("\t")
    features = parse_zeek_conn_log(fields)   # map to KDD features
    pred     = model.predict(scaler.transform([features]))[0]
    if pred == 1:
        trigger_alert(fields[2], fields[3])   # src/dst IP