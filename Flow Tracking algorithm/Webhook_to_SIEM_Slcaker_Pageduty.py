import requests, json

def send_alert_webhook(src_ip, confidence, attack_type="unknown"):
    payload = {
        "alert": "NIDS intrusion detected",
        "src_ip": src_ip,
        "confidence": f"{confidence:.1%}",
        "attack_type": attack_type,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    requests.post("https://your-siem/api/alerts", json=payload,
                  headers={"Authorization": "Bearer YOUR_TOKEN"})