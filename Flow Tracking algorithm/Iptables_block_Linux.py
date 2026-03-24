import subprocess

def block_ip(ip_address):
    subprocess.run([
        "iptables", "-A", "INPUT",
        "-s", ip_address,
        "-j", "DROP"
    ], check=True)
    print(f"[BLOCKED] {ip_address} via iptables")