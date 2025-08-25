import os
import sys
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

################################################################################
# Constants
DATASET_FOLDER = "./../data/output"
FEATURES_RESULT_PATH = "./../data/features"

# Optional configuration
CLIENT_IP = None
DROP_ZERO_PAYLOAD = True
MAX_WORKERS = 4
################################################################################

TSHARK_FIELDS = [
    "-T", "fields",
    "-e", "frame.time_epoch",   # absolute epoch timestamp (float seconds)
    "-e", "ip.src",
    "-e", "ip.dst",
    "-e", "tcp.len",            # TCP payload length (bytes), may be empty
    "-e", "udp.length"          # UDP length incl. header; we'll convert to payload
]

# Display filter: only IP (v4/v6) and TCP/UDP; ignore ARP, etc.
DISPLAY_FILTER = "(ip or ipv6) and (tcp or udp)"

def filter_insufficient_samples(dataset_dir, min_count=2):
    """
    Remove websites (classes) that have fewer than `min_count` PCAP files.
    Assumes files are named like x_y.pcap (x = class/website id).
    """
    files = os.listdir(dataset_dir)
    # Count occurrences per class (before the underscore)
    counter = Counter(f.split("_")[0] for f in files if f.endswith(".pcap"))

    removed_classes = [cls for cls, count in counter.items() if count < min_count]
    if removed_classes:
        print(f"[INFO] Removing classes with <{min_count} samples: {removed_classes}")
        for f in files:
            cls = f.split("_")[0]
            if cls in removed_classes and f.endswith(".pcap"):
                os.remove(os.path.join(dataset_dir, f))


def ensure_tshark():
    if shutil.which("tshark") is None:
        print("[ERROR] tshark not found. Install with: brew install wireshark")
        sys.exit(1)

def parse_line_to_record(line: str):
    """
    Parses a TSV line exported by tshark with fields:
    frame.time_epoch, ip.src, ip.dst, tcp.len, udp.length
    Returns (time_epoch: float, src_ip: str, dst_ip: str, payload_len: int) or None
    """
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        return None
    try:
        t = float(parts[0]) if parts[0] else None
        src = parts[1] or None
        dst = parts[2] or None
        tcp_len_str = parts[3]
        udp_len_str = parts[4]

        # Determine payload length:
        # - Prefer TCP payload length when present.
        # - If UDP, udp.length is UDP total (header+payload). UDP header is 8 bytes.
        if tcp_len_str and tcp_len_str.isdigit():
            payload = int(tcp_len_str)
        elif udp_len_str and udp_len_str.isdigit():
            udp_total = int(udp_len_str)
            payload = max(udp_total - 8, 0)  # payload only
        else:
            payload = 0

        if t is None or src is None or dst is None:
            return None
        return (t, src, dst, payload)
    except Exception:
        return None

def detect_client_ip(records):
    """
    Heuristic: choose the IP (v4 or v6) that appears most often among src/dst.
    Works well for Docker bridge captures to find the container's IP.
    """
    counter = Counter()
    for (_, src, dst, _) in records:
        if src: counter[src] += 1
        if dst: counter[dst] += 1
    return counter.most_common(1)[0][0] if counter else None

def convert_pcap_to_wang14(pcap_path, out_dir, client_ip=None, drop_zero_payload=True):
    """
    Convert x_y.pcap -> x-y (Wang14-style trace: rel_time \t signed_len)
    """
    base = os.path.basename(pcap_path)
    stem, ext = os.path.splitext(base)
    if "_" not in stem:
        print(f"[WARN] Skip {base}: expected name like x_y.pcap")
        return
    x, y = stem.split("_", 1)
    out_name = f"{x}-{y}"
    out_path = os.path.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)

    # Run tshark once to get structured TSV
    cmd = ["tshark", "-r", pcap_path, "-Y", DISPLAY_FILTER] + TSHARK_FIELDS
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("[ERROR] tshark not found. Install with: brew install wireshark")
        return

    if result.returncode != 0:
        err = result.stderr.strip()
        print(f"[ERROR] tshark failed on {base}: {err}")
        return

    # Parse all lines first to (t, src, dst, payload_len)
    records = []
    for line in result.stdout.splitlines():
        rec = parse_line_to_record(line)
        if rec is None:
            continue
        t, src, dst, payload = rec
        if drop_zero_payload and payload == 0:
            continue
        records.append(rec)

    if not records:
        # still write a minimal file with just terminator
        with open(out_path, "w") as f:
            f.write("0\t0\n")
        print(f"[OK] {base} -> {out_name} (no payload records)")
        return

    # Determine client IP
    local_ip = client_ip or detect_client_ip(records)

    # Build output lines: relative time, signed length ( +payload if src==local_ip else -payload )
    t0 = records[0][0]
    lines = []
    for (t, src, _dst, payload) in records:
        rel = t - t0
        direction = 1 if (local_ip and src == local_ip) else -1
        signed_len = direction * payload
        lines.append(f"{rel:.6f}\t{signed_len}")

    # Append sentinel row for Wang14 reader compatibility
    lines.append("0\t0")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[OK] {base} -> {out_name} (client IP assumed: {local_ip}, kept {len(records)} packets)")

def batch_convert(dataset_dir, out_dir, client_ip=None, drop_zero_payload=True):
    pcaps = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".pcap")]
    if not pcaps:
        print(f"[WARN] No .pcap files found in {dataset_dir}")
        return
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for _ in ex.map(lambda p: convert_pcap_to_wang14(p, out_dir, client_ip, drop_zero_payload), pcaps):
            pass

if __name__ == "__main__":
    ensure_tshark()

    if not os.path.isdir(DATASET_FOLDER):
        print(f"[ERROR] Input folder not found: {DATASET_FOLDER}")
        sys.exit(1)
    os.makedirs(FEATURES_RESULT_PATH, exist_ok=True)
    
    # Remove classes with <2 samples
    filter_insufficient_samples(DATASET_FOLDER, min_count=2)

    batch_convert(DATASET_FOLDER, FEATURES_RESULT_PATH, client_ip=CLIENT_IP, drop_zero_payload=DROP_ZERO_PAYLOAD)
