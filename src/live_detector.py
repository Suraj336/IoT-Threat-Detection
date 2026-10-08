"""
Real-world mode: watch live traffic on the hotspot interface, turn it into the
same 20 flow features the federated model was trained on, classify every flow
with models/global_model.pt, and write the results to live_detections.json for
the dashboard.

    sudo venv/bin/python src/live_detector.py                 # live, on wlp3s0
    sudo venv/bin/python src/live_detector.py --record benign # also save labelled flows
    venv/bin/python src/live_detector.py --pcap capture.pcap  # replay a capture file

Needs root (packet capture). Install the extra dependency once in your venv:
    pip install scapy

IMPORTANT: the current training data (data/prepare_data.py) is synthetic, so
the model has never seen real packets. Use --record to save real, labelled
flows extracted by THIS script, then retrain on them (data/build_real_dataset.py)
so training and deployment use exactly the same feature extractor.
"""
import argparse
import csv
import json
import os
import signal
import sys
import threading
import time
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))

from prepare_data import FEATURE_COLUMNS  # noqa: E402
from utils import CLASSES, list_client_files  # noqa: E402

SEVERITY = {"benign": "info", "scan": "medium", "brute_force": "high", "ddos": "critical"}


# --------------------------------------------------------------------------
# Flow tracking
# --------------------------------------------------------------------------
class Flow:
    """A bidirectional conversation, keyed by 5-tuple. 'fwd' = the side that
    sent the first packet (the initiator), 'bwd' = replies."""

    __slots__ = ("key", "src", "dst", "sport", "dport", "proto", "first", "last",
                 "times", "sizes", "fwd", "bwd", "syn", "rst")

    def __init__(self, key, src, dst, sport, dport, proto, ts):
        self.key = key
        self.src, self.dst, self.sport, self.dport, self.proto = src, dst, sport, dport, proto
        self.first = self.last = ts
        self.times, self.sizes = [], []
        self.fwd = self.bwd = self.syn = self.rst = 0

    def add(self, ts, size, forward, syn, rst):
        self.times.append(ts)
        self.sizes.append(size)
        self.last = ts
        if forward:
            self.fwd += 1
        else:
            self.bwd += 1
        self.syn += syn
        self.rst += rst

    def features(self):
        """Compute FEATURE_COLUMNS in the same units the training data uses:
        sizes in bytes, times in milliseconds, rates per second."""
        sizes = np.asarray(self.sizes, dtype=np.float64)
        n = len(sizes)
        iats = np.diff(np.asarray(self.times)) * 1000.0 if n > 1 else np.zeros(1)
        duration_ms = (self.last - self.first) * 1000.0
        # Rates use at least a 1-second window so 1-2 packet flows don't explode.
        window_s = max(duration_ms / 1000.0, 1.0)
        f = {
            "pkt_size_mean": sizes.mean(), "pkt_size_std": sizes.std(),
            "pkt_size_min": sizes.min(), "pkt_size_max": sizes.max(),
            "iat_mean": iats.mean(), "iat_std": iats.std(),
            "iat_min": iats.min(), "iat_max": iats.max(),
            "flow_duration": duration_ms,
            "fwd_pkt_count": self.fwd, "bwd_pkt_count": self.bwd,
            "byte_rate": sizes.sum() / window_s, "pkt_rate": n / window_s,
            "syn_flag_ratio": self.syn / n, "rst_flag_ratio": self.rst / n,
            "proto_tcp": float(self.proto == "tcp"),
            "proto_udp": float(self.proto == "udp"),
            "proto_icmp": float(self.proto == "icmp"),
            "dst_port_bucket": port_bucket(self.dport),
            "src_port_bucket": port_bucket(self.sport),
        }
        return [float(f[c]) for c in FEATURE_COLUMNS]


def port_bucket(port):
    return float(min(int(port) * 10 // 65536, 9)) if port else 0.0


class FlowTable:
    def __init__(self, idle_timeout, active_timeout, ignore_ports=()):
        self.flows = {}
        self.ignore_ports = set(ignore_ports)
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.lock = threading.Lock()

    def add_packet(self, pkt):
        from scapy.layers.inet import ICMP, IP, TCP, UDP

        if IP not in pkt:
            return
        ip = pkt[IP]
        ts = float(pkt.time)
        if TCP in pkt:
            proto, sport, dport = "tcp", ip[TCP].sport, ip[TCP].dport
            flags = int(ip[TCP].flags)
            syn, rst = int(bool(flags & 0x02)), int(bool(flags & 0x04))
        elif UDP in pkt:
            proto, sport, dport, syn, rst = "udp", ip[UDP].sport, ip[UDP].dport, 0, 0
        elif ICMP in pkt:
            proto, sport, dport, syn, rst = "icmp", 0, 0, 0, 0
        else:
            return
        if sport in self.ignore_ports or dport in self.ignore_ports:
            return

        fkey = (proto, ip.src, sport, ip.dst, dport)
        rkey = (proto, ip.dst, dport, ip.src, sport)
        with self.lock:
            if fkey in self.flows:
                flow, forward = self.flows[fkey], True
            elif rkey in self.flows:
                flow, forward = self.flows[rkey], False
            else:
                flow = Flow(fkey, ip.src, ip.dst, sport, dport, proto, ts)
                self.flows[fkey], forward = flow, True
            flow.add(ts, len(ip), forward, syn, rst)

    def pop_expired(self, now, flush_all=False):
        done = []
        with self.lock:
            for key, fl in list(self.flows.items()):
                if (flush_all or now - fl.last > self.idle_timeout
                        or now - fl.first > self.active_timeout):
                    done.append(self.flows.pop(key))
        return done


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Predictor:
    """Loads the trained global model plus a scaler fitted on the training
    shards (training scales each client with its own StandardScaler and never
    saves it, so we refit one on all client data combined)."""

    def __init__(self, model_path, clients_dir, cfg):
        import torch
        from model import ThreatClassifier

        self.torch = torch
        files = list_client_files(clients_dir)
        if not files:
            raise SystemExit(f"No client CSVs in {clients_dir}; the scaler needs them.")
        X = pd.concat([pd.read_csv(f) for f in files])[FEATURE_COLUMNS].to_numpy(np.float32)
        self.scaler = StandardScaler().fit(X)

        self.model = ThreatClassifier(
            len(FEATURE_COLUMNS), len(CLASSES),
            hidden_dims=cfg["model"]["hidden_dims"], dropout=cfg["model"]["dropout"],
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, X):
        Xs = self.scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(self.torch.tensor(Xs)), dim=1).numpy()
        return probs


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
class Reporter:
    def __init__(self, out_json, log_csv, record_csv, record_label, local_prefix,
                 iface, keep=300):
        self.out_json = out_json
        self.log_csv = log_csv
        self.record_csv = record_csv
        self.record_label = record_label
        self.local_prefix = local_prefix
        self.iface = iface
        self.recent = deque(maxlen=keep)
        self.totals = Counter()
        self.devices = defaultdict(Counter)
        self.started = time.time()
        for path in (log_csv, record_csv):
            if path:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def handle(self, flows, probs):
        rows_log, rows_rec = [], []
        for fl, p in zip(flows, probs):
            idx = int(p.argmax())
            cls = CLASSES[idx]
            feats = fl.features()
            device = fl.src if fl.src.startswith(self.local_prefix) else (
                fl.dst if fl.dst.startswith(self.local_prefix) else fl.src)
            event = {
                "time": round(fl.first, 3), "src": fl.src, "sport": fl.sport,
                "dst": fl.dst, "dport": fl.dport, "proto": fl.proto,
                "packets": len(fl.sizes), "bytes": int(sum(fl.sizes)),
                "duration_ms": round((fl.last - fl.first) * 1000, 1),
                "prediction": cls, "confidence": round(float(p[idx]), 3),
                "severity": SEVERITY[cls], "device": device,
            }
            self.recent.appendleft(event)
            self.totals[cls] += 1
            self.devices[device][cls] += 1
            rows_log.append({**event, **dict(zip(FEATURE_COLUMNS, feats))})
            if self.record_csv:
                rows_rec.append({**dict(zip(FEATURE_COLUMNS, feats)), "label": self.record_label})
        self._append(self.log_csv, rows_log)
        self._append(self.record_csv, rows_rec)

    @staticmethod
    def _append(path, rows):
        if not path or not rows:
            return
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)

    def write(self, status="running", active_flows=0):
        payload = {
            "status": status,
            "interface": self.iface,
            "started_at": self.started,
            "updated_at": time.time(),
            "active_flows": active_flows,
            "totals": {c: self.totals.get(c, 0) for c in CLASSES},
            "devices": {d: {c: n.get(c, 0) for c in CLASSES}
                        for d, n in sorted(self.devices.items())},
            "recording_label": self.record_label if self.record_csv else None,
            "recent": list(self.recent),
        }
        tmp = self.out_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, self.out_json)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="wlp3s0", help="interface to capture on")
    ap.add_argument("--pcap", help="read packets from a capture file instead of live")
    ap.add_argument("--filter", default=None,
                    help="optional BPF capture filter, e.g. 'host 192.168.50.22' (needs tcpdump)")
    ap.add_argument("--ignore_ports", type=int, nargs="*", default=[8000],
                    help="ports to skip (default 8000 = the dashboard's own web server)")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "global_model.pt"))
    ap.add_argument("--clients_dir", default=os.path.join(ROOT, "data", "clients"))
    ap.add_argument("--out", default=os.path.join(ROOT, "live_detections.json"))
    ap.add_argument("--log", default=os.path.join(ROOT, "data", "live", "flows_log.csv"),
                    help="every classified flow with IPs, ports and features")
    ap.add_argument("--record", metavar="LABEL", choices=CLASSES,
                    help="also save flows in training format with this label")
    ap.add_argument("--record_file", default=None,
                    help="default: data/recorded/<LABEL>_<timestamp>.csv")
    ap.add_argument("--local_prefix", default="192.168.50.",
                    help="IP prefix of devices on your hotspot")
    ap.add_argument("--idle_timeout", type=float, default=2.0)
    ap.add_argument("--active_timeout", type=float, default=30.0)
    args = ap.parse_args()

    from scapy.all import AsyncSniffer, sniff

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model}. Train first: python src/server.py")

    predictor = Predictor(args.model, args.clients_dir, cfg)
    record_file = None
    if args.record:
        record_file = args.record_file or os.path.join(
            ROOT, "data", "recorded", f"{args.record}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    reporter = Reporter(args.out, args.log, record_file, args.record,
                        args.local_prefix, args.pcap or args.iface)
    table = FlowTable(args.idle_timeout, args.active_timeout, args.ignore_ports)

    def process(flows):
        flows = [fl for fl in flows if fl.sizes]
        if flows:
            probs = predictor.predict([fl.features() for fl in flows])
            reporter.handle(flows, probs)
            for fl, p in zip(flows, probs):
                cls = CLASSES[int(p.argmax())]
                if cls != "benign":
                    print(f"[ALERT] {cls:<11} {p.max():.2f}  {fl.proto} "
                          f"{fl.src}:{fl.sport} -> {fl.dst}:{fl.dport}  pkts={len(fl.sizes)}")

    if args.pcap:
        sniff(offline=args.pcap, filter=args.filter, prn=table.add_packet, store=False)
        process(table.pop_expired(0, flush_all=True))
        reporter.write(status="finished")
        print(f"Done. Totals: {dict(reporter.totals)} -> {args.out}")
        return

    sniffer = AsyncSniffer(iface=args.iface, filter=args.filter,
                           prn=table.add_packet, store=False)
    sniffer.start()
    print(f"Watching {args.iface} ... writing {args.out}  (Ctrl+C to stop)")
    if record_file:
        print(f"Recording flows labelled '{args.record}' to {record_file}")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.is_set():
            stop.wait(1.0)
            process(table.pop_expired(time.time()))
            reporter.write(active_flows=len(table.flows))
    except KeyboardInterrupt:
        pass
    finally:
        sniffer.stop()
        process(table.pop_expired(0, flush_all=True))
        reporter.write(status="stopped")
        print(f"\nStopped. Totals: {dict(reporter.totals)}")


if __name__ == "__main__":
    main()
