"""
Builds a labeled IoT-network-traffic dataset and partitions it across simulated
smart-home clients in a non-IID way (different homes see different attack mixes,
just like in real deployments).

REAL-DATA SWAP-IN NOTES
------------------------
To use a real dataset instead of the synthetic generator below, replace
`generate_synthetic_traffic()` with a loader that reads the dataset's CSV and maps
its columns onto the same feature schema used here:

  N-BaIoT (per-device CSVs, e.g. "Danmini_Doorbell/benign_traffic.csv",
           "..._mirai_attacks/scan.csv", etc.):
      - Concatenate benign + each attack CSV, label from filename/folder.
      - Columns are already numeric statistical features (MI_dir, H_L0.1, etc.) —
        just select/rename ~20 of them into FEATURE_COLUMNS below.

  TON_IoT (Train_Test_Network.csv):
      - `label` column: 0=benign,1=attack; `type` column has attack subtype.
      - One-hot encode `proto`, `service`; drop IP/port/timestamp columns before
        feeding to the model (they leak identity, not signal).

  CICIoT2023 (per-attack CSVs):
      - Attack category is the CSV/folder name; concatenate and map to your
        classes ({benign, scan, ddos, brute_force} or a finer-grained set).

In all cases: keep the per-client partitioning logic in `partition_non_iid()` —
that's what makes the simulation realistic, not the specific dataset.
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml

FEATURE_COLUMNS = [
    "pkt_size_mean", "pkt_size_std", "pkt_size_min", "pkt_size_max",
    "iat_mean", "iat_std", "iat_min", "iat_max",           # inter-arrival time stats
    "flow_duration", "fwd_pkt_count", "bwd_pkt_count",
    "byte_rate", "pkt_rate", "syn_flag_ratio", "rst_flag_ratio",
    "proto_tcp", "proto_udp", "proto_icmp",
    "dst_port_bucket", "src_port_bucket",
]

CLASSES = ["benign", "scan", "ddos", "brute_force"]


def _rng(seed):
    return np.random.default_rng(seed)


def generate_synthetic_traffic(n_samples, seed=42):
    """Generates feature vectors with class-conditional distributions loosely
    modeled on real IoT attack signatures:
      - scan: many short flows, high SYN ratio, low byte rate (port/host scanning)
      - ddos: very high packet rate, high byte rate, short flow duration
      - brute_force: repeated short flows to one dst_port_bucket, moderate rate
      - benign: everything else, low variance, periodic-ish IoT traffic
    """
    rng = _rng(seed)
    n_per_class = n_samples // len(CLASSES)
    rows, labels = [], []

    def clip_rand(loc, scale, size, lo=0.0):
        return np.clip(rng.normal(loc, scale, size), lo, None)

    for cls in CLASSES:
        n = n_per_class
        if cls == "benign":
            feat = {
                "pkt_size_mean": clip_rand(300, 60, n),
                "pkt_size_std": clip_rand(40, 10, n),
                "pkt_size_min": clip_rand(60, 10, n),
                "pkt_size_max": clip_rand(600, 80, n),
                "iat_mean": clip_rand(500, 100, n),
                "iat_std": clip_rand(80, 20, n),
                "iat_min": clip_rand(100, 20, n),
                "iat_max": clip_rand(1200, 200, n),
                "flow_duration": clip_rand(5000, 1500, n),
                "fwd_pkt_count": clip_rand(20, 5, n),
                "bwd_pkt_count": clip_rand(18, 5, n),
                "byte_rate": clip_rand(200, 50, n),
                "pkt_rate": clip_rand(4, 1, n),
                "syn_flag_ratio": clip_rand(0.05, 0.03, n, 0),
                "rst_flag_ratio": clip_rand(0.02, 0.02, n, 0),
            }
        elif cls == "scan":
            feat = {
                "pkt_size_mean": clip_rand(60, 10, n),
                "pkt_size_std": clip_rand(5, 2, n),
                "pkt_size_min": clip_rand(40, 5, n),
                "pkt_size_max": clip_rand(80, 10, n),
                "iat_mean": clip_rand(10, 5, n),
                "iat_std": clip_rand(5, 2, n),
                "iat_min": clip_rand(1, 0.5, n),
                "iat_max": clip_rand(30, 10, n),
                "flow_duration": clip_rand(50, 20, n),
                "fwd_pkt_count": clip_rand(2, 1, n),
                "bwd_pkt_count": clip_rand(0.2, 0.3, n),
                "byte_rate": clip_rand(30, 10, n),
                "pkt_rate": clip_rand(50, 15, n),
                "syn_flag_ratio": clip_rand(0.9, 0.08, n, 0),
                "rst_flag_ratio": clip_rand(0.6, 0.15, n, 0),
            }
        elif cls == "ddos":
            feat = {
                "pkt_size_mean": clip_rand(120, 30, n),
                "pkt_size_std": clip_rand(15, 5, n),
                "pkt_size_min": clip_rand(60, 10, n),
                "pkt_size_max": clip_rand(200, 40, n),
                "iat_mean": clip_rand(1, 0.5, n),
                "iat_std": clip_rand(0.5, 0.2, n),
                "iat_min": clip_rand(0.1, 0.05, n),
                "iat_max": clip_rand(3, 1, n),
                "flow_duration": clip_rand(200, 80, n),
                "fwd_pkt_count": clip_rand(500, 150, n),
                "bwd_pkt_count": clip_rand(5, 3, n),
                "byte_rate": clip_rand(5000, 1200, n),
                "pkt_rate": clip_rand(800, 200, n),
                "syn_flag_ratio": clip_rand(0.7, 0.1, n, 0),
                "rst_flag_ratio": clip_rand(0.1, 0.05, n, 0),
            }
        else:  # brute_force
            feat = {
                "pkt_size_mean": clip_rand(150, 25, n),
                "pkt_size_std": clip_rand(20, 5, n),
                "pkt_size_min": clip_rand(70, 10, n),
                "pkt_size_max": clip_rand(250, 40, n),
                "iat_mean": clip_rand(80, 30, n),
                "iat_std": clip_rand(20, 8, n),
                "iat_min": clip_rand(10, 5, n),
                "iat_max": clip_rand(200, 50, n),
                "flow_duration": clip_rand(300, 100, n),
                "fwd_pkt_count": clip_rand(15, 5, n),
                "bwd_pkt_count": clip_rand(14, 5, n),
                "byte_rate": clip_rand(400, 100, n),
                "pkt_rate": clip_rand(20, 6, n),
                "syn_flag_ratio": clip_rand(0.3, 0.1, n, 0),
                "rst_flag_ratio": clip_rand(0.35, 0.12, n, 0),
            }

        df_cls = pd.DataFrame(feat)
        proto = rng.choice(["tcp", "udp", "icmp"], size=n, p=[0.7, 0.25, 0.05]
                            if cls != "ddos" else [0.5, 0.4, 0.1])
        df_cls["proto_tcp"] = (proto == "tcp").astype(float)
        df_cls["proto_udp"] = (proto == "udp").astype(float)
        df_cls["proto_icmp"] = (proto == "icmp").astype(float)
        df_cls["dst_port_bucket"] = rng.integers(0, 10, n).astype(float)
        df_cls["src_port_bucket"] = rng.integers(0, 10, n).astype(float)

        rows.append(df_cls[FEATURE_COLUMNS])
        labels.extend([cls] * n)

    X = pd.concat(rows, ignore_index=True)
    y = pd.Series(labels, name="label")
    # shuffle
    idx = rng.permutation(len(X))
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def partition_non_iid(X, y, num_clients, alpha, seed=42):
    """Dirichlet-based non-IID partitioning (standard FL benchmark technique).

    Each client gets a different class distribution drawn from Dir(alpha) per
    class — e.g. a "camera-heavy" home might see mostly ddos-flavored traffic
    while a "sensor-heavy" home sees mostly benign + occasional scans. Lower
    alpha => more skewed/realistic; alpha -> inf => IID.
    """
    rng = _rng(seed)
    classes = sorted(y.unique())
    class_indices = {c: y[y == c].index.to_numpy().copy() for c in classes}
    for c in classes:
        rng.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in classes:
        idxs = class_indices[c]
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        splits = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        parts = np.split(idxs, splits)
        for client_id, part in enumerate(parts):
            client_indices[client_id].extend(part.tolist())

    client_data = []
    for indices in client_indices:
        rng.shuffle(indices)
        client_data.append((X.loc[indices].reset_index(drop=True),
                             y.loc[indices].reset_index(drop=True)))
    return client_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "config.yaml"))
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--non_iid_alpha", type=float, default=None)
    parser.add_argument("--out_dir", default=os.path.join(
        os.path.dirname(__file__), "clients"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["data"]

    num_clients = args.num_clients or cfg["num_clients"]
    alpha = args.non_iid_alpha or cfg["non_iid_alpha"]
    total_samples = num_clients * cfg["samples_per_client"]

    X, y = generate_synthetic_traffic(total_samples, seed=cfg["seed"])
    client_data = partition_non_iid(X, y, num_clients, alpha, seed=cfg["seed"])

    os.makedirs(args.out_dir, exist_ok=True)
    for i, (Xc, yc) in enumerate(client_data):
        df = Xc.copy()
        df["label"] = yc
        path = os.path.join(args.out_dir, f"client_{i}.csv")
        df.to_csv(path, index=False)
        dist = yc.value_counts(normalize=True).round(2).to_dict()
        print(f"client_{i}: n={len(df):5d}  class_dist={dist}")

    print(f"\nWrote {num_clients} client shards to {args.out_dir}")


if __name__ == "__main__":
    main()
