"""

    python data/build_real_dataset.py            # reads data/recorded/*.csv
    python src/server.py --dp_noise_multiplier 0.1

The old shards are moved to data/clients_synthetic_backup/ the first time.
Uses the same non-IID partitioning as prepare_data.py.
"""
import argparse
import glob
import os
import shutil

import pandas as pd
import yaml

from prepare_data import CLASSES, FEATURE_COLUMNS, partition_non_iid

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorded_dir", default=os.path.join(HERE, "recorded"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "clients"))
    ap.add_argument("--config", default=os.path.join(HERE, "..", "config.yaml"))
    ap.add_argument("--num_clients", type=int, default=None)
    ap.add_argument("--non_iid_alpha", type=float, default=None)
    ap.add_argument("--min_per_class", type=int, default=200,
                    help="warn when a class has fewer flows than this")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["data"]
    num_clients = args.num_clients or cfg["num_clients"]
    alpha = args.non_iid_alpha or cfg["non_iid_alpha"]

    files = sorted(glob.glob(os.path.join(args.recorded_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No recordings in {args.recorded_dir}. "
                         "Run: sudo venv/bin/python src/live_detector.py --record benign")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[FEATURE_COLUMNS + ["label"]].dropna()

    counts = df["label"].value_counts()
    print("Recorded flows per class:")
    for c in CLASSES:
        n = int(counts.get(c, 0))
        note = "  <-- too few, record more" if n < args.min_per_class else ""
        print(f"  {c:<12}{n:>7}{note}")
    missing = [c for c in CLASSES if counts.get(c, 0) == 0]
    if missing:
        raise SystemExit(f"No flows recorded for: {missing}. Every class needs examples.")

    # Balance classes so 'benign' (always the biggest) doesn't swamp the attacks.
    n_each = int(counts.min() if counts.min() >= args.min_per_class
                 else min(counts.max(), args.min_per_class))
    df = pd.concat([
        df[df["label"] == c].sample(n_each, replace=int(counts[c]) < n_each,
                                    random_state=cfg["seed"])
        for c in CLASSES
    ]).sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)

    if os.path.isdir(args.out_dir):
        backup = os.path.join(os.path.dirname(os.path.abspath(args.out_dir)),
                              "clients_synthetic_backup")
        if not os.path.exists(backup):
            shutil.copytree(args.out_dir, backup)
            print(f"Backed up old shards to {backup}")
        for f in glob.glob(os.path.join(args.out_dir, "client_*.csv")):
            os.remove(f)
    os.makedirs(args.out_dir, exist_ok=True)

    parts = partition_non_iid(df[FEATURE_COLUMNS], df["label"], num_clients, alpha,
                              seed=cfg["seed"])
    for i, (Xc, yc) in enumerate(parts):
        out = Xc.copy()
        out["label"] = yc
        out.to_csv(os.path.join(args.out_dir, f"client_{i}.csv"), index=False)
        print(f"client_{i}: n={len(out):5d}  {yc.value_counts().to_dict()}")
    print(f"\nWrote {num_clients} shards from {len(df)} real flows to {args.out_dir}")


if __name__ == "__main__":
    main()
