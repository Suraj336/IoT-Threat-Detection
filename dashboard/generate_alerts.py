"""
Each row is one flow the model classified as something other than benign,
which is the same trigger logic a real IDS would use to raise an alert.
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ThreatClassifier  # noqa: E402
from utils import CLASSES  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")

SEVERITY = {"benign": "info", "scan": "medium", "brute_force": "high", "ddos": "critical"}


def main():
    model_path = os.path.join(ROOT, "models", "global_model.pt")
    clients_dir = os.path.join(ROOT, "data", "clients")

    if not os.path.exists(model_path):
        raise SystemExit(
            "No trained model found at models/global_model.pt.\n"
            "Run `python src/server.py` first to train and save the global model."
        )
    if not os.path.isdir(clients_dir):
        raise SystemExit(
            "No client data found. Run `python data/prepare_data.py` first."
        )

    num_features = 20
    model = ThreatClassifier(num_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    client_files = sorted(
        [f for f in os.listdir(clients_dir) if f.startswith("client_")],
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )

    rows = []
    now = datetime.datetime.now()
    rng = np.random.default_rng(7)

    for cf in client_files:
        client_id = int(cf.split("_")[1].split(".")[0])
        df = pd.read_csv(os.path.join(clients_dir, cf))
        X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
        y_true = df["label"].to_numpy()

        with torch.no_grad():
            logits = model(torch.tensor(X))
            probs = torch.softmax(logits, dim=1).numpy()
            preds = probs.argmax(axis=1)

        # Only rows the model flagged as non-benign become alerts, mirroring
        # how a real IDS would only page an analyst on suspected threats.
        for i in range(len(df)):
            pred_class = CLASSES[preds[i]]
            if pred_class == "benign":
                continue
            confidence = float(probs[i, preds[i]])
            minutes_ago = int(rng.integers(0, 24 * 60))  # spread over last 24h
            rows.append({
                "timestamp": (now - datetime.timedelta(minutes=minutes_ago)).isoformat(timespec="seconds"),
                "home_id": f"home-{client_id:02d}",
                "predicted_class": pred_class,
                "true_class": y_true[i],
                "confidence": round(confidence, 3),
                "severity": SEVERITY[pred_class],
                "correct": bool(pred_class == y_true[i]),
            })

    alerts_df = pd.DataFrame(rows).sort_values("timestamp", ascending=False)
    out_path = os.path.join(ROOT, "data", "alerts.csv")
    alerts_df.to_csv(out_path, index=False)
    print(f"Wrote {len(alerts_df)} alerts from {len(client_files)} homes to {out_path}")
    print(alerts_df["severity"].value_counts().to_string())


if __name__ == "__main__":
    main()
