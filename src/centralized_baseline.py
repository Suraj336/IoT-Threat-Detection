"""Trains the same model architecture on ALL clients' data pooled together,
with no federation and no differential privacy. This is your comparison point:
"how much accuracy do we give up for privacy + decentralization?" is exactly
the gap between this script's results and src/server.py's results.
"""
import os

import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import ThreatClassifier
from utils import CLASS_TO_IDX, compute_metrics, list_client_files

HERE = os.path.dirname(__file__)


def main():
    with open(os.path.join(HERE, "..", "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    clients_dir = os.path.join(HERE, "..", "data", "clients")
    files = list_client_files(clients_dir)
    if not files:
        raise SystemExit("No client shards found. Run data/prepare_data.py first.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    y = df["label"].map(CLASS_TO_IDX).to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype="float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"]["test_split"], random_state=cfg["data"]["seed"], stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=cfg["federated"]["batch_size"], shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                       torch.tensor(y_test, dtype=torch.long)),
        batch_size=cfg["federated"]["batch_size"], shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreatClassifier(
        cfg["data"]["num_features"], len(cfg["data"]["classes"]),
        hidden_dims=cfg["model"]["hidden_dims"], dropout=cfg["model"]["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = cfg["federated"]["rounds"] * cfg["federated"]["local_epochs"]
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    metrics = compute_metrics(all_labels, all_preds)
    print(f"Centralized (non-federated, no-DP) baseline after {epochs} epochs:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
