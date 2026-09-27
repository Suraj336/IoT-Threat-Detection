import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ["benign", "scan", "ddos", "brute_force"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_client_csv(path: str):
    df = pd.read_csv(path)
    y = df["label"].map(CLASS_TO_IDX).to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    return X, y


def make_client_dataloaders(csv_path: str, test_split: float, batch_size: int,
                             seed: int = 42, scaler: StandardScaler = None):
    X, y = load_client_csv(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y if len(set(y)) > 1 else None
    )

    if scaler is None:
        scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, scaler


def list_client_files(clients_dir: str):
    files = sorted(
        [f for f in os.listdir(clients_dir) if f.startswith("client_") and f.endswith(".csv")],
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )
    return [os.path.join(clients_dir, f) for f in files]


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
