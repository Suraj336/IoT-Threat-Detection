"""One Flower client = one simulated smart home. It trains locally on its own
traffic shard and only ever exports DP-noised model weights.
"""
import flwr as fl
import numpy as np
import torch
import torch.nn as nn

from model import ThreatClassifier, get_parameters, set_parameters
from privacy import privatize_update
from utils import compute_metrics


class SmartHomeClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, test_loader, num_features,
                 num_classes, cfg):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ThreatClassifier(
            num_features, num_classes,
            hidden_dims=cfg["model"]["hidden_dims"], dropout=cfg["model"]["dropout"],
        ).to(self.device)
        self.rng = np.random.default_rng(hash(client_id) % (2**32))

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        old_weights = [p.copy() for p in parameters]
        set_parameters(self.model, parameters)

        optimizer = torch.optim.SGD(self.model.parameters(),
                                     lr=self.cfg["federated"]["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for _ in range(self.cfg["federated"]["local_epochs"]):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

        new_weights = get_parameters(self.model)

        if self.cfg["privacy"]["enable_dp"]:
            new_weights = privatize_update(
                new_weights, old_weights,
                max_norm=self.cfg["privacy"]["max_grad_norm"],
                noise_multiplier=self.cfg["privacy"]["noise_multiplier"],
                rng=self.rng,
            )

        return new_weights, len(self.train_loader.dataset), {"client_id": self.client_id}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, all_preds, all_labels = 0.0, [], []

        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                total_loss += criterion(logits, yb).item() * len(yb)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        n = len(self.test_loader.dataset)
        metrics = compute_metrics(all_labels, all_preds)
        metrics["client_id"] = self.client_id
        return total_loss / n, n, metrics
