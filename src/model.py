"""Lightweight MLP classifier for IoT traffic feature vectors.

Kept intentionally small: real smart-home hubs / gateways (Raspberry Pi class
hardware) are the realistic deployment target for the *client* side of an FL
system, so the local model needs to train fast on modest compute.
"""
import torch
import torch.nn as nn


class ThreatClassifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dims=(64, 32),
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_parameters(model: nn.Module):
    """Flower needs model weights as a list of numpy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters):
    import numpy as np
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
