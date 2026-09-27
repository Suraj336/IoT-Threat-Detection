"""
Differential-privacy mechanisms applied client-side before any update leaves
the simulated smart home.

This implements the standard two-step DP-FedAvg client recipe (McMahan et al.,
"Learning Differentially Private Recurrent Language Models"):

  1. Clip the client's model-update vector (new_weights - old_weights) to a
     maximum L2 norm `max_grad_norm`. This bounds the *sensitivity* of any
     single client's contribution.
  2. Add i.i.d. Gaussian noise with std = noise_multiplier * max_grad_norm to
     every coordinate of the clipped update before sending it to the server.

Together, clipping + calibrated noise is what makes the mechanism (epsilon,
delta)-differentially private with respect to that client's local dataset:
no single client's presence/absence in training can be inferred (above a
bounded confidence) from what the server observes.

For a production-grade privacy accountant (tight epsilon tracking across many
rounds via the moments accountant / RDP), swap `estimate_epsilon` below for
Opacus's `opacus.accountants.RDPAccountant`, which composes noise across steps
far more tightly than the naive bound used here. This module's accountant is
intentionally simple/conservative so it has zero extra dependencies.
"""
import math
from typing import List

import numpy as np


def clip_update(update: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
    """Clip a flattened model-update (list of per-layer arrays) to max L2 norm."""
    flat_norm = math.sqrt(sum(float(np.sum(np.square(layer))) for layer in update))
    scale = min(1.0, max_norm / (flat_norm + 1e-12))
    return [layer * scale for layer in update]


def add_gaussian_noise(update: List[np.ndarray], max_norm: float,
                        noise_multiplier: float, rng: np.random.Generator = None
                        ) -> List[np.ndarray]:
    """Add Gaussian noise calibrated to the clipping bound, coordinate-wise."""
    rng = rng or np.random.default_rng()
    std = max_norm * noise_multiplier
    return [layer + rng.normal(0, std, size=layer.shape).astype(layer.dtype)
            for layer in update]


def privatize_update(new_weights: List[np.ndarray], old_weights: List[np.ndarray],
                      max_norm: float, noise_multiplier: float,
                      rng: np.random.Generator = None) -> List[np.ndarray]:
    """Full client-side DP step: compute update -> clip -> noise -> re-add to base.

    Returns weights ready to send to the server (old_weights + noised, clipped delta),
    so the server-side aggregation code doesn't need to know DP happened at all.
    """
    delta = [new - old for new, old in zip(new_weights, old_weights)]
    delta = clip_update(delta, max_norm)
    delta = add_gaussian_noise(delta, max_norm, noise_multiplier, rng)
    return [old + d for old, d in zip(old_weights, delta)]


def estimate_epsilon(num_rounds: int, sample_rate: float, noise_multiplier: float,
                      target_delta: float = 1e-5) -> float:
    """Loose, deliberately conservative epsilon estimate for a ballpark number
    in a writeup — NOT a tight accounting of your actual privacy loss.

    Per-round epsilon uses the standard single-application Gaussian-mechanism
    bound (Dwork & Roth, Thm 3.22): for a mechanism with sensitivity 1 and
    noise std = noise_multiplier, releasing one (delta)-approximate DP query
    costs eps_per_round = sqrt(2*ln(1.25/delta)) / noise_multiplier.
    Composing `num_rounds` of these under *basic* (linear, non-adaptive-tight)
    composition gives total_eps = num_rounds * eps_per_round — this over-counts
    privacy loss (badly, at realistic round counts) compared to a real RDP/
    moments-accountant, which is why it's only useful as a sanity-check upper
    bound, not a number to publish as your model's actual guarantee.

    For a real privacy audit, use Opacus's RDP accountant instead, which
    composes far more tightly and accounts for client subsampling correctly:

        from opacus.accountants import RDPAccountant
        acc = RDPAccountant()
        for _ in range(num_rounds):
            acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        epsilon = acc.get_epsilon(delta=target_delta)
    """
    eps_per_round = math.sqrt(2 * math.log(1.25 / target_delta)) / noise_multiplier
    total_eps = num_rounds * sample_rate * eps_per_round
    return total_eps


if __name__ == "__main__":
    # quick sanity check / demo
    eps = estimate_epsilon(num_rounds=20, sample_rate=1.0, noise_multiplier=1.0)
    print(f"Rough estimated privacy budget after 20 rounds: epsilon ~= {eps:.2f}")
