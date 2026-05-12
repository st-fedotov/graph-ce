"""The policy MLP that decides each edge bit.

Following Wagner: input is the concatenation of the current state (all edge
bits, with future positions zero) and a one-hot encoding of the current
position. Output is a single sigmoid: P(bit = 1 at this position).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PolicyMLP(nn.Module):
    def __init__(
        self,
        num_edges: int,
        hidden_sizes: list[int],
        init: str = "keras",
    ) -> None:
        super().__init__()
        input_dim = 2 * num_edges
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.num_edges = num_edges
        self._apply_init(init)

    def _apply_init(self, scheme: str) -> None:
        """Apply the configured weight init.

        ``keras`` (default; matches Adam Wagner's reference setup):
            Glorot/Xavier uniform weights, zero biases. Matches Keras Dense
            defaults. Initial sigmoid output is unbiased around 0.5, giving
            CEM an honest uniform-Bernoulli starting policy.

        ``pytorch_default`` (kept for ablation/repro of the original bug):
            PyTorch nn.Linear default: Kaiming uniform with ``a=sqrt(5)``
            weights and non-zero uniform biases in ``[-1/sqrt(fan_in),
            +1/sqrt(fan_in)]``. For a 4-layer stack the non-zero biases shift
            the initial policy away from Bernoulli(0.5), and CEM plateaus.
        """
        if scheme == "keras":
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif scheme == "pytorch_default":
            # nn.Linear's own __init__ has already applied its default Kaiming
            # uniform + non-zero bias; nothing more to do.
            pass
        else:
            raise ValueError(
                f"unknown model.init scheme: {scheme!r} "
                "(must be 'keras' or 'pytorch_default')"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (pre-sigmoid). Shape: (batch, 1)."""
        return self.net(x)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(bit = 1). Shape: (batch, 1)."""
        return torch.sigmoid(self.forward(x))
