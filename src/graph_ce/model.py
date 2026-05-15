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
        activation: str = "relu",
    ) -> None:
        super().__init__()
        input_dim = 2 * num_edges
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(self._make_activation(activation))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.num_edges = num_edges
        self._apply_init(init)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "leaky_relu":
            return nn.LeakyReLU()
        raise ValueError(
            f"unknown model.activation: {name!r} "
            "(must be 'relu' or 'leaky_relu')"
        )

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

        ``pytorch_weights_zero_bias`` (ablation: isolate the bias):
            PyTorch nn.Linear default weights (Kaiming uniform with
            ``a=sqrt(5)``) but biases forced to zero. Tests whether the
            plateau is caused purely by the non-zero biases.

        ``xavier_weights_pytorch_bias`` (ablation: isolate the weight scale):
            Xavier uniform weights, but PyTorch's default non-zero uniform
            biases left in place. Tests whether the larger Xavier weights
            alone restore enough input-dependence to escape the plateau.
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
        elif scheme == "pytorch_weights_zero_bias":
            # Keep nn.Linear's default Kaiming-uniform weights; zero biases.
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif scheme == "xavier_weights_pytorch_bias":
            # Replace weights with Xavier; leave PyTorch's nonzero uniform
            # biases as nn.Linear's __init__ set them.
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
        else:
            raise ValueError(
                f"unknown model.init scheme: {scheme!r} "
                "(must be 'keras', 'pytorch_default', "
                "'pytorch_weights_zero_bias', or 'xavier_weights_pytorch_bias')"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (pre-sigmoid). Shape: (batch, 1)."""
        return self.net(x)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(bit = 1). Shape: (batch, 1)."""
        return torch.sigmoid(self.forward(x))
