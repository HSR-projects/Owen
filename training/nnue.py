# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small NNUE-style evaluator trained from Owen replay batches.

The native engine consumes the text export written by ``export_nnue``. The input
features intentionally match ``engine/src/nnue.cpp``: 12 piece-square planes plus
side-to-move and castling-right flags.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NNUE_INPUT_SIZE = 12 * 64 + 5


class NNUENet(nn.Module):
    """Sparse-friendly value net: binary board features -> scalar in [-1, 1]."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(NNUE_INPUT_SIZE, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc2(F.relu(self.fc1(x)))).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def states_to_nnue_features(states: torch.Tensor) -> torch.Tensor:
    """Convert Owen board tensors (B, 19, 8, 8) into NNUE feature vectors."""
    pieces = states[:, :12].flatten(1)
    side = states[:, 12, 0, 0].unsqueeze(1)
    castling = torch.stack(
        [states[:, 13, 0, 0], states[:, 14, 0, 0], states[:, 15, 0, 0], states[:, 16, 0, 0]],
        dim=1,
    )
    return torch.cat([pieces, side, castling], dim=1)


def wdl_to_scalar(wdl: torch.Tensor) -> torch.Tensor:
    """WDL class target from side-to-move POV: win=1, draw=0, loss=-1."""
    return torch.where(wdl == 0, 1.0, torch.where(wdl == 2, -1.0, 0.0)).to(torch.float32)


def export_nnue(net: NNUENet, path: str):
    """Write a deterministic text format that the C++ engine can load directly."""
    clone = NNUENet(net.hidden)
    clone.load_state_dict({k: v.detach().cpu() for k, v in net.state_dict().items()})
    clone.eval()
    net = clone
    with open(path, "w", encoding="utf-8") as f:
        f.write("OWEN_NNUE 1\n")
        f.write(f"input {NNUE_INPUT_SIZE} hidden {net.hidden}\n")
        _write_tensor(f, "w1", net.fc1.weight)
        _write_tensor(f, "b1", net.fc1.bias)
        _write_tensor(f, "w2", net.fc2.weight.squeeze(0))
        _write_tensor(f, "b2", net.fc2.bias)


def _write_tensor(f, name: str, tensor: torch.Tensor):
    vals = tensor.detach().cpu().reshape(-1).tolist()
    f.write(f"{name} {len(vals)}\n")
    for i, v in enumerate(vals):
        f.write(f"{float(v):.9g}")
        f.write("\n" if (i + 1) % 8 == 0 else " ")
    if len(vals) % 8:
        f.write("\n")

