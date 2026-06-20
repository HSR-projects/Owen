# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
"""
model.py — OwenNet: a residual conv tower with policy + WDL value heads.

Sizes are configurable. Defaults are intentionally small so the full self-play
loop runs end-to-end on modest hardware (e.g. a 4 GB GPU or CPU). Scale `channels`
and `blocks` up when you have the compute -- the architecture is unchanged.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import N_PLANES, POLICY_SIZE


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 8))
        self.fc2 = nn.Linear(max(channels // reduction, 8), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))               # global average pool
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if self.se is not None:
            h = self.se(h)
        return F.relu(x + h)


class OwenNet(nn.Module):
    """Input (B, 19, 8, 8) -> policy logits (B, POLICY_SIZE), WDL logits (B, 3)."""

    def __init__(self, channels: int = 64, blocks: int = 6, use_se: bool = True):
        super().__init__()
        self.channels, self.blocks = channels, blocks

        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(channels, use_se) for _ in range(blocks)])

        # Policy head.
        self.p_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 64, POLICY_SIZE)

        # Value head (WDL).
        self.v_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(32 * 64, 128)
        self.v_fc2 = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor):
        x = self.tower(self.stem(x))

        p = F.relu(self.p_bn(self.p_conv(x)))
        p = self.p_fc(p.flatten(1))                       # raw logits

        v = F.relu(self.v_bn(self.v_conv(x)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        v = self.v_fc2(v)                                 # WDL logits
        return p, v

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def value_scalar(wdl_logits: torch.Tensor) -> torch.Tensor:
    """Scalar value in [-1, 1] = P(win) - P(loss) from WDL logits."""
    probs = F.softmax(wdl_logits, dim=-1)
    return probs[..., 0] - probs[..., 2]


if __name__ == "__main__":
    net = OwenNet()
    x = torch.zeros(2, N_PLANES, 8, 8)
    p, v = net(x)
    print(f"params={net.num_params():,}  policy={tuple(p.shape)}  value={tuple(v.shape)}")
