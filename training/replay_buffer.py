# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
"""replay_buffer.py — FIFO experience buffer of (state, policy, wdl) samples."""

from __future__ import annotations
import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.buf: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buf)

    def add_game(self, samples: list[tuple[np.ndarray, np.ndarray, int]]):
        """samples: list of (state(19,8,8) f32, policy(POLICY_SIZE) f32, wdl_class int)."""
        self.buf.extend(samples)

    def sample(self, batch_size: int):
        n = min(batch_size, len(self.buf))
        batch = self.rng.sample(self.buf, n)
        states = np.stack([b[0] for b in batch]).astype(np.float32)
        policies = np.stack([b[1] for b in batch]).astype(np.float32)
        wdl = np.array([b[2] for b in batch], dtype=np.int64)
        return states, policies, wdl
