# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
"""
mcts.py — AlphaZero-style PUCT Monte-Carlo Tree Search guided by OwenNet.

Single-tree, single-threaded for clarity and correctness. Each simulation:
  1. SELECT  down the tree by maximising  Q + c_puct * P * sqrt(N_parent)/(1+N).
  2. EXPAND  the leaf with priors from the policy head (masked to legal moves).
  3. EVALUATE the leaf position with the value head (or game-over result).
  4. BACKUP  the value up the path, flipping sign each ply (negamax).

Terminal positions are scored exactly; the net is only queried at non-terminal leaves.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import chess

from encoder import encode_board, POLICY_SIZE, MOVE_TO_IDX
from model import OwenNet


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "to_play")

    def __init__(self, prior: float, to_play: bool):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[chess.Move, "Node"] = {}
        self.to_play = to_play          # side to move at this node (chess.WHITE/BLACK)

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@torch.inference_mode()
def _evaluate(net: OwenNet, board: chess.Board, device, use_amp: bool = True) -> tuple[np.ndarray, float]:
    """Return (legal-move priors over vocab, scalar value in [-1,1]) for `board`."""
    x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
    amp = use_amp and str(device).startswith("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        p_logits, v_logits = net(x)
    policy = torch.softmax(p_logits[0], dim=0).cpu().numpy()
    vprob = torch.softmax(v_logits[0], dim=0).cpu().numpy()
    value = float(vprob[0] - vprob[2])      # P(win) - P(loss), side-to-move POV
    return policy, value


def _terminal_value(board: chess.Board) -> float | None:
    """Value from the side-to-move POV if the game is over, else None."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


class MCTS:
    def __init__(self, net: OwenNet, config: dict, device="cpu"):
        self.net = net
        self.device = device
        self.c_puct = config.get("c_puct", 1.5)
        self.dirichlet_alpha = config.get("dirichlet_alpha", 0.3)
        self.dirichlet_eps = config.get("dirichlet_eps", 0.25)
        self.use_amp = config.get("use_amp", True)
        self.rng = np.random.default_rng(config.get("seed", 0))

    def _expand(self, node: Node, board: chess.Board, policy: np.ndarray):
        legal = list(board.legal_moves)
        priors = np.array([policy[MOVE_TO_IDX[m.uci()]] for m in legal], dtype=np.float64)
        s = priors.sum()
        priors = priors / s if s > 1e-8 else np.full(len(legal), 1.0 / len(legal))
        for mv, p in zip(legal, priors):
            node.children[mv] = Node(float(p), not board.turn)

    def _add_dirichlet_noise(self, node: Node):
        moves = list(node.children.keys())
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(moves))
        for mv, n in zip(moves, noise):
            c = node.children[mv]
            c.prior = (1 - self.dirichlet_eps) * c.prior + self.dirichlet_eps * float(n)

    def _select_child(self, node: Node):
        sqrt_n = math.sqrt(node.visit_count)
        best, best_move, best_child = -1e18, None, None
        for mv, child in node.children.items():
            # Q is stored from the child's POV; negate for the parent's POV.
            q = -child.value if child.visit_count > 0 else 0.0
            u = self.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            score = q + u
            if score > best:
                best, best_move, best_child = score, mv, child
        return best_move, best_child

    def run(self, board: chess.Board, simulations: int, add_noise: bool = True) -> dict:
        """Run search from `board`; return {move: visit_count}."""
        root = Node(0.0, board.turn)
        policy, _ = _evaluate(self.net, board, self.device, self.use_amp)
        self._expand(root, board, policy)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(simulations):
            node = root
            scratch = board.copy(stack=False)
            path = [node]

            # SELECT
            while node.expanded:
                mv, node = self._select_child(node)
                scratch.push(mv)
                path.append(node)

            # EVALUATE (+ EXPAND if non-terminal)
            value = _terminal_value(scratch)
            if value is None:
                policy, value = _evaluate(self.net, scratch, self.device, self.use_amp)
                self._expand(node, scratch, policy)

            # BACKUP (negamax: flip sign every ply going up)
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value

        return {mv: child.visit_count for mv, child in root.children.items()}


def select_move(visit_counts: dict, temperature: float, rng) -> chess.Move:
    """Pick a move from visit counts: sample at T>0, argmax at T==0."""
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)
    if temperature <= 1e-6:
        return moves[int(counts.argmax())]
    probs = counts ** (1.0 / temperature)
    probs /= probs.sum()
    return moves[int(rng.choice(len(moves), p=probs))]
