# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
"""
selfplay.py — generate a self-play game with MCTS and emit training samples.

Each visited position yields (state, mcts_policy, side_to_move). After the game
ends the result is converted to a WDL class from each position's POV.
"""

from __future__ import annotations
import numpy as np
import chess

from encoder import encode_board, visits_to_policy, result_to_wdl
from mcts import MCTS, select_move


def play_game(net, config: dict, device="cpu", seed: int = 0):
    """Play one self-play game. Returns (samples, result_str, n_moves)."""
    rng = np.random.default_rng(seed)
    mcts_cfg = dict(config)
    mcts_cfg["seed"] = seed
    mcts = MCTS(net, mcts_cfg, device=device)

    sims = config.get("simulations", 100)
    temp_moves = config.get("temperature_moves", 30)
    max_moves = config.get("max_moves", 200)
    resign_threshold = config.get("resign_threshold", None)  # e.g. -0.95

    board = chess.Board()
    history: list[tuple[np.ndarray, np.ndarray, bool]] = []
    result = "1/2-1/2"

    for move_num in range(max_moves):
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            break

        visit_counts = mcts.run(board, sims, add_noise=True)
        history.append((encode_board(board), visits_to_policy(visit_counts), board.turn))

        temperature = 1.0 if move_num < temp_moves else 0.0
        move = select_move(visit_counts, temperature, rng)

        # Optional resignation to shorten hopeless games.
        if resign_threshold is not None:
            root_value = _root_value(visit_counts, board)
            if root_value is not None and root_value < resign_threshold:
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                break

        board.push(move)
    else:
        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"

    samples = [
        (state, policy, result_to_wdl(result, pov_white=(turn == chess.WHITE)))
        for state, policy, turn in history
    ]
    return samples, result, len(history)


def _root_value(visit_counts: dict, board: chess.Board):
    """Crude value estimate from the most-visited child (for resign logic)."""
    if not visit_counts:
        return None
    return None  # placeholder; resignation disabled unless a value source is wired
