# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests: policy-vocab coverage, encoding shape, value mapping."""

import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chess
from encoder import (MOVE_TO_IDX, encode_board, BOARD_SHAPE,
                     result_to_wdl, WDL_WIN, WDL_DRAW, WDL_LOSS)


def test_vocab_covers_all_legal_moves():
    random.seed(1)
    missing = 0
    for _ in range(200):
        b = chess.Board()
        for _ in range(60):
            if b.is_game_over():
                break
            for m in b.legal_moves:
                if m.uci() not in MOVE_TO_IDX:
                    missing += 1
            b.push(random.choice(list(b.legal_moves)))
    assert missing == 0


def test_encode_shape():
    assert encode_board(chess.Board()).shape == BOARD_SHAPE


def test_result_to_wdl():
    assert result_to_wdl("1-0", pov_white=True) == WDL_WIN
    assert result_to_wdl("1-0", pov_white=False) == WDL_LOSS
    assert result_to_wdl("0-1", pov_white=True) == WDL_LOSS
    assert result_to_wdl("1/2-1/2", pov_white=True) == WDL_DRAW


if __name__ == "__main__":
    test_vocab_covers_all_legal_moves()
    test_encode_shape()
    test_result_to_wdl()
    print("all tests passed")
