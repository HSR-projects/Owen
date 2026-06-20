# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
"""
encoder.py — board <-> tensor encoding and the fixed policy move vocabulary.

Design choices (kept deliberately simple and *verifiable*):
  * Board is encoded in absolute coordinates (no color mirroring) with an explicit
    side-to-move plane. Mirroring improves sample efficiency and is a documented
    future optimization; absolute encoding is easier to validate and fully correct.
  * The policy is a fixed-size vector over a precomputed move vocabulary built from
    queen-ray + knight geometry plus all promotions. Every legal chess move's
    (from, to, promotion) is guaranteed to map into this vocabulary -- unit-tested
    in tests/test_encoder.py via random self-play move dumps.
"""

from __future__ import annotations
import numpy as np
import chess

# ── Board planes ──────────────────────────────────────────────────────────────
# 0-5   white  P N B R Q K
# 6-11  black  p n b r q k
# 12    side to move (1.0 = white to move)
# 13-16 castling rights  WK WQ BK BQ
# 17    en-passant target square
# 18    halfmove clock (normalised by 100)
N_PLANES = 19
BOARD_SHAPE = (N_PLANES, 8, 8)

_PIECE_TO_PLANE = {
    (chess.PAWN, True): 0, (chess.KNIGHT, True): 1, (chess.BISHOP, True): 2,
    (chess.ROOK, True): 3, (chess.QUEEN, True): 4, (chess.KING, True): 5,
    (chess.PAWN, False): 6, (chess.KNIGHT, False): 7, (chess.BISHOP, False): 8,
    (chess.ROOK, False): 9, (chess.QUEEN, False): 10, (chess.KING, False): 11,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Return a float32 (19, 8, 8) tensor for the given position."""
    planes = np.zeros(BOARD_SHAPE, dtype=np.float32)
    for sq, piece in board.piece_map().items():
        rank, file = chess.square_rank(sq), chess.square_file(sq)
        planes[_PIECE_TO_PLANE[(piece.piece_type, piece.color)], rank, file] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):  planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):  planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): planes[16, :, :] = 1.0
    if board.ep_square is not None:
        planes[17, chess.square_rank(board.ep_square),
                   chess.square_file(board.ep_square)] = 1.0
    planes[18, :, :] = min(board.halfmove_clock, 100) / 100.0
    return planes


# ── Move vocabulary ───────────────────────────────────────────────────────────
def _build_move_vocab() -> list[str]:
    """Deterministic list of every UCI move geometry that can ever be legal."""
    moves: set[str] = set()
    queen_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    knight = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]

    for frm in chess.SQUARES:
        ff, fr = chess.square_file(frm), chess.square_rank(frm)
        # Sliding (covers rook/bishop/queen/king/castling/pawn pushes & captures).
        for df, dr in queen_dirs:
            for dist in range(1, 8):
                tf, tr = ff + df * dist, fr + dr * dist
                if 0 <= tf < 8 and 0 <= tr < 8:
                    moves.add(chess.Move(frm, chess.square(tf, tr)).uci())
        for df, dr in knight:
            tf, tr = ff + df, fr + dr
            if 0 <= tf < 8 and 0 <= tr < 8:
                moves.add(chess.Move(frm, chess.square(tf, tr)).uci())

    # Promotions (white rank7->8, black rank2->1, straight + both captures).
    promos = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    for ff in range(8):
        for dff in (-1, 0, 1):
            tf = ff + dff
            if not 0 <= tf < 8:
                continue
            for frm, tr in ((chess.square(ff, 6), 7), (chess.square(ff, 1), 0)):
                for promo in promos:
                    moves.add(chess.Move(frm, chess.square(tf, tr), promotion=promo).uci())

    return sorted(moves)


MOVE_VOCAB: list[str] = _build_move_vocab()
MOVE_TO_IDX: dict[str, int] = {uci: i for i, uci in enumerate(MOVE_VOCAB)}
POLICY_SIZE: int = len(MOVE_VOCAB)


def move_index(move: chess.Move) -> int:
    return MOVE_TO_IDX[move.uci()]


def legal_indices(board: chess.Board) -> list[int]:
    return [MOVE_TO_IDX[m.uci()] for m in board.legal_moves]


def visits_to_policy(visit_counts: dict[chess.Move, int]) -> np.ndarray:
    """Convert MCTS visit counts to a normalised policy target over the vocab."""
    pi = np.zeros(POLICY_SIZE, dtype=np.float32)
    total = sum(visit_counts.values())
    if total == 0:
        return pi
    for mv, n in visit_counts.items():
        pi[MOVE_TO_IDX[mv.uci()]] = n / total
    return pi


# ── Value (WDL) helpers ───────────────────────────────────────────────────────
# WDL index from the perspective of the side to move at a given position.
WDL_WIN, WDL_DRAW, WDL_LOSS = 0, 1, 2


def result_to_wdl(result: str, pov_white: bool) -> int:
    """Map a game result ("1-0"/"0-1"/"1/2-1/2") to a WDL class for `pov_white`."""
    if result == "1/2-1/2":
        return WDL_DRAW
    white_won = result == "1-0"
    if white_won == pov_white:
        return WDL_WIN
    return WDL_LOSS


if __name__ == "__main__":
    print(f"planes={N_PLANES}  policy_size={POLICY_SIZE}")
