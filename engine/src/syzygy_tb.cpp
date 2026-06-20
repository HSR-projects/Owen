// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later
//
// syzygy_tb.cpp — adapts Owen positions to the Fathom Syzygy prober (src/syzygy/).

#include "syzygy_tb.h"
#include "bitboard.h"
#include "syzygy/tbprobe.h"

namespace owen {

namespace {
bool g_ready = false;
}

void tb_setup(const std::string& path) {
    g_ready = false;
    if (path.empty()) return;
    if (tb_init(path.c_str()) && TB_LARGEST > 0)
        g_ready = true;
}

int tb_max_pieces() { return g_ready ? int(TB_LARGEST) : 0; }

bool tb_probe_root_best(const Board& b, Move& best, int& score) {
    if (!g_ready) return false;
    if (popcount(b.occupied()) > int(TB_LARGEST)) return false;
    if (b.castling() != 0) return false;            // TB positions carry no castling rights

    unsigned ep = (b.ep_square() == SQ_NONE) ? 0 : unsigned(b.ep_square());
    unsigned results[TB_MAX_MOVES];
    unsigned res = tb_probe_root(
        b.color_bb(WHITE), b.color_bb(BLACK),
        b.pieces(WHITE, KING)   | b.pieces(BLACK, KING),
        b.pieces(WHITE, QUEEN)  | b.pieces(BLACK, QUEEN),
        b.pieces(WHITE, ROOK)   | b.pieces(BLACK, ROOK),
        b.pieces(WHITE, BISHOP) | b.pieces(BLACK, BISHOP),
        b.pieces(WHITE, KNIGHT) | b.pieces(BLACK, KNIGHT),
        b.pieces(WHITE, PAWN)   | b.pieces(BLACK, PAWN),
        b.halfmove(), b.castling(), ep,
        b.side_to_move() == WHITE, results);

    if (res == TB_RESULT_FAILED || res == TB_RESULT_CHECKMATE || res == TB_RESULT_STALEMATE)
        return false;

    unsigned from = TB_GET_FROM(res), to = TB_GET_TO(res), promo = TB_GET_PROMOTES(res);

    // Match the probe's (from,to,promotion) to one of Owen's legal moves so the move
    // carries the correct flags (en passant / promotion) for make_move.
    MoveList legal;
    generate_legal(b, legal);
    Move found = MOVE_NONE;
    for (Move m : legal) {
        if (unsigned(from_sq(m)) != from || unsigned(to_sq(m)) != to) continue;
        if (promo != TB_PROMOTES_NONE) {
            if (type_of_move(m) != MT_PROMOTION) continue;
            PieceType want = promo == TB_PROMOTES_QUEEN  ? QUEEN
                           : promo == TB_PROMOTES_ROOK   ? ROOK
                           : promo == TB_PROMOTES_BISHOP ? BISHOP : KNIGHT;
            if (promo_type(m) != want) continue;
        } else if (type_of_move(m) == MT_PROMOTION) {
            continue;
        }
        found = m;
        break;
    }
    if (found == MOVE_NONE) return false;

    best = found;
    switch (TB_GET_WDL(res)) {
        case TB_WIN:          score =  4000; break;
        case TB_LOSS:         score = -4000; break;
        case TB_CURSED_WIN:   score =   100; break;   // win but drawn under 50-move rule
        case TB_BLESSED_LOSS: score =  -100; break;
        default:              score =     0; break;   // draw
    }
    return true;
}

} // namespace owen
