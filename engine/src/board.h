// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later
//
// board.h — board representation (bitboards + mailbox), Zobrist hashing,
// FEN parsing/generation, and incremental make/undo with an undo stack.

#pragma once

#include "types.h"
#include "bitboard.h"
#include <string>
#include <vector>

namespace owen {

// State that cannot be reconstructed from the move alone, saved per make_move.
struct UndoInfo {
    Move      move;
    Piece     captured;       // NO_PIECE if none
    Square    captured_sq;    // differs from to_sq() for en passant
    uint8_t   castling;
    Square    ep_square;
    uint16_t  halfmove;
    uint64_t  hash;
};

class Board {
public:
    Board() { set_startpos(); }

    // ── Setup ────────────────────────────────────────────────────────────────
    void set_startpos();
    void set_fen(const std::string& fen);
    std::string fen() const;

    // ── Accessors ──────────────────────────────────────────────────────────────
    Piece    piece_on(Square s) const { return mailbox_[s]; }
    Bitboard pieces(Piece p) const    { return pieces_[p]; }
    Bitboard pieces(Color c, PieceType pt) const { return pieces_[make_piece(c, pt)]; }
    Bitboard occupied() const         { return occupied_; }
    Bitboard color_bb(Color c) const  { return side_[c]; }
    Color    side_to_move() const     { return stm_; }
    Square   ep_square() const        { return ep_; }
    uint8_t  castling() const         { return castling_; }
    uint16_t halfmove() const         { return halfmove_; }
    uint64_t hash() const             { return hash_; }
    int      ply() const              { return ply_; }

    Square king_square(Color c) const { return lsb(pieces_[make_piece(c, KING)]); }

    // ── Queries ──────────────────────────────────────────────────────────────
    bool is_square_attacked(Square s, Color by) const;
    // Guarded against a missing king (illegal/king-captured positions) so the
    // engine never dereferences an empty king bitboard and crashes.
    bool is_in_check(Color c) const {
        Bitboard k = pieces_[make_piece(c, KING)];
        return k && is_square_attacked(lsb(k), ~c);
    }

    // True if the current position has occurred earlier in the game/search line
    // (within the fifty-move window) — used to detect draws by repetition.
    bool is_repetition() const;

    // True if `c` has any piece other than king/pawns (null-move safety guard).
    bool has_non_pawn_material(Color c) const {
        return pieces_[make_piece(c, KNIGHT)] | pieces_[make_piece(c, BISHOP)]
             | pieces_[make_piece(c, ROOK)]   | pieces_[make_piece(c, QUEEN)];
    }

    // ── Make / undo ────────────────────────────────────────────────────────────
    void make_move(Move m);
    void undo_move();
    void make_null_move();   // pass the turn (for null-move pruning)
    void undo_null_move();

    std::string to_string() const;

private:
    void put_piece(Piece p, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
    void recompute_occupancy();
    void clear();

    Bitboard pieces_[PIECE_NB] = {};
    Bitboard side_[COLOR_NB]   = {};
    Bitboard occupied_         = 0;
    Piece    mailbox_[SQUARE_NB];

    Color    stm_      = WHITE;
    uint8_t  castling_ = 0;
    Square   ep_       = SQ_NONE;
    uint16_t halfmove_ = 0;
    int      ply_      = 0;
    uint64_t hash_     = 0;

    std::vector<UndoInfo> history_;
};

// Zobrist keys (initialized by init_zobrist, called from init_bitboards path).
namespace Zobrist {
    extern uint64_t psq[PIECE_NB][SQUARE_NB];
    extern uint64_t castling[16];
    extern uint64_t ep_file[8];
    extern uint64_t side;
    void init();
}

} // namespace owen
