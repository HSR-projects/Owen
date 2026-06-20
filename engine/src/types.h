// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later
//
// types.h — fundamental types: squares, pieces, colors, and 32-bit move encoding.

#pragma once

#include <cstdint>
#include <string>

namespace owen {

using Bitboard = uint64_t;

// ── Colors ──────────────────────────────────────────────────────────────────
enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };

constexpr Color operator~(Color c) { return Color(c ^ 1); }

// ── Piece types and pieces ───────────────────────────────────────────────────
// PieceType is color-agnostic; Piece encodes color*6 + type.
enum PieceType : int {
    PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5,
    PIECE_TYPE_NB = 6
};

enum Piece : int {
    WP = 0, WN = 1, WB = 2, WR = 3, WQ = 4, WK = 5,
    BP = 6, BN = 7, BB = 8, BR = 9, BQ = 10, BK = 11,
    NO_PIECE = 12, PIECE_NB = 12
};

constexpr Piece make_piece(Color c, PieceType pt) { return Piece(c * 6 + pt); }
constexpr Color color_of(Piece p) { return Color(p >= BP ? BLACK : WHITE); }
constexpr PieceType type_of(Piece p) { return PieceType(p % 6); }

// ── Squares ──────────────────────────────────────────────────────────────────
// 0 = a1, 1 = b1, ... 7 = h1, 8 = a2, ... 63 = h8.
enum Square : int {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    SQ_NONE = 64, SQUARE_NB = 64
};

constexpr int file_of(Square s) { return int(s) & 7; }
constexpr int rank_of(Square s) { return int(s) >> 3; }
constexpr Square make_square(int file, int rank) { return Square(rank * 8 + file); }

// ── Castling rights (KQkq bitmask) ────────────────────────────────────────────
enum CastlingRight : uint8_t {
    NO_CASTLING = 0,
    WHITE_OO  = 1,   // White king-side  (K)
    WHITE_OOO = 2,   // White queen-side (Q)
    BLACK_OO  = 4,   // Black king-side  (k)
    BLACK_OOO = 8,   // Black queen-side (q)
    ANY_CASTLING = 15
};

// ── Move encoding (32-bit) ────────────────────────────────────────────────────
// bits  0-5 : from square (0-63)
// bits  6-11: to square (0-63)
// bits 12-13: move type
// bits 14-15: promotion piece type offset (added to KNIGHT)
// bits 16-31: reserved for ordering score
enum MoveType : uint32_t {
    MT_NORMAL    = 0u << 12,
    MT_PROMOTION = 1u << 12,
    MT_ENPASSANT = 2u << 12,
    MT_CASTLING  = 3u << 12
};

using Move = uint32_t;
constexpr Move MOVE_NONE = 0;

constexpr Move make_move(Square from, Square to, MoveType mt = MoveType(MT_NORMAL),
                         PieceType promo = KNIGHT) {
    return uint32_t(from) | (uint32_t(to) << 6) | uint32_t(mt)
         | (uint32_t(promo - KNIGHT) << 14);
}

constexpr Square from_sq(Move m) { return Square(m & 0x3F); }
constexpr Square to_sq(Move m)   { return Square((m >> 6) & 0x3F); }
constexpr MoveType type_of_move(Move m) { return MoveType(m & (3u << 12)); }
constexpr PieceType promo_type(Move m)  { return PieceType(((m >> 14) & 3) + KNIGHT); }

// Long algebraic (UCI) string, e.g. "e2e4", "e7e8q".
inline std::string move_to_uci(Move m) {
    if (m == MOVE_NONE) return "0000";
    Square f = from_sq(m), t = to_sq(m);
    std::string s;
    s += char('a' + file_of(f));
    s += char('1' + rank_of(f));
    s += char('a' + file_of(t));
    s += char('1' + rank_of(t));
    if (type_of_move(m) == MT_PROMOTION) {
        const char pc[] = {'n', 'b', 'r', 'q'};
        s += pc[promo_type(m) - KNIGHT];
    }
    return s;
}

} // namespace owen
