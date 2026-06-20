// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later
//
// bitboard.h — bitboard helpers and attack generation (magic bitboards for sliders,
// precomputed tables for pawns/knights/king).

#pragma once

#include "types.h"
#include <cassert>

namespace owen {

// ── Basic bit operations ──────────────────────────────────────────────────────
constexpr Bitboard square_bb(Square s) { return Bitboard(1) << int(s); }

inline int popcount(Bitboard b)   { return __builtin_popcountll(b); }
inline Square lsb(Bitboard b)     { return Square(__builtin_ctzll(b)); }

// Returns and clears the least-significant set bit.
inline Square pop_lsb(Bitboard& b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

// File / rank masks.
constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_H = FILE_A << 7;
constexpr Bitboard RANK_1 = 0xFFULL;
constexpr Bitboard RANK_2 = RANK_1 << 8;
constexpr Bitboard RANK_7 = RANK_1 << 48;
constexpr Bitboard RANK_8 = RANK_1 << 56;

// ── Precomputed leaper tables (defined in bitboard.cpp) ───────────────────────
extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
extern Bitboard KnightAttacks[SQUARE_NB];
extern Bitboard KingAttacks[SQUARE_NB];

// ── Magic bitboard machinery ──────────────────────────────────────────────────
struct Magic {
    Bitboard  mask;     // relevant occupancy mask (edges excluded)
    Bitboard  magic;    // multiplier
    Bitboard* attacks;  // pointer into the shared attack table
    unsigned  shift;    // 64 - relevant bits

    unsigned index(Bitboard occ) const {
        return unsigned(((occ & mask) * magic) >> shift);
    }
};

extern Magic RookMagics[SQUARE_NB];
extern Magic BishopMagics[SQUARE_NB];

inline Bitboard rook_attacks(Square s, Bitboard occ) {
    return RookMagics[s].attacks[RookMagics[s].index(occ)];
}
inline Bitboard bishop_attacks(Square s, Bitboard occ) {
    return BishopMagics[s].attacks[BishopMagics[s].index(occ)];
}
inline Bitboard queen_attacks(Square s, Bitboard occ) {
    return rook_attacks(s, occ) | bishop_attacks(s, occ);
}

// Must be called once at startup before any attack query.
void init_bitboards();

// Debug helper: render a bitboard as an 8x8 grid.
std::string bb_to_string(Bitboard b);

} // namespace owen
