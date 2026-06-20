// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later
//
// bitboard.cpp — initialization of attack tables and runtime magic-number search.

#include "bitboard.h"
#include <random>
#include <vector>
#include <sstream>

namespace owen {

Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
Bitboard KnightAttacks[SQUARE_NB];
Bitboard KingAttacks[SQUARE_NB];

Magic RookMagics[SQUARE_NB];
Magic BishopMagics[SQUARE_NB];

// Backing storage for all sliding-attack tables (rook: ~102K, bishop: ~5K entries).
static std::vector<Bitboard> RookTable;
static std::vector<Bitboard> BishopTable;

namespace {

// Shift a bitboard by (df, dr); returns 0 if it would wrap off the board.
Bitboard safe_shift(Square s, int df, int dr) {
    int f = file_of(s) + df, r = rank_of(s) + dr;
    if (f < 0 || f > 7 || r < 0 || r > 7) return 0;
    return square_bb(make_square(f, r));
}

// Sliding attacks computed on-the-fly by ray scanning, stopping at blockers.
Bitboard slider_rays(Square s, Bitboard occ, const int dirs[4][2]) {
    Bitboard att = 0;
    for (int d = 0; d < 4; ++d) {
        int f = file_of(s), r = rank_of(s);
        while (true) {
            f += dirs[d][0]; r += dirs[d][1];
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            Square t = make_square(f, r);
            att |= square_bb(t);
            if (occ & square_bb(t)) break;  // blocked
        }
    }
    return att;
}

const int RookDirs[4][2]   = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
const int BishopDirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

Bitboard rook_rays(Square s, Bitboard occ)   { return slider_rays(s, occ, RookDirs); }
Bitboard bishop_rays(Square s, Bitboard occ) { return slider_rays(s, occ, BishopDirs); }

// Relevant occupancy mask: rays excluding the board edges (and the square itself).
Bitboard slider_mask(Square s, bool rook) {
    Bitboard mask = 0;
    int f0 = file_of(s), r0 = rank_of(s);
    const int (*dirs)[2] = rook ? RookDirs : BishopDirs;
    for (int d = 0; d < 4; ++d) {
        int f = f0, r = r0;
        while (true) {
            f += dirs[d][0]; r += dirs[d][1];
            // Stop before the last square in each direction (edge excluded).
            int nf = f + dirs[d][0], nr = r + dirs[d][1];
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            if (nf < 0 || nf > 7 || nr < 0 || nr > 7) break;
            mask |= square_bb(make_square(f, r));
        }
    }
    return mask;
}

void init_leapers() {
    const int knightMoves[8][2] = {
        {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    const int kingMoves[8][2] = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

    for (int s = 0; s < 64; ++s) {
        Square sq = Square(s);
        Bitboard n = 0, k = 0;
        for (int i = 0; i < 8; ++i) {
            n |= safe_shift(sq, knightMoves[i][0], knightMoves[i][1]);
            k |= safe_shift(sq, kingMoves[i][0], kingMoves[i][1]);
        }
        KnightAttacks[sq] = n;
        KingAttacks[sq] = k;
        // Pawn capture squares (forward diagonals).
        PawnAttacks[WHITE][sq] = safe_shift(sq, -1, 1) | safe_shift(sq, 1, 1);
        PawnAttacks[BLACK][sq] = safe_shift(sq, -1, -1) | safe_shift(sq, 1, -1);
    }
}

// Find a collision-free magic for one square and fill its attack table slice.
void init_magics(bool rook, Magic magics[SQUARE_NB], std::vector<Bitboard>& table) {
    std::mt19937_64 rng(0xC0FFEE123456789ULL);  // fixed seed → reproducible builds
    auto sparse_rand = [&]() { return rng() & rng() & rng(); };

    // First pass: compute sizes and total table length.
    size_t total = 0;
    unsigned bits[SQUARE_NB];
    for (int s = 0; s < 64; ++s) {
        Bitboard mask = slider_mask(Square(s), rook);
        bits[s] = popcount(mask);
        magics[s].mask = mask;
        magics[s].shift = 64 - bits[s];
        total += size_t(1) << bits[s];
    }
    table.assign(total, 0);

    size_t offset = 0;
    for (int s = 0; s < 64; ++s) {
        Square sq = Square(s);
        Bitboard mask = magics[s].mask;
        unsigned n = bits[s];
        size_t size = size_t(1) << n;
        magics[s].attacks = &table[offset];
        offset += size;

        // Enumerate every occupancy subset of the mask (carry-rippler) + its attacks.
        std::vector<Bitboard> occ(size), ref(size);
        Bitboard b = 0;
        for (size_t i = 0; i < size; ++i) {
            occ[i] = b;
            ref[i] = rook ? rook_rays(sq, b) : bishop_rays(sq, b);
            b = (b - mask) & mask;
        }

        // Search for a magic that maps occupancies to indices without bad collisions.
        std::vector<int> epoch(size, -1);
        std::vector<Bitboard> used(size, 0);
        int cur = 0;
        while (true) {
            Bitboard magic = sparse_rand();
            // Heuristic reject: top byte should carry enough bits.
            if (popcount((mask * magic) & 0xFF00000000000000ULL) < 6) continue;

            magics[s].magic = magic;
            ++cur;
            bool ok = true;
            for (size_t i = 0; i < size && ok; ++i) {
                unsigned idx = magics[s].index(occ[i]);
                if (epoch[idx] < cur) {
                    epoch[idx] = cur;
                    used[idx] = ref[i];
                } else if (used[idx] != ref[i]) {
                    ok = false;  // index collision with differing attack set
                }
            }
            if (ok) {
                for (size_t i = 0; i < size; ++i)
                    magics[s].attacks[magics[s].index(occ[i])] = ref[i];
                break;
            }
        }
    }
}

} // namespace

void init_bitboards() {
    init_leapers();
    init_magics(true,  RookMagics,   RookTable);
    init_magics(false, BishopMagics, BishopTable);
}

std::string bb_to_string(Bitboard b) {
    std::ostringstream os;
    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f)
            os << ((b & square_bb(make_square(f, r))) ? " X" : " .");
        os << '\n';
    }
    return os.str();
}

} // namespace owen
