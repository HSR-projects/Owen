// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later
//
// movegen.h — pseudo-legal and legal move generation.

#pragma once

#include "board.h"
#include <vector>

namespace owen {

using MoveList = std::vector<Move>;

constexpr int MAX_MOVES = 256;   // safe upper bound on legal moves in any position

// Fast path: fill a caller-provided buffer, return the move count (no allocation).
int generate_pseudo_legal(const Board& b, Move* moves);

// Convenience wrappers (allocate a vector).
void generate_pseudo_legal(const Board& b, MoveList& out);

// Appends only fully legal moves (king not left in check).
void generate_legal(const Board& b, MoveList& out);

inline MoveList legal_moves(const Board& b) {
    MoveList m;
    generate_legal(b, m);
    return m;
}

} // namespace owen
