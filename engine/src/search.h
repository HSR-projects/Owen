// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "movegen.h"
#include "nnue.h"
#include <iosfwd>

namespace owen {

struct SearchLimits {
    int depth = 64;
    int movetime_ms = 0;
    int wtime_ms = 0;
    int btime_ms = 0;
    int winc_ms = 0;
    int binc_ms = 0;
    uint64_t nodes = 0;
    bool infinite = false;          // true only for "go infinite" (run until stop)
    int default_movetime_ms = 3000; // anti-hang cap when no clock/movetime is given
    int threads = 1;                // Lazy SMP search threads
};

struct SearchResult {
    Move best = MOVE_NONE;
    int score = 0;
    uint64_t nodes = 0;
    int depth = 0;
    int elapsed_ms = 0;
};

SearchResult search_bestmove(Board& board, const NNUE& nnue, const SearchLimits& limits,
                             std::ostream* info = nullptr);

} // namespace owen
