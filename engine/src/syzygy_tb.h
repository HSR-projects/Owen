// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later
//
// syzygy_tb.h — thin Owen-side wrapper around the Fathom Syzygy tablebase prober
// (third-party, in src/syzygy/, see that folder's notice). Owen's board/search are
// original; this only adapts Owen's position to Fathom's probe API.

#pragma once

#include "board.h"
#include "movegen.h"
#include <string>

namespace owen {

// Initialise tablebases from a directory of .rtbw/.rtbz files (no-op if empty/missing).
void tb_setup(const std::string& path);

// Largest piece count covered by the loaded tablebases (0 = none loaded).
int tb_max_pieces();

// If the current position is in the tablebases, fill `best` with a DTZ-optimal move
// (guarantees progress / avoids 50-move draws in won endgames) and `score` with a
// side-to-move-relative value, then return true. Otherwise return false.
bool tb_probe_root_best(const Board& b, Move& best, int& score);

} // namespace owen
