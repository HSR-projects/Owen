// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later
//
// main.cpp — UCI entry point plus perft validation helpers.
// Usage:
//   owen            run the UCI loop
//   owen bench      run the perft self-test suite
//   owen perft "<fen>" <depth>   run perft on a position
//   owen divide "<fen>" <depth>  perft split per root move (debugging)

#include "board.h"
#include "movegen.h"
#include "uci.h"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

using namespace owen;

// Counts legal leaf nodes via pseudo-legal generation + inline legality check.
static uint64_t perft(Board& b, int depth) {
    if (depth == 0) return 1;
    MoveList moves;
    moves.reserve(64);
    generate_pseudo_legal(b, moves);
    Color us = b.side_to_move();
    uint64_t nodes = 0;
    for (Move m : moves) {
        b.make_move(m);
        if (!b.is_in_check(us))
            nodes += (depth == 1) ? 1 : perft(b, depth - 1);
        b.undo_move();
    }
    return nodes;
}

// Per-root-move breakdown — invaluable for locating a movegen bug.
static uint64_t divide(Board& b, int depth) {
    MoveList moves;
    generate_legal(b, moves);
    uint64_t total = 0;
    for (Move m : moves) {
        b.make_move(m);
        uint64_t n = (depth <= 1) ? 1 : perft(b, depth - 1);
        b.undo_move();
        std::cout << move_to_uci(m) << ": " << n << "\n";
        total += n;
    }
    std::cout << "total: " << total << "\n";
    return total;
}

struct Case { const char* fen; int depth; uint64_t expected; };

static int run_suite() {
    const Case cases[] = {
        // Start position (the canonical perft sequence).
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324},
        // Kiwipete — exercises castling, en passant, pins.
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690},
        // Position 3 — promotions, ep edge cases.
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083},
        // Position 4 (and its mirror).
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292},
        {"r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 5, 15833292},
        // Position 5 and 6.
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594},
    };

    int failures = 0;
    for (const Case& c : cases) {
        Board b;
        b.set_fen(c.fen);
        auto t0 = std::chrono::steady_clock::now();
        uint64_t got = perft(b, c.depth);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        bool ok = (got == c.expected);
        double nps = ms > 0 ? got / (ms / 1000.0) : 0;
        std::cout << (ok ? "[PASS] " : "[FAIL] ")
                  << "depth " << c.depth << " = " << got;
        if (!ok) std::cout << " (expected " << c.expected << ")";
        std::cout << "  [" << int(ms) << " ms, " << uint64_t(nps / 1000) << " kN/s]  "
                  << c.fen << "\n";
        if (!ok) ++failures;
    }
    std::cout << "\n" << (failures ? "FAILURES: " : "ALL PASSED (")
              << (failures ? std::to_string(failures) : std::to_string(sizeof(cases) / sizeof(Case)))
              << (failures ? "" : " cases)") << "\n";
    return failures;
}

int main(int argc, char** argv) {
    init_bitboards();
    Zobrist::init();

    if (argc >= 2 && std::string(argv[1]) == "bench") {
        return run_suite();
    }
    if (argc >= 4 && std::string(argv[1]) == "perft") {
        Board b; b.set_fen(argv[2]);
        int d = std::stoi(argv[3]);
        std::cout << perft(b, d) << "\n";
        return 0;
    }
    if (argc >= 4 && std::string(argv[1]) == "divide") {
        Board b; b.set_fen(argv[2]);
        divide(b, std::stoi(argv[3]));
        return 0;
    }
    uci_loop();
    return 0;
}
