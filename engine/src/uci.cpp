// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later

#include "uci.h"
#include "search.h"
#include "syzygy_tb.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#if defined(_WIN32)
#  include <windows.h>
#elif defined(__has_include)
#  if __has_include(<unistd.h>)
#    include <unistd.h>
#    define OWEN_HAVE_UNISTD 1
#  endif
#endif

namespace owen {

namespace {

Move parse_legal_move(const Board& b, const std::string& uci) {
    MoveList moves;
    generate_legal(b, moves);
    for (Move m : moves) {
        if (move_to_uci(m) == uci) return m;
    }
    return MOVE_NONE;
}

void set_position(Board& board, std::istringstream& ss) {
    std::string token;
    ss >> token;
    if (token == "startpos") {
        board.set_startpos();
        ss >> token;
    } else if (token == "fen") {
        std::string fen, part;
        for (int i = 0; i < 6 && ss >> part; ++i) {
            if (part == "moves") {
                token = part;
                break;
            }
            if (!fen.empty()) fen += ' ';
            fen += part;
        }
        if (!fen.empty()) board.set_fen(fen);
        if (part != "moves") ss >> token;
    }

    if (token != "moves") return;
    while (ss >> token) {
        Move m = parse_legal_move(board, token);
        if (m != MOVE_NONE) board.make_move(m);
    }
}

SearchLimits parse_go(std::istringstream& ss) {
    SearchLimits limits;
    std::string token;
    bool depth_set = false;
    while (ss >> token) {
        if (token == "depth") {
            ss >> limits.depth;
            depth_set = true;
        } else if (token == "movetime") {
            ss >> limits.movetime_ms;
        } else if (token == "wtime") {
            ss >> limits.wtime_ms;
        } else if (token == "btime") {
            ss >> limits.btime_ms;
        } else if (token == "winc") {
            ss >> limits.winc_ms;
        } else if (token == "binc") {
            ss >> limits.binc_ms;
        } else if (token == "nodes") {
            ss >> limits.nodes;
        } else if (token == "infinite") {
            limits.infinite = true;
            limits.depth = 64;
            depth_set = true;
        }
    }
    (void)depth_set;
    if (limits.depth < 1) limits.depth = 1;
    if (limits.depth > 100) limits.depth = 100;   // hard cap; time safety bounds real thinking
    return limits;
}

bool default_use_gpu() {
#ifdef OWEN_USE_CUDA
    return true;
#else
    return false;
#endif
}

std::string bool_text(bool value) {
    return value ? "true" : "false";
}

// Directory containing the running executable (so we can find the net regardless
// of the working directory a GUI launches us from).
std::string executable_dir() {
#if defined(_WIN32)
    char buf[4096];
    DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf)) {
        std::string p(buf, n);
        auto slash = p.find_last_of("/\\");
        if (slash != std::string::npos) return p.substr(0, slash);
    }
#elif defined(OWEN_HAVE_UNISTD)
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string p(buf);
        auto slash = p.find_last_of('/');
        if (slash != std::string::npos) return p.substr(0, slash);
    }
#endif
    return {};
}

std::string try_load_default_nnue(NNUE& nnue) {
    const char* names[] = {"nn-62ef826d1a6d.nnue", "owen.nnue"};
    std::string dir = executable_dir();

    std::vector<std::string> candidates;
    for (const char* nm : names) {
        if (!dir.empty()) {                 // executable-relative (works from any cwd)
            candidates.push_back(dir + "/../weights/" + nm);
            candidates.push_back(dir + "/weights/" + nm);
            candidates.push_back(dir + "/" + nm);
        }
        candidates.push_back(std::string("../weights/") + nm);   // cwd-relative fallback
        candidates.push_back(std::string("weights/") + nm);
        candidates.push_back(nm);
    }

    for (const std::string& path : candidates) {
        std::string error;
        if (nnue.load(path, &error)) return path;
    }
    return {};
}

} // namespace

void uci_loop() {
    Board board;
    NNUE nnue;
    std::string eval_file = try_load_default_nnue(nnue);
    bool use_gpu = default_use_gpu();
    std::string startup_gpu_error;
    int default_movetime = 3000;   // think time when a GUI gives no clock (anti-hang)
    int threads = 1;               // Lazy SMP search threads

    // Syzygy tablebases: default to <exe>/../weights/syzygy, override via SyzygyPath.
    std::string syzygy_path;
    { std::string d = executable_dir(); if (!d.empty()) syzygy_path = d + "/../weights/syzygy"; }
    tb_setup(syzygy_path);
    if (tb_max_pieces() > 0)
        std::cerr << "info string Syzygy: up to " << tb_max_pieces()
                  << "-man from " << syzygy_path << "\n";

    if (use_gpu) {
        if (!nnue.set_gpu_enabled(true, &startup_gpu_error)) use_gpu = false;
        if (!eval_file.empty() && nnue.loaded() && nnue.backend_name() != std::string("cuda")) {
            std::string error;
            if (!nnue.load(eval_file, &error)) eval_file.clear();
        }
    }

    // Debug log of the raw GUI <-> engine conversation (to diagnose GUI setups).
    // Override location with OWEN_LOG=/path; default /tmp/owen.log.
    std::ofstream dbg;
    { const char* lp = std::getenv("OWEN_LOG"); dbg.open(lp ? lp : "/tmp/owen.log", std::ios::app); }
    dbg << "\n=== Owen start (built " __DATE__ " " __TIME__ ") | net="
        << (eval_file.empty() ? "NONE(weak fallback eval!)" : eval_file)
        << " | backend=" << nnue.backend_name() << " ===\n";
    dbg.flush();

    std::string line;

    while (std::getline(std::cin, line)) {
        if (dbg) { dbg << "< " << line << "\n"; dbg.flush(); }
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "uci") {
            std::cout << "id name Owen\n";
            std::cout << "id author HSR-Projects\n";
            std::cout << "option name EvalFile type string default "
                      << (eval_file.empty() ? "<empty>" : eval_file) << "\n";
            std::cout << "option name UseGPU type check default " << bool_text(use_gpu) << "\n";
            std::cout << "option name MoveTime type spin default " << default_movetime
                      << " min 0 max 600000\n";
            std::cout << "option name Threads type spin default " << threads
                      << " min 1 max 256\n";
            std::cout << "option name SyzygyPath type string default "
                      << (syzygy_path.empty() ? "<empty>" : syzygy_path) << "\n";
            if (!startup_gpu_error.empty()) {
                std::cout << "info string GPU unavailable: " << startup_gpu_error << "\n";
            }
            if (!eval_file.empty()) {
                std::cout << "info string NNUE loaded " << eval_file
                          << " backend " << nnue.backend_name() << "\n";
            }
            std::cout << "uciok\n";
        } else if (cmd == "isready") {
            std::cout << "readyok\n";
        } else if (cmd == "ucinewgame") {
            board.set_startpos();
        } else if (cmd == "setoption") {
            std::string tok, name, value;
            while (ss >> tok) {
                if (tok == "name") {
                    ss >> name;
                } else if (tok == "value") {
                    std::getline(ss, value);
                    if (!value.empty() && value[0] == ' ') value.erase(0, 1);
                }
            }
            if (name == "EvalFile" && !value.empty() && value != "<empty>") {
                std::string error;
                if (!nnue.load(value, &error)) std::cout << "info string NNUE load failed: " << error << "\n";
                else {
                    eval_file = value;
                    std::cout << "info string NNUE loaded " << value << " backend " << nnue.backend_name() << "\n";
                }
            } else if (name == "MoveTime") {
                try { default_movetime = std::max(0, std::stoi(value)); } catch (...) {}
            } else if (name == "Threads") {
                try { threads = std::clamp(std::stoi(value), 1, 256); } catch (...) {}
            } else if (name == "SyzygyPath") {
                syzygy_path = (value == "<empty>") ? "" : value;
                tb_setup(syzygy_path);
                std::cerr << "info string Syzygy: " << tb_max_pieces() << "-man loaded\n";
            } else if (name == "UseGPU") {
                bool enabled = value == "true" || value == "1" || value == "on";
                std::string error;
                if (!nnue.set_gpu_enabled(enabled, &error)) {
                    use_gpu = false;
                    std::cout << "info string GPU unavailable: " << error << "\n";
                } else {
                    use_gpu = enabled;
                    std::cout << "info string NNUE backend " << nnue.backend_name() << "\n";
                }
            }
        } else if (cmd == "position") {
            set_position(board, ss);
        } else if (cmd == "go") {
            SearchLimits limits = parse_go(ss);
            limits.default_movetime_ms = default_movetime;
            limits.threads = threads;
            SearchResult r = search_bestmove(board, nnue, limits, &std::cout);
            if (r.depth == 0) {
                std::cout << "info depth 0 score cp " << r.score << " nodes " << r.nodes << "\n";
            }
            std::cout << "bestmove " << move_to_uci(r.best) << "\n";
            if (dbg) {
                dbg << "> bestmove " << move_to_uci(r.best) << "  (depth " << r.depth
                    << " score " << r.score << " threads " << threads
                    << " backend " << nnue.backend_name() << ")\n";
                dbg.flush();
            }
        } else if (cmd == "d") {
            std::cout << board.to_string();
        } else if (cmd == "quit") {
            break;
        }
    }
}

} // namespace owen
