// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later

#include "search.h"
#include "syzygy_tb.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace owen {

namespace {

constexpr int INF = 30000;
constexpr int MATE = 29000;
constexpr int MATE_IN_MAX = MATE - 256;
constexpr int MAX_PLY = 128;
constexpr int TT_SIZE = 1 << 20;

enum TTFlag : uint8_t { TT_EXACT, TT_LOWER, TT_UPPER };

struct TTEntry {
    uint64_t key = 0;
    Move best = MOVE_NONE;
    int16_t score = 0;
    int8_t depth = -1;
    uint8_t flag = TT_EXACT;
};

struct SearchState {
    const NNUE& nnue;
    SearchLimits limits;
    std::vector<TTEntry>& tt;             // shared across threads (Lazy SMP)
    std::atomic<bool>& stop;              // shared stop flag
    Move killers[MAX_PLY][2] = {};
    int history[COLOR_NB][SQUARE_NB][SQUARE_NB] = {};
    uint64_t nodes = 0;
    bool stopped = false;
    std::chrono::steady_clock::time_point start;
    int time_limit_ms = 0;
    bool can_timeout = false;   // false until the minimum-depth floor is reached

    SearchState(const NNUE& n, const SearchLimits& l, const Board& b,
                std::vector<TTEntry>& tt_, std::atomic<bool>& stop_,
                std::chrono::steady_clock::time_point start_)
        : nnue(n), limits(l), tt(tt_), stop(stop_), start(start_) {
        if (limits.movetime_ms > 0) {
            time_limit_ms = limits.movetime_ms;
        } else {
            int remaining = b.side_to_move() == WHITE ? limits.wtime_ms : limits.btime_ms;
            int inc = b.side_to_move() == WHITE ? limits.winc_ms : limits.binc_ms;
            if (remaining > 0) {
                time_limit_ms = std::max(20, remaining / 30 + inc / 2);
                time_limit_ms = std::min(time_limit_ms, std::max(20, remaining / 4));
            }
        }
        // Anti-hang safety: with no clock/movetime (e.g. a GUI sending "go depth 29"),
        // still bound thinking time so Owen always returns a move instead of freezing.
        // "go infinite" and explicit node limits are left uncapped.
        if (time_limit_ms == 0 && !limits.infinite && limits.nodes == 0)
            time_limit_ms = limits.default_movetime_ms;
    }

    int elapsed_ms() const {
        auto e = std::chrono::steady_clock::now() - start;
        return int(std::chrono::duration_cast<std::chrono::milliseconds>(e).count());
    }

    bool should_stop() {
        if (stopped) return true;
        if (stop.load(std::memory_order_relaxed)) { stopped = true; return true; }
        if (limits.nodes > 0 && nodes >= limits.nodes) {
            stopped = true; stop.store(true, std::memory_order_relaxed); return true;
        }
        // Never abort on time before the minimum-depth floor is reached, so a GUI
        // that hands Owen almost no time can't force depth-1 (blunder) moves.
        if (can_timeout && time_limit_ms > 0 && (nodes & 2047) == 0 &&
            elapsed_ms() >= time_limit_ms) {
            stopped = true; stop.store(true, std::memory_order_relaxed); return true;
        }
        return false;
    }
};

int piece_value(Piece p) {
    static constexpr int value[PIECE_TYPE_NB] = {100, 320, 330, 500, 900, 20000};
    return p == NO_PIECE ? 0 : value[type_of(p)];
}

bool is_capture_or_promotion(const Board& b, Move m) {
    return b.piece_on(to_sq(m)) != NO_PIECE || type_of_move(m) == MT_ENPASSANT ||
           type_of_move(m) == MT_PROMOTION;
}

int see(const Board& b, Move m);   // forward declaration (defined below)

int move_order_score(const Board& b, Move m, Move tt_move, SearchState& st, int ply) {
    if (m == tt_move) return 2'000'000;
    int score = 0;
    Piece victim = b.piece_on(to_sq(m));
    Piece attacker = b.piece_on(from_sq(m));
    if (type_of_move(m) == MT_ENPASSANT) victim = make_piece(~b.side_to_move(), PAWN);
    if (victim != NO_PIECE) {
        int mvv_lva = 10 * piece_value(victim) - piece_value(attacker);
        // Winning/equal captures sort above killers; losing captures (SEE<0) below them.
        score += (see(b, m) >= 0 ? 1'000'000 : 100'000) + mvv_lva;
    }
    if (type_of_move(m) == MT_PROMOTION)
        score += 800'000 + piece_value(make_piece(b.side_to_move(), promo_type(m)));
    if (ply < MAX_PLY && m == st.killers[ply][0]) score += 700'000;
    if (ply < MAX_PLY && m == st.killers[ply][1]) score += 600'000;
    score += st.history[b.side_to_move()][from_sq(m)][to_sq(m)];
    return score;
}

// Selection sort step: move the highest-scoring remaining move into slot `i`.
inline void pick_move(Move* moves, int* scores, int n, int i) {
    int best = i;
    for (int j = i + 1; j < n; ++j)
        if (scores[j] > scores[best]) best = j;
    if (best != i) { std::swap(moves[i], moves[best]); std::swap(scores[i], scores[best]); }
}

constexpr int SEE_VAL[PIECE_TYPE_NB] = {100, 320, 330, 500, 900, 20000};

// All pieces (either color) attacking square `s` under occupancy `occ`.
Bitboard attackers_to(const Board& b, Square s, Bitboard occ) {
    Bitboard bishops = b.pieces(WHITE, BISHOP) | b.pieces(BLACK, BISHOP)
                     | b.pieces(WHITE, QUEEN)  | b.pieces(BLACK, QUEEN);
    Bitboard rooks   = b.pieces(WHITE, ROOK)   | b.pieces(BLACK, ROOK)
                     | b.pieces(WHITE, QUEEN)  | b.pieces(BLACK, QUEEN);
    return (PawnAttacks[BLACK][s] & b.pieces(WHITE, PAWN))
         | (PawnAttacks[WHITE][s] & b.pieces(BLACK, PAWN))
         | (KnightAttacks[s] & (b.pieces(WHITE, KNIGHT) | b.pieces(BLACK, KNIGHT)))
         | (KingAttacks[s]   & (b.pieces(WHITE, KING)   | b.pieces(BLACK, KING)))
         | (bishop_attacks(s, occ) & bishops)
         | (rook_attacks(s, occ) & rooks);
}

// Static Exchange Evaluation: net material from the capture sequence on to_sq(m),
// assuming both sides always recapture with their least valuable attacker.
int see(const Board& b, Move m) {
    Square to = to_sq(m), from = from_sq(m);
    Color stm = b.side_to_move();
    Bitboard occ = b.occupied();

    int target_val;
    if (type_of_move(m) == MT_ENPASSANT) {
        target_val = SEE_VAL[PAWN];
        occ ^= square_bb(Square(int(to) + (stm == WHITE ? -8 : 8)));   // remove ep pawn
    } else {
        Piece cp = b.piece_on(to);
        target_val = cp == NO_PIECE ? 0 : SEE_VAL[type_of(cp)];
    }

    int gain[32];
    int d = 0;
    gain[0] = target_val;
    PieceType attackerType = type_of(b.piece_on(from));
    occ ^= square_bb(from);                       // first attacker moves
    Bitboard attackers = attackers_to(b, to, occ);
    Color side = ~stm;

    while (true) {
        ++d;
        gain[d] = SEE_VAL[attackerType] - gain[d - 1];
        Bitboard side_att = attackers & b.color_bb(side) & occ;
        if (!side_att) break;
        // Least valuable attacker for the side to recapture.
        PieceType pt = PAWN;
        Bitboard lva = 0;
        for (; pt <= KING; pt = PieceType(pt + 1)) {
            Bitboard bb = side_att & b.pieces(side, pt);
            if (bb) { lva = square_bb(lsb(bb)); break; }
        }
        attackerType = pt;
        occ ^= lva;                               // remove it, re-scan for x-rays
        attackers = attackers_to(b, to, occ);
        side = ~side;
        if (d >= 31) break;
    }
    while (--d > 0) gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    return gain[0];
}

int quiescence(Board& b, SearchState& st, int alpha, int beta, int ply) {
    st.nodes++;
    if (st.should_stop()) return 0;
    if (ply >= MAX_PLY) return st.nnue.evaluate(b);

    Color us = b.side_to_move();
    bool in_check = b.is_in_check(us);
    int best = -INF;
    if (!in_check) {
        best = st.nnue.evaluate(b);
        if (best >= beta) return best;
        if (best > alpha) alpha = best;
    }

    Move moves[MAX_MOVES];
    int scores[MAX_MOVES];
    int n = generate_pseudo_legal(b, moves);
    for (int i = 0; i < n; ++i) scores[i] = move_order_score(b, moves[i], MOVE_NONE, st, ply);

    int legal = 0;
    for (int i = 0; i < n; ++i) {
        pick_move(moves, scores, n, i);
        Move m = moves[i];
        bool noisy = is_capture_or_promotion(b, m);
        if (!in_check && !noisy) continue;
        // Skip losing captures (SEE < 0) when not evading check.
        if (!in_check && b.piece_on(to_sq(m)) != NO_PIECE &&
            type_of_move(m) != MT_PROMOTION && see(b, m) < 0)
            continue;
        b.make_move(m);
        if (b.is_in_check(us)) { b.undo_move(); continue; }
        ++legal;
        int score = -quiescence(b, st, -beta, -alpha, ply + 1);
        b.undo_move();
        if (st.stopped) return 0;
        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) return best;
    }
    if (in_check && legal == 0) return -MATE + ply;
    return best;
}

int negamax(Board& b, SearchState& st, int depth, int alpha, int beta, int ply, bool can_null) {
    st.nodes++;
    if (st.should_stop()) return 0;
    if (ply >= MAX_PLY) return st.nnue.evaluate(b);
    if (ply > 0 && (b.halfmove() >= 100 || b.is_repetition())) return 0;   // draw

    Color us = b.side_to_move();
    bool in_check = b.is_in_check(us);
    if (depth <= 0) return quiescence(b, st, alpha, beta, ply);

    bool pv = beta - alpha > 1;                 // wide window ⇒ principal-variation node
    int alpha_orig = alpha;
    TTEntry& entry = st.tt[b.hash() & (TT_SIZE - 1)];
    Move tt_move = MOVE_NONE;
    if (entry.key == b.hash()) {
        tt_move = entry.best;
        if (entry.depth >= depth && ply > 0 && !pv) {
            if (entry.flag == TT_EXACT) return entry.score;
            if (entry.flag == TT_LOWER) alpha = std::max(alpha, int(entry.score));
            else if (entry.flag == TT_UPPER) beta = std::min(beta, int(entry.score));
            if (alpha >= beta) return entry.score;
        }
    }

    int static_eval = in_check ? -INF : st.nnue.evaluate(b);   // full eval (consistent with leaves)

    // Reverse futility pruning: if we're already far above beta, trust it.
    if (!pv && !in_check && depth <= 6 && beta > -MATE_IN_MAX && beta < MATE_IN_MAX &&
        static_eval - 85 * depth >= beta)
        return static_eval;

    // Null-move pruning: if passing the turn still beats beta, the position is winning.
    if (can_null && !pv && !in_check && depth >= 3 && static_eval >= beta &&
        beta < MATE_IN_MAX && b.has_non_pawn_material(us)) {
        int R = 2 + depth / 6 + std::min((static_eval - beta) / 200, 3);
        b.make_null_move();
        int score = -negamax(b, st, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
        b.undo_null_move();
        if (st.stopped) return 0;
        if (score >= beta) return beta;
    }

    Move moves[MAX_MOVES];
    int scores[MAX_MOVES];
    int n = generate_pseudo_legal(b, moves);
    for (int i = 0; i < n; ++i) scores[i] = move_order_score(b, moves[i], tt_move, st, ply);

    int best_score = -INF;
    Move best_move = MOVE_NONE;
    int searched = 0;

    for (int i = 0; i < n; ++i) {
        pick_move(moves, scores, n, i);
        Move m = moves[i];
        bool noisy = is_capture_or_promotion(b, m);

        // Shallow pruning of quiet moves (after at least one move has been searched).
        if (!pv && !in_check && !noisy && searched > 0 && best_score > -MATE_IN_MAX) {
            if (depth <= 4 && searched >= 4 + depth * depth) continue;          // late move pruning
            if (depth <= 6 && static_eval + 90 + 70 * depth <= alpha) continue; // futility pruning
        }

        b.make_move(m);
        if (b.is_in_check(us)) { b.undo_move(); continue; }
        bool gives_check = b.is_in_check(b.side_to_move());
        int new_depth = depth - 1 + (gives_check ? 1 : 0);

        int score;
        if (searched == 0) {
            score = -negamax(b, st, new_depth, -beta, -alpha, ply + 1, true);
        } else {
            int reduction = 0;
            if (depth >= 3 && searched >= 3 && !noisy && !gives_check && !in_check) {
                reduction = 1;
                if (depth >= 6) reduction++;
                if (searched >= 6) reduction++;
                if (pv) reduction = std::max(1, reduction - 1);
                reduction = std::min(reduction, new_depth - 1);
            }
            score = -negamax(b, st, new_depth - reduction, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && reduction > 0)
                score = -negamax(b, st, new_depth, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && score < beta)
                score = -negamax(b, st, new_depth, -beta, -alpha, ply + 1, true);
        }
        b.undo_move();
        if (st.stopped) return 0;
        ++searched;

        if (score > best_score) { best_score = score; best_move = m; }
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            if (!noisy && ply < MAX_PLY) {
                st.killers[ply][1] = st.killers[ply][0];
                st.killers[ply][0] = m;
                st.history[us][from_sq(m)][to_sq(m)] += depth * depth;
            }
            break;
        }
    }

    if (searched == 0) return in_check ? -MATE + ply : 0;

    entry.key = b.hash();
    entry.best = best_move;
    entry.score = int16_t(std::clamp(best_score, -INF, INF));
    entry.depth = int8_t(std::min(depth, 127));
    if (best_score <= alpha_orig) entry.flag = TT_UPPER;
    else if (best_score >= beta) entry.flag = TT_LOWER;
    else entry.flag = TT_EXACT;
    return best_score;
}

int root_search(Board& b, SearchState& st, int depth, int alpha, int beta, Move& best_move) {
    Move moves[MAX_MOVES];
    int scores[MAX_MOVES];
    int n = generate_pseudo_legal(b, moves);
    Move tt_move = best_move;
    for (int i = 0; i < n; ++i) scores[i] = move_order_score(b, moves[i], tt_move, st, 0);

    Color us = b.side_to_move();
    int best_score = -INF;
    Move local_best = MOVE_NONE;
    int searched = 0;

    for (int i = 0; i < n; ++i) {
        pick_move(moves, scores, n, i);
        Move m = moves[i];
        bool noisy = is_capture_or_promotion(b, m);
        b.make_move(m);
        if (b.is_in_check(us)) { b.undo_move(); continue; }
        bool gives_check = b.is_in_check(b.side_to_move());
        int new_depth = depth - 1 + (gives_check ? 1 : 0);

        int score;
        if (searched == 0) {
            score = -negamax(b, st, new_depth, -beta, -alpha, 1, true);
        } else {
            int reduction = (depth >= 3 && searched >= 3 && !noisy && !gives_check) ? 1 : 0;
            score = -negamax(b, st, new_depth - reduction, -alpha - 1, -alpha, 1, true);
            if (score > alpha && reduction > 0)
                score = -negamax(b, st, new_depth, -alpha - 1, -alpha, 1, true);
            if (score > alpha && score < beta)
                score = -negamax(b, st, new_depth, -beta, -alpha, 1, true);
        }
        b.undo_move();
        if (st.stopped) break;
        ++searched;

        if (score > best_score) { best_score = score; local_best = m; }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }

    if (local_best != MOVE_NONE) best_move = local_best;
    return best_score;
}

uint64_t nps(uint64_t nodes, int elapsed_ms) {
    return elapsed_ms > 0 ? nodes * 1000ULL / uint64_t(elapsed_ms) : nodes;
}

// One thread's iterative-deepening loop. `info` is non-null only for the main thread.
void run_loop(SearchState& st, Board& board, const SearchLimits& limits,
              SearchResult& best, std::ostream* info) {
    MoveList root_moves;
    generate_legal(board, root_moves);
    if (root_moves.empty()) return;
    best.best = root_moves.front();

    int max_depth = limits.depth > 0 ? limits.depth : MAX_PLY;
    int min_depth = std::min(max_depth, 6);   // always complete at least this many plies
    int prev_score = 0;

    for (int depth = 1; depth <= max_depth; ++depth) {
        st.can_timeout = depth > min_depth;
        Move iter_best = best.best;
        int score;
        int window = 30;
        int alpha = depth <= 3 ? -INF : prev_score - window;
        int beta  = depth <= 3 ?  INF : prev_score + window;
        while (true) {
            score = root_search(board, st, depth, alpha, beta, iter_best);
            if (st.stopped) break;
            if (score <= alpha) alpha = -INF;
            else if (score >= beta) beta = INF;
            else break;
        }
        if (st.stopped) break;

        prev_score = score;
        best.best = iter_best;
        best.score = score;
        best.depth = depth;
        best.nodes = st.nodes;
        best.elapsed_ms = st.elapsed_ms();

        if (info) {
            *info << "info depth " << depth << " score ";
            if (score >= MATE_IN_MAX)       *info << "mate " << (MATE - score + 1) / 2;
            else if (score <= -MATE_IN_MAX) *info << "mate " << -(MATE + score + 1) / 2;
            else                            *info << "cp " << score;
            *info << " nodes " << best.nodes << " nps " << nps(best.nodes, best.elapsed_ms)
                  << " time " << best.elapsed_ms << " pv " << move_to_uci(best.best) << "\n";
            info->flush();
        }

        if (score >= MATE_IN_MAX || score <= -MATE_IN_MAX) break;
        if (depth >= min_depth && st.time_limit_ms > 0 && st.elapsed_ms() >= st.time_limit_ms) break;
        if (limits.nodes > 0 && st.nodes >= limits.nodes) break;
    }
}

} // namespace

SearchResult search_bestmove(Board& board, const NNUE& nnue, const SearchLimits& limits,
                             std::ostream* info) {
    // Syzygy: in a tablebase position, play the DTZ-optimal move directly. This
    // guarantees correct, progress-making endgame play (no 50-move-rule draws in
    // won endings) and instantly correct conversions.
    {
        Move tb_move;
        int tb_score;
        if (tb_probe_root_best(board, tb_move, tb_score)) {
            SearchResult r;
            r.best = tb_move;
            r.score = tb_score;
            r.depth = 1;
            if (info) {
                *info << "info depth 1 score cp " << tb_score
                      << " pv " << move_to_uci(tb_move) << " string syzygy tablebase\n";
                info->flush();
            }
            return r;
        }
    }

    int n_threads = std::max(1, limits.threads);
    std::vector<TTEntry> tt(TT_SIZE);              // one transposition table shared by all
    std::atomic<bool> stop{false};
    auto start = std::chrono::steady_clock::now();

    std::vector<SearchResult> results(n_threads);
    std::vector<std::thread> pool;

    // Lazy SMP: helper threads search the same position, diverging through the
    // shared TT; the main thread drives time and reports the chosen move.
    for (int i = 1; i < n_threads; ++i) {
        pool.emplace_back([&, i]() {
            Board b = board;
            SearchState st(nnue, limits, b, tt, stop, start);
            run_loop(st, b, limits, results[i], nullptr);
        });
    }
    {
        Board b = board;
        SearchState st(nnue, limits, b, tt, stop, start);
        run_loop(st, b, limits, results[0], info);
    }
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : pool) t.join();
    return results[0];
}

} // namespace owen
