// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later

#include "nnue.h"
#include "nnue_cuda.h"
#include "sfnnue/nnue.h"   // third-party HalfKP probe (see file header)
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace owen {

#ifndef OWEN_USE_CUDA
NNUECudaBackend::~NNUECudaBackend() = default;

bool NNUECudaBackend::available() const { return false; }

bool NNUECudaBackend::upload(int, const std::vector<float>&, const std::vector<float>&,
                             const std::vector<float>&, float, std::string* error) {
    if (error) *error = "CUDA backend was not compiled in";
    return false;
}

bool NNUECudaBackend::evaluate(const std::vector<int>&, float*, std::string* error) const {
    if (error) *error = "CUDA backend was not compiled in";
    return false;
}
#endif

namespace {


bool read_named_vector(std::istream& in, const std::string& expected, int count,
                       std::vector<float>& out, std::string* error) {
    std::string name;
    int got = 0;
    if (!(in >> name >> got) || name != expected || got != count) {
        if (error) *error = "bad NNUE tensor header for " + expected;
        return false;
    }
    out.resize(count);
    for (int i = 0; i < count; ++i) {
        if (!(in >> out[i])) {
            if (error) *error = "truncated NNUE tensor " + expected;
            return false;
        }
    }
    return true;
}

// PeSTO tuned evaluation: separate middlegame/endgame material values and
// piece-square tables (rank-8-first layout), combined via game phase.
constexpr int MG_VALUE[6] = {82, 337, 365, 477, 1025, 0};
constexpr int EG_VALUE[6] = {94, 281, 297, 512,  936, 0};

constexpr int MG_PST[6][64] = {
    {   0,   0,   0,   0,   0,   0,   0,   0,    // pawn
       98, 134,  61,  95,  68, 126,  34, -11,
       -6,   7,  26,  31,  65,  56,  25, -20,
      -14,  13,   6,  21,  23,  12,  17, -23,
      -27,  -2,  -5,  12,  17,   6,  10, -25,
      -26,  -4,  -4, -10,   3,   3,  33, -12,
      -35,  -1, -20, -23, -15,  24,  38, -22,
        0,   0,   0,   0,   0,   0,   0,   0 },
    { -167, -89, -34, -49,  61, -97, -15,-107,   // knight
       -73, -41,  72,  36,  23,  62,   7, -17,
       -47,  60,  37,  65,  84, 129,  73,  44,
        -9,  17,  19,  53,  37,  69,  18,  22,
       -13,   4,  16,  13,  28,  19,  21,  -8,
       -23,  -9,  12,  10,  19,  17,  25, -16,
       -29, -53, -12,  -3,  -1,  18, -14, -19,
      -105, -21, -58, -33, -17, -28, -19, -23 },
    {  -29,   4, -82, -37, -25, -42,   7,  -8,   // bishop
       -26,  16, -18, -13,  30,  59,  18, -47,
       -16,  37,  43,  40,  35,  50,  37,  -2,
        -4,   5,  19,  50,  37,  37,   7,  -2,
        -6,  13,  13,  26,  34,  12,  10,   4,
         0,  15,  15,  15,  14,  27,  18,  10,
         4,  15,  16,   0,   7,  21,  33,   1,
       -33,  -3, -14, -21, -13, -12, -39, -21 },
    {   32,  42,  32,  51,  63,   9,  31,  43,   // rook
        27,  32,  58,  62,  80,  67,  26,  44,
        -5,  19,  26,  36,  17,  45,  61,  16,
       -24, -11,   7,  26,  24,  35,  -8, -20,
       -36, -26, -12,  -1,   9,  -7,   6, -23,
       -45, -25, -16, -17,   3,   0,  -5, -33,
       -44, -16, -20,  -9,  -1,  11,  -6, -71,
       -19, -13,   1,  17,  16,   7, -37, -26 },
    {  -28,   0,  29,  12,  59,  44,  43,  45,   // queen
       -24, -39,  -5,   1, -16,  57,  28,  54,
       -13, -17,   7,   8,  29,  56,  47,  57,
       -27, -27, -16, -16,  -1,  17,  -2,   1,
        -9, -26,  -9, -10,  -2,  -4,   3,  -3,
       -14,   2, -11,  -2,  -5,   2,  14,   5,
       -35,  -8,  11,   2,   8,  15,  -3,   1,
        -1, -18,  -9,  10, -15, -25, -31, -50 },
    {  -65,  23,  16, -15, -56, -34,   2,  13,   // king
        29,  -1, -20,  -7,  -8,  -4, -38, -29,
        -9,  24,   2, -16, -20,   6,  22, -22,
       -17, -20, -12, -27, -30, -25, -14, -36,
       -49,  -1, -27, -39, -46, -44, -33, -51,
       -14, -14, -22, -46, -44, -30, -15, -27,
         1,   7,  -8, -64, -43, -16,   9,   8,
       -15,  36,  12, -54,   8, -28,  24,  14 },
};

constexpr int EG_PST[6][64] = {
    {   0,   0,   0,   0,   0,   0,   0,   0,    // pawn
      178, 173, 158, 134, 147, 132, 165, 187,
       94, 100,  85,  67,  56,  53,  82,  84,
       32,  24,  13,   5,  -2,   4,  17,  17,
       13,   9,  -3,  -7,  -7,  -8,   3,  -1,
        4,   7,  -6,   1,   0,  -5,  -1,  -8,
       13,   8,   8,  10,  13,   0,   2,  -7,
        0,   0,   0,   0,   0,   0,   0,   0 },
    {  -58, -38, -13, -28, -31, -27, -63, -99,   // knight
       -25,  -8, -25,  -2,  -9, -25, -24, -52,
       -24, -20,  10,   9,  -1,  -9, -19, -41,
       -17,   3,  22,  22,  22,  11,   8, -18,
       -18,  -6,  16,  25,  16,  17,   4, -18,
       -23,  -3,  -1,  15,  10,  -3, -20, -22,
       -42, -20, -10,  -5,  -2, -20, -23, -44,
       -29, -51, -23, -15, -22, -18, -50, -64 },
    {  -14, -21, -11,  -8,  -7,  -9, -17, -24,   // bishop
        -8,  -4,   7, -12,  -3, -13,  -4, -14,
         2,  -8,   0,  -1,  -2,   6,   0,   4,
        -3,   9,  12,   9,  14,  10,   3,   2,
        -6,   3,  13,  19,   7,  10,  -3,  -9,
       -12,  -3,   8,  10,  13,   3,  -7, -15,
       -14, -18,  -7,  -1,   4,  -9, -15, -27,
       -23,  -9, -23,  -5,  -9, -16,  -5, -17 },
    {   13,  10,  18,  15,  12,  12,   8,   5,   // rook
        11,  13,  13,  11,  -3,   3,   8,   3,
         7,   7,   7,   5,   4,  -3,  -5,  -3,
         4,   3,  13,   1,   2,   1,  -1,   2,
         3,   5,   8,   4,  -5,  -6,  -8, -11,
        -4,   0,  -5,  -1,  -7, -12,  -8, -16,
        -6,  -6,   0,   2,  -9,  -9, -11,  -3,
        -9,   2,   3,  -1,  -5, -13,   4, -20 },
    {   -9,  22,  22,  27,  27,  19,  10,  20,   // queen
       -17,  20,  32,  41,  58,  25,  30,   0,
       -20,   6,   9,  49,  47,  35,  19,   9,
         3,  22,  24,  45,  57,  40,  57,  36,
       -18,  28,  19,  47,  31,  34,  39,  23,
       -16, -27,  15,   6,   9,  17,  10,   5,
       -22, -23, -30, -16, -16, -23, -36, -32,
       -33, -28, -22, -43,  -5, -32, -20, -41 },
    {  -74, -35, -18, -18, -11,  15,   4, -17,   // king
       -12,  17,  14,  17,  17,  38,  23,  11,
        10,  17,  23,  15,  20,  45,  44,  13,
        -8,  22,  24,  27,  26,  33,  26,   3,
       -18,  -4,  21,  24,  27,  23,   9, -11,
       -19,  -3,  11,  21,  23,  16,   7,  -9,
       -27, -11,   4,  13,  14,   4,  -5, -17,
       -53, -34, -21, -11, -28, -14, -24, -43 },
};

// psq is the square from White's perspective (caller mirrors for Black); ^56
// converts a1=0 indexing to the rank-8-first PeSTO table layout.
int pesto_mg(PieceType pt, Square psq) { return MG_VALUE[pt] + MG_PST[pt][int(psq) ^ 56]; }
int pesto_eg(PieceType pt, Square psq) { return EG_VALUE[pt] + EG_PST[pt][int(psq) ^ 56]; }

int file_distance(int a, int b) {
    return std::abs(a - b);
}

bool has_pawn_on_file(const Board& b, Color c, int file) {
    for (int r = 0; r < 8; ++r) {
        Piece p = b.piece_on(make_square(file, r));
        if (p == make_piece(c, PAWN)) return true;
    }
    return false;
}

int pawn_structure(const Board& b, Color c) {
    int score = 0;
    int pawns_on_file[8] = {};
    for (int s = 0; s < SQUARE_NB; ++s) {
        Piece p = b.piece_on(Square(s));
        if (p == make_piece(c, PAWN)) pawns_on_file[file_of(Square(s))]++;
    }

    for (int f = 0; f < 8; ++f) {
        if (pawns_on_file[f] > 1) score -= 12 * (pawns_on_file[f] - 1);
        if (pawns_on_file[f] > 0) {
            bool isolated = (f == 0 || pawns_on_file[f - 1] == 0) &&
                            (f == 7 || pawns_on_file[f + 1] == 0);
            if (isolated) score -= 10;
        }
    }

    for (int s = 0; s < SQUARE_NB; ++s) {
        Square sq = Square(s);
        Piece p = b.piece_on(sq);
        if (p != make_piece(c, PAWN)) continue;

        int f = file_of(sq);
        int r = rank_of(sq);
        bool passed = true;
        for (int df = -1; df <= 1; ++df) {
            int ef = f + df;
            if (ef < 0 || ef > 7) continue;
            for (int er = 0; er < 8; ++er) {
                if ((c == WHITE && er <= r) || (c == BLACK && er >= r)) continue;
                if (b.piece_on(make_square(ef, er)) == make_piece(~c, PAWN)) passed = false;
            }
        }
        if (passed) {
            int advanced = c == WHITE ? r : 7 - r;
            score += 12 + advanced * advanced * 2;
        }
    }
    return score;
}

int king_safety(const Board& b, Color c) {
    Square k = b.king_square(c);
    int score = 0;
    int kf = file_of(k);
    int kr = rank_of(k);

    for (int df = -1; df <= 1; ++df) {
        int f = kf + df;
        if (f < 0 || f > 7) continue;
        int shield_rank = c == WHITE ? kr + 1 : kr - 1;
        if (shield_rank >= 0 && shield_rank < 8 &&
            b.piece_on(make_square(f, shield_rank)) == make_piece(c, PAWN)) {
            score += 10;
        }
    }

    if (!has_pawn_on_file(b, c, kf)) score -= 15;
    for (int f = 0; f < 8; ++f) {
        if (!has_pawn_on_file(b, c, f) && file_distance(f, kf) <= 1) score -= 8;
    }
    return score;
}

// Endgame king table: centralisation is good once queens/rooks are gone.
// Game phase: 24 = full material (opening/middlegame), 0 = bare kings (endgame).
int game_phase(const Board& b) {
    int phase = 1 * popcount(b.pieces(WHITE, KNIGHT) | b.pieces(BLACK, KNIGHT))
              + 1 * popcount(b.pieces(WHITE, BISHOP) | b.pieces(BLACK, BISHOP))
              + 2 * popcount(b.pieces(WHITE, ROOK)   | b.pieces(BLACK, ROOK))
              + 4 * popcount(b.pieces(WHITE, QUEEN)  | b.pieces(BLACK, QUEEN));
    return std::min(phase, 24);
}

// Squares attacked by a side's pawns.
Bitboard pawn_attacks_bb(const Board& b, Color c) {
    Bitboard p = b.pieces(c, PAWN);
    if (c == WHITE) return ((p & ~FILE_A) << 7) | ((p & ~FILE_H) << 9);
    return ((p & ~FILE_H) >> 7) | ((p & ~FILE_A) >> 9);
}

// Piece mobility — reachable squares not blocked by our own pieces. For minor
// pieces, squares controlled by enemy pawns are excluded (a knight/bishop can't
// safely stand there); this is what gives knights their worth in closed games.
int mobility(const Board& b, Color c, Bitboard enemy_pawn_att) {
    Bitboard own = b.color_bb(c);
    Bitboard occ = b.occupied();
    Bitboard safe = ~own & ~enemy_pawn_att;
    int score = 0;
    Bitboard bb;
    bb = b.pieces(c, KNIGHT);
    while (bb) { Square s = pop_lsb(bb); score += 4 * popcount(KnightAttacks[s] & safe); }
    bb = b.pieces(c, BISHOP);
    while (bb) { Square s = pop_lsb(bb); score += 3 * popcount(bishop_attacks(s, occ) & safe); }
    bb = b.pieces(c, ROOK);
    while (bb) { Square s = pop_lsb(bb); score += 2 * popcount(rook_attacks(s, occ) & ~own); }
    bb = b.pieces(c, QUEEN);
    while (bb) { Square s = pop_lsb(bb); score += 1 * popcount(queen_attacks(s, occ) & ~own); }
    return score;
}

// Outposts: an advanced minor defended by a friendly pawn that no enemy pawn can
// ever attack is a powerful, stable piece — especially knights in closed games.
int outposts(const Board& b, Color c) {
    int score = 0;
    Bitboard own_pawns = b.pieces(c, PAWN);
    Bitboard enemy_pawns = b.pieces(~c, PAWN);
    Bitboard minors = b.pieces(c, KNIGHT) | b.pieces(c, BISHOP);
    while (minors) {
        Square s = pop_lsb(minors);
        int r = rank_of(s), f = file_of(s);
        int rel = c == WHITE ? r : 7 - r;
        if (rel < 3 || rel > 5) continue;                       // must be advanced
        Bitboard defenders = c == WHITE ? PawnAttacks[BLACK][s] : PawnAttacks[WHITE][s];
        if (!(defenders & own_pawns)) continue;                 // must be pawn-defended

        bool attackable = false;
        for (int df = -1; df <= 1 && !attackable; df += 2) {
            int ef = f + df;
            if (ef < 0 || ef > 7) continue;
            int lo = c == WHITE ? r + 1 : 0;
            int hi = c == WHITE ? 7 : r - 1;
            for (int rr = lo; rr <= hi; ++rr)
                if (enemy_pawns & square_bb(make_square(ef, rr))) { attackable = true; break; }
        }
        if (!attackable)
            score += (b.piece_on(s) == make_piece(c, KNIGHT)) ? 28 : 18;
    }
    return score;
}

// Rook on (semi-)open files.
int rook_files(const Board& b, Color c) {
    int score = 0;
    Bitboard bb = b.pieces(c, ROOK);
    while (bb) {
        Square s = pop_lsb(bb);
        int f = file_of(s);
        bool own_pawn = has_pawn_on_file(b, c, f);
        bool opp_pawn = has_pawn_on_file(b, ~c, f);
        if (!own_pawn && !opp_pawn) score += 18;      // fully open
        else if (!own_pawn)         score += 9;       // semi-open
    }
    return score;
}

int handcrafted_evaluate(const Board& b) {
    int phase = game_phase(b);
    int mg = 0, eg = 0;                                // white-positive, tapered
    int bishops[COLOR_NB] = {};

    for (int s = 0; s < SQUARE_NB; ++s) {
        Square sq = Square(s);
        Piece p = b.piece_on(sq);
        if (p == NO_PIECE) continue;

        Color c = color_of(p);
        PieceType pt = type_of(p);
        Square psq = c == WHITE ? sq : make_square(file_of(sq), 7 - rank_of(sq));
        int sgn = c == WHITE ? 1 : -1;
        mg += sgn * pesto_mg(pt, psq);
        eg += sgn * pesto_eg(pt, psq);
        if (pt == BISHOP) bishops[c]++;
    }

    int score = (mg * phase + eg * (24 - phase)) / 24;   // tapered material + PST

    if (bishops[WHITE] >= 2) score += 30;
    if (bishops[BLACK] >= 2) score -= 30;

    Bitboard wpa = pawn_attacks_bb(b, WHITE), bpa = pawn_attacks_bb(b, BLACK);
    score += mobility(b, WHITE, bpa) - mobility(b, BLACK, wpa);
    score += outposts(b, WHITE) - outposts(b, BLACK);
    score += rook_files(b, WHITE) - rook_files(b, BLACK);
    score += pawn_structure(b, WHITE) - pawn_structure(b, BLACK);
    // King safety only matters while there is attacking material on the board.
    score += (king_safety(b, WHITE) - king_safety(b, BLACK)) * phase / 24;

    score += b.side_to_move() == WHITE ? 12 : -12;    // tempo
    return b.side_to_move() == WHITE ? score : -score;
}

} // namespace

NNUE::NNUE() : cuda_(std::make_unique<NNUECudaBackend>()) {}

NNUE::~NNUE() = default;

bool NNUE::load(const std::string& path, std::string* error) {
    // HalfKP net? Detect by the 4-byte version magic and hand off to the
    // (third-party) probe in sfnnue/. Owen then evaluates via the NNUE network.
    {
        std::ifstream bin(path, std::ios::binary);
        if (!bin) {
            if (error) *error = "could not open " + path;
            return false;
        }
        uint32_t ver = 0;
        bin.read(reinterpret_cast<char*>(&ver), sizeof(ver));
        if (bin && ver == 0x7AF32F16u) {       // NNUE HalfKP version
            nnue_init(path.c_str());
            sf_active_ = true;
            loaded_ = true;
            return true;
        }
    }

    std::ifstream in(path);
    if (!in) {
        if (error) *error = "could not open " + path;
        return false;
    }

    std::string magic;
    int version = 0;
    if (!(in >> magic >> version) || magic != "OWEN_NNUE" || version != 1) {
        if (error) *error = "unsupported NNUE file";
        return false;
    }

    std::string input_key, hidden_key;
    int input = 0;
    if (!(in >> input_key >> input >> hidden_key >> hidden_) ||
        input_key != "input" || hidden_key != "hidden" || input != INPUT_SIZE || hidden_ <= 0) {
        if (error) *error = "bad NNUE shape";
        return false;
    }

    if (!read_named_vector(in, "w1", hidden_ * INPUT_SIZE, w1_, error)) return false;
    if (!read_named_vector(in, "b1", hidden_, b1_, error)) return false;
    if (!read_named_vector(in, "w2", hidden_, w2_, error)) return false;

    std::string b2_name;
    int b2_count = 0;
    if (!(in >> b2_name >> b2_count) || b2_name != "b2" || b2_count != 1 || !(in >> b2_)) {
        if (error) *error = "bad NNUE output bias";
        return false;
    }

    if (gpu_enabled_ && !cuda_->upload(hidden_, w1_, b1_, w2_, b2_, error)) {
        gpu_enabled_ = false;
        return false;
    }
    loaded_ = true;
    return true;
}

// Evaluate via the HalfKP network (reentrant: local accumulator per call).
static int sf_evaluate(const Board& b) {
    // Owen Piece (WP..BK = 0..11) -> nnue-probe codes (wking=1 .. bpawn=12).
    static const int code[12] = {6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7};
    int pieces[33], squares[33];
    pieces[0] = 1; squares[0] = int(b.king_square(WHITE));   // white king first
    pieces[1] = 7; squares[1] = int(b.king_square(BLACK));   // black king second
    int n = 2;
    for (int s = 0; s < SQUARE_NB; ++s) {
        Piece p = b.piece_on(Square(s));
        if (p == NO_PIECE || type_of(p) == KING) continue;
        pieces[n] = code[p];
        squares[n] = s;
        ++n;
    }
    pieces[n] = 0;   // terminator
    int player = b.side_to_move() == WHITE ? 0 : 1;
    return nnue_evaluate(player, pieces, squares);   // cp relative to side to move
}

// Decay scores toward 0 as the fifty-move counter advances, so the engine is
// rewarded for making progress (a pawn move or capture resets the clock) instead of
// shuffling in a won position until a 50-move-rule draw.
static int scale_rule50(int v, int halfmove) {
    int hm = halfmove < 0 ? 0 : (halfmove > 100 ? 100 : halfmove);
    return v * (100 - hm / 2) / 100;   // factor 1.00 at 0 plies -> 0.50 at 100
}

int NNUE::evaluate(const Board& b) const {
    // Robustness: a side missing its king (only reachable from illegal positions)
    // is simply lost — return without touching the king-dependent evaluators.
    bool wk = b.pieces(WHITE, KING) != 0, bk = b.pieces(BLACK, KING) != 0;
    if (!wk || !bk) {
        int s = (wk ? 1 : 0) - (bk ? 1 : 0);
        return (b.side_to_move() == WHITE ? s : -s) * 10000;
    }
    int v = sf_active_ ? sf_evaluate(b)   // HalfKP NNUE
                       : material_evaluate(b);   // handcrafted fallback
    return scale_rule50(v, b.halfmove());
}

bool NNUE::set_gpu_enabled(bool enabled, std::string* error) {
    if (!enabled) {
        gpu_enabled_ = false;
        return true;
    }
    if (!cuda_->available()) {
        if (error) *error = "CUDA backend is unavailable";
        return false;
    }
    if (loaded_ && !cuda_->upload(hidden_, w1_, b1_, w2_, b2_, error)) return false;
    gpu_enabled_ = true;
    return true;
}

bool NNUE::gpu_available() const {
    return cuda_->available();
}

const char* NNUE::backend_name() const {
    if (sf_active_) return "sf-nnue";
    return gpu_enabled_ ? "cuda" : "cpu";
}

int material_evaluate(const Board& b) {
    return std::clamp(handcrafted_evaluate(b), -10000, 10000);
}

int lazy_evaluate(const Board& b) {
    int phase = game_phase(b);
    int mg = 0, eg = 0;
    for (int s = 0; s < SQUARE_NB; ++s) {
        Square sq = Square(s);
        Piece p = b.piece_on(sq);
        if (p == NO_PIECE) continue;
        Color c = color_of(p);
        PieceType pt = type_of(p);
        Square psq = c == WHITE ? sq : make_square(file_of(sq), 7 - rank_of(sq));
        int sgn = c == WHITE ? 1 : -1;
        mg += sgn * pesto_mg(pt, psq);
        eg += sgn * pesto_eg(pt, psq);
    }
    int score = (mg * phase + eg * (24 - phase)) / 24;
    score += b.side_to_move() == WHITE ? 12 : -12;
    int v = std::clamp(b.side_to_move() == WHITE ? score : -score, -10000, 10000);
    return scale_rule50(v, b.halfmove());
}

} // namespace owen
