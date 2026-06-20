// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later
//
// board.cpp — Zobrist init, FEN, make/undo, and attack queries.

#include "board.h"
#include <algorithm>
#include <random>
#include <sstream>
#include <stdexcept>

namespace owen {

// ── Zobrist ──────────────────────────────────────────────────────────────────
namespace Zobrist {
    uint64_t psq[PIECE_NB][SQUARE_NB];
    uint64_t castling[16];
    uint64_t ep_file[8];
    uint64_t side;

    void init() {
        std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);  // fixed seed → reproducible
        for (int p = 0; p < PIECE_NB; ++p)
            for (int s = 0; s < SQUARE_NB; ++s) psq[p][s] = rng();
        for (int i = 0; i < 16; ++i) castling[i] = rng();
        for (int i = 0; i < 8; ++i) ep_file[i] = rng();
        side = rng();
    }
}

// Castling-rights mask applied to from/to squares (AND into castling_).
static uint8_t CastleMask[SQUARE_NB];
static bool castle_mask_ready = false;

static void init_castle_mask() {
    for (int s = 0; s < 64; ++s) CastleMask[s] = ANY_CASTLING;
    CastleMask[A1] &= ~WHITE_OOO; CastleMask[H1] &= ~WHITE_OO;
    CastleMask[E1] &= ~(WHITE_OO | WHITE_OOO);
    CastleMask[A8] &= ~BLACK_OOO; CastleMask[H8] &= ~BLACK_OO;
    CastleMask[E8] &= ~(BLACK_OO | BLACK_OOO);
    castle_mask_ready = true;
}

// ── Low-level piece manipulation (keeps mailbox + hash in sync) ───────────────
void Board::put_piece(Piece p, Square s) {
    Bitboard bb = square_bb(s);
    pieces_[p] |= bb;
    side_[color_of(p)] |= bb;
    occupied_ |= bb;
    mailbox_[s] = p;
    hash_ ^= Zobrist::psq[p][s];
}
void Board::remove_piece(Square s) {
    Piece p = mailbox_[s];
    Bitboard bb = square_bb(s);
    pieces_[p] ^= bb;
    side_[color_of(p)] ^= bb;
    occupied_ ^= bb;
    mailbox_[s] = NO_PIECE;
    hash_ ^= Zobrist::psq[p][s];
}
void Board::move_piece(Square from, Square to) {
    Piece p = mailbox_[from];
    Bitboard ft = square_bb(from) | square_bb(to);
    pieces_[p] ^= ft;
    side_[color_of(p)] ^= ft;
    occupied_ ^= ft;
    mailbox_[from] = NO_PIECE;
    mailbox_[to] = p;
    hash_ ^= Zobrist::psq[p][from] ^ Zobrist::psq[p][to];
}
void Board::recompute_occupancy() {
    side_[WHITE] = side_[BLACK] = 0;
    for (int p = WP; p <= WK; ++p) side_[WHITE] |= pieces_[p];
    for (int p = BP; p <= BK; ++p) side_[BLACK] |= pieces_[p];
    occupied_ = side_[WHITE] | side_[BLACK];
}
void Board::clear() {
    for (int p = 0; p < PIECE_NB; ++p) pieces_[p] = 0;
    for (int s = 0; s < SQUARE_NB; ++s) mailbox_[s] = NO_PIECE;
    side_[WHITE] = side_[BLACK] = occupied_ = 0;
    stm_ = WHITE; castling_ = 0; ep_ = SQ_NONE;
    halfmove_ = 0; ply_ = 0; hash_ = 0;
    history_.clear();
}

// ── Setup ────────────────────────────────────────────────────────────────────
void Board::set_startpos() {
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Board::set_fen(const std::string& fen) {
    if (!castle_mask_ready) init_castle_mask();
    clear();

    std::istringstream ss(fen);
    std::string placement, stm, castle, ep;
    int halfmove = 0, fullmove = 1;
    ss >> placement >> stm >> castle >> ep >> halfmove >> fullmove;

    int rank = 7, file = 0;
    for (char c : placement) {
        if (c == '/') { rank--; file = 0; continue; }
        if (c >= '1' && c <= '8') { file += c - '0'; continue; }
        Piece p = NO_PIECE;
        switch (c) {
            case 'P': p = WP; break; case 'N': p = WN; break; case 'B': p = WB; break;
            case 'R': p = WR; break; case 'Q': p = WQ; break; case 'K': p = WK; break;
            case 'p': p = BP; break; case 'n': p = BN; break; case 'b': p = BB; break;
            case 'r': p = BR; break; case 'q': p = BQ; break; case 'k': p = BK; break;
            default: throw std::runtime_error("bad FEN piece");
        }
        put_piece(p, make_square(file, rank));
        file++;
    }

    stm_ = (stm == "w") ? WHITE : BLACK;
    if (stm_ == BLACK) hash_ ^= Zobrist::side;

    castling_ = 0;
    for (char c : castle) {
        switch (c) {
            case 'K': castling_ |= WHITE_OO;  break;
            case 'Q': castling_ |= WHITE_OOO; break;
            case 'k': castling_ |= BLACK_OO;  break;
            case 'q': castling_ |= BLACK_OOO; break;
            default: break;
        }
    }
    hash_ ^= Zobrist::castling[castling_];

    if (ep != "-" && ep.size() == 2) {
        ep_ = make_square(ep[0] - 'a', ep[1] - '1');
        hash_ ^= Zobrist::ep_file[file_of(ep_)];
    }

    halfmove_ = uint16_t(halfmove);
    ply_ = (fullmove - 1) * 2 + (stm_ == BLACK ? 1 : 0);
    recompute_occupancy();
}

std::string Board::fen() const {
    std::ostringstream os;
    const char syms[] = "PNBRQKpnbrqk";
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = mailbox_[make_square(f, r)];
            if (p == NO_PIECE) { empty++; continue; }
            if (empty) { os << empty; empty = 0; }
            os << syms[p];
        }
        if (empty) os << empty;
        if (r) os << '/';
    }
    os << (stm_ == WHITE ? " w " : " b ");
    std::string c;
    if (castling_ & WHITE_OO)  c += 'K';
    if (castling_ & WHITE_OOO) c += 'Q';
    if (castling_ & BLACK_OO)  c += 'k';
    if (castling_ & BLACK_OOO) c += 'q';
    os << (c.empty() ? "-" : c) << ' ';
    if (ep_ == SQ_NONE) os << '-';
    else { os << char('a' + file_of(ep_)) << char('1' + rank_of(ep_)); }
    os << ' ' << int(halfmove_) << ' ' << (ply_ / 2 + 1);
    return os.str();
}

// ── Attack query ─────────────────────────────────────────────────────────────
bool Board::is_square_attacked(Square s, Color by) const {
    if (PawnAttacks[~by][s] & pieces_[make_piece(by, PAWN)]) return true;
    if (KnightAttacks[s]    & pieces_[make_piece(by, KNIGHT)]) return true;
    if (KingAttacks[s]      & pieces_[make_piece(by, KING)]) return true;
    Bitboard bq = pieces_[make_piece(by, BISHOP)] | pieces_[make_piece(by, QUEEN)];
    if (bishop_attacks(s, occupied_) & bq) return true;
    Bitboard rq = pieces_[make_piece(by, ROOK)] | pieces_[make_piece(by, QUEEN)];
    if (rook_attacks(s, occupied_) & rq) return true;
    return false;
}

// ── Make / undo ────────────────────────────────────────────────────────────────
static inline int iabs(int x) { return x < 0 ? -x : x; }

static void castle_rook_squares(Square kingTo, Square& rfrom, Square& rto) {
    switch (kingTo) {
        case G1: rfrom = H1; rto = F1; break;
        case C1: rfrom = A1; rto = D1; break;
        case G8: rfrom = H8; rto = F8; break;
        case C8: rfrom = A8; rto = D8; break;
        default: rfrom = rto = SQ_NONE; break;
    }
}

void Board::make_move(Move m) {
    UndoInfo u;
    u.move = m;
    u.captured = NO_PIECE;
    u.captured_sq = SQ_NONE;
    u.castling = castling_;
    u.ep_square = ep_;
    u.halfmove = halfmove_;
    u.hash = hash_;

    Color us = stm_, them = ~us;
    Square from = from_sq(m), to = to_sq(m);
    MoveType mt = type_of_move(m);
    Piece pc = mailbox_[from];

    if (ep_ != SQ_NONE) hash_ ^= Zobrist::ep_file[file_of(ep_)];
    ep_ = SQ_NONE;

    // Captures.
    if (mt == MT_ENPASSANT) {
        Square capsq = Square(int(to) + (us == WHITE ? -8 : 8));
        u.captured = mailbox_[capsq];
        u.captured_sq = capsq;
        remove_piece(capsq);
    } else if (mailbox_[to] != NO_PIECE) {
        u.captured = mailbox_[to];
        u.captured_sq = to;
        remove_piece(to);
    }

    // Move the moving piece (promotion replaces the pawn with the promo piece).
    if (mt == MT_PROMOTION) {
        remove_piece(from);
        put_piece(make_piece(us, promo_type(m)), to);
    } else {
        move_piece(from, to);
    }

    if (mt == MT_CASTLING) {
        Square rfrom, rto;
        castle_rook_squares(to, rfrom, rto);
        move_piece(rfrom, rto);
    }

    // Castling rights.
    hash_ ^= Zobrist::castling[castling_];
    castling_ &= CastleMask[from];
    castling_ &= CastleMask[to];
    hash_ ^= Zobrist::castling[castling_];

    // Double pawn push → set en-passant target.
    if (type_of(pc) == PAWN && iabs(int(to) - int(from)) == 16) {
        Square eps = Square((int(from) + int(to)) / 2);
        ep_ = eps;
        hash_ ^= Zobrist::ep_file[file_of(eps)];
    }

    halfmove_ = (type_of(pc) == PAWN || u.captured != NO_PIECE) ? 0 : halfmove_ + 1;

    stm_ = them;
    hash_ ^= Zobrist::side;
    ply_++;

    history_.push_back(u);   // occupancy kept incrementally by put/remove/move_piece
}

void Board::undo_move() {
    UndoInfo u = history_.back();
    history_.pop_back();

    stm_ = ~stm_;                  // restore the side that moved
    Color us = stm_;
    Move m = u.move;
    Square from = from_sq(m), to = to_sq(m);
    MoveType mt = type_of_move(m);

    if (mt == MT_PROMOTION) {
        remove_piece(to);
        put_piece(make_piece(us, PAWN), from);
    } else {
        move_piece(to, from);
    }

    if (mt == MT_CASTLING) {
        Square rfrom, rto;
        castle_rook_squares(to, rfrom, rto);
        move_piece(rto, rfrom);
    }

    if (u.captured != NO_PIECE)
        put_piece(u.captured, u.captured_sq);

    // Restore irreversible state exactly (hash restored, not recomputed).
    castling_ = u.castling;
    ep_ = u.ep_square;
    halfmove_ = u.halfmove;
    hash_ = u.hash;
    ply_--;   // occupancy kept incrementally by put/remove/move_piece
}

bool Board::is_repetition() const {
    int n = int(history_.size());
    int limit = std::min(n, int(halfmove_));   // can't repeat past an irreversible move
    // Same side to move recurs every 2 plies; history_[n-i].hash holds prior positions.
    for (int i = 2; i <= limit; i += 2) {
        if (history_[n - i].hash == hash_) return true;
    }
    return false;
}

void Board::make_null_move() {
    UndoInfo u;
    u.move = MOVE_NONE;
    u.captured = NO_PIECE;
    u.captured_sq = SQ_NONE;
    u.castling = castling_;
    u.ep_square = ep_;
    u.halfmove = halfmove_;
    u.hash = hash_;

    if (ep_ != SQ_NONE) hash_ ^= Zobrist::ep_file[file_of(ep_)];
    ep_ = SQ_NONE;
    stm_ = ~stm_;
    hash_ ^= Zobrist::side;
    ply_++;
    history_.push_back(u);
}

void Board::undo_null_move() {
    UndoInfo u = history_.back();
    history_.pop_back();
    stm_ = ~stm_;
    castling_ = u.castling;
    ep_ = u.ep_square;
    halfmove_ = u.halfmove;
    hash_ = u.hash;
    ply_--;
}

std::string Board::to_string() const {
    const char syms[] = "PNBRQKpnbrqk";
    std::ostringstream os;
    os << "  +-----------------+\n";
    for (int r = 7; r >= 0; --r) {
        os << (r + 1) << " | ";
        for (int f = 0; f < 8; ++f) {
            Piece p = mailbox_[make_square(f, r)];
            os << (p == NO_PIECE ? '.' : syms[p]) << ' ';
        }
        os << "|\n";
    }
    os << "  +-----------------+\n    a b c d e f g h\n";
    os << "FEN: " << fen() << '\n';
    return os.str();
}

} // namespace owen
