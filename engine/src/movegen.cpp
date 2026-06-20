// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later
//
// movegen.cpp — pseudo-legal generation (into a raw buffer) + legality filtering.

#include "movegen.h"

namespace owen {

namespace {

inline void add_promotions(Move*& m, Square from, Square to) {
    *m++ = make_move(from, to, MT_PROMOTION, QUEEN);
    *m++ = make_move(from, to, MT_PROMOTION, ROOK);
    *m++ = make_move(from, to, MT_PROMOTION, BISHOP);
    *m++ = make_move(from, to, MT_PROMOTION, KNIGHT);
}

void gen_pawns(const Board& b, Color us, Move*& m) {
    Color them = ~us;
    Bitboard occ   = b.occupied();
    Bitboard empty = ~occ;
    Bitboard enemy = b.color_bb(them);
    Bitboard pawns = b.pieces(us, PAWN);
    Bitboard ep    = b.ep_square() != SQ_NONE ? square_bb(b.ep_square()) : 0;

    int forward    = (us == WHITE) ? 8 : -8;
    int startRank  = (us == WHITE) ? 1 : 6;
    int promoRank  = (us == WHITE) ? 6 : 1;

    while (pawns) {
        Square from = pop_lsb(pawns);
        int r = rank_of(from);

        Square one = Square(int(from) + forward);
        if (empty & square_bb(one)) {
            if (r == promoRank) {
                add_promotions(m, from, one);
            } else {
                *m++ = make_move(from, one);
                if (r == startRank) {
                    Square two = Square(int(one) + forward);
                    if (empty & square_bb(two)) *m++ = make_move(from, two);
                }
            }
        }

        Bitboard atts = PawnAttacks[us][from];
        Bitboard caps = atts & enemy;
        while (caps) {
            Square t = pop_lsb(caps);
            if (r == promoRank) add_promotions(m, from, t);
            else *m++ = make_move(from, t);
        }

        if (ep && (atts & ep))
            *m++ = make_move(from, b.ep_square(), MT_ENPASSANT);
    }
}

void gen_piece_moves(const Board& b, Color us, Move*& m) {
    Bitboard occ = b.occupied();
    Bitboard own = b.color_bb(us);

    Bitboard knights = b.pieces(us, KNIGHT);
    while (knights) {
        Square from = pop_lsb(knights);
        Bitboard t = KnightAttacks[from] & ~own;
        while (t) *m++ = make_move(from, pop_lsb(t));
    }
    Bitboard bishops = b.pieces(us, BISHOP);
    while (bishops) {
        Square from = pop_lsb(bishops);
        Bitboard t = bishop_attacks(from, occ) & ~own;
        while (t) *m++ = make_move(from, pop_lsb(t));
    }
    Bitboard rooks = b.pieces(us, ROOK);
    while (rooks) {
        Square from = pop_lsb(rooks);
        Bitboard t = rook_attacks(from, occ) & ~own;
        while (t) *m++ = make_move(from, pop_lsb(t));
    }
    Bitboard queens = b.pieces(us, QUEEN);
    while (queens) {
        Square from = pop_lsb(queens);
        Bitboard t = queen_attacks(from, occ) & ~own;
        while (t) *m++ = make_move(from, pop_lsb(t));
    }
    Bitboard king = b.pieces(us, KING);
    Square ksq = pop_lsb(king);
    Bitboard kt = KingAttacks[ksq] & ~own;
    while (kt) *m++ = make_move(ksq, pop_lsb(kt));
}

void gen_castling(const Board& b, Color us, Move*& m) {
    Color them = ~us;
    Bitboard occ = b.occupied();
    uint8_t cr = b.castling();

    if (us == WHITE) {
        if ((cr & WHITE_OO)
            && !(occ & (square_bb(F1) | square_bb(G1)))
            && !b.is_square_attacked(E1, them) && !b.is_square_attacked(F1, them)
            && !b.is_square_attacked(G1, them))
            *m++ = make_move(E1, G1, MT_CASTLING);
        if ((cr & WHITE_OOO)
            && !(occ & (square_bb(B1) | square_bb(C1) | square_bb(D1)))
            && !b.is_square_attacked(E1, them) && !b.is_square_attacked(D1, them)
            && !b.is_square_attacked(C1, them))
            *m++ = make_move(E1, C1, MT_CASTLING);
    } else {
        if ((cr & BLACK_OO)
            && !(occ & (square_bb(F8) | square_bb(G8)))
            && !b.is_square_attacked(E8, them) && !b.is_square_attacked(F8, them)
            && !b.is_square_attacked(G8, them))
            *m++ = make_move(E8, G8, MT_CASTLING);
        if ((cr & BLACK_OOO)
            && !(occ & (square_bb(B8) | square_bb(C8) | square_bb(D8)))
            && !b.is_square_attacked(E8, them) && !b.is_square_attacked(D8, them)
            && !b.is_square_attacked(C8, them))
            *m++ = make_move(E8, C8, MT_CASTLING);
    }
}

} // namespace

int generate_pseudo_legal(const Board& b, Move* moves) {
    Color us = b.side_to_move();
    Move* m = moves;
    gen_pawns(b, us, m);
    gen_piece_moves(b, us, m);
    gen_castling(b, us, m);
    return int(m - moves);
}

void generate_pseudo_legal(const Board& b, MoveList& out) {
    Move buf[MAX_MOVES];
    int n = generate_pseudo_legal(b, buf);
    out.insert(out.end(), buf, buf + n);
}

void generate_legal(const Board& b, MoveList& out) {
    Move buf[MAX_MOVES];
    int n = generate_pseudo_legal(b, buf);

    Color us = b.side_to_move();
    Board tmp = b;  // mutable copy for make/undo legality test (only used off the hot path)
    for (int i = 0; i < n; ++i) {
        tmp.make_move(buf[i]);
        if (!tmp.is_in_check(us)) out.push_back(buf[i]);
        tmp.undo_move();
    }
}

} // namespace owen
