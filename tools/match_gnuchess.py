# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
"""Play Owen vs GNU Chess and print the game(s) + result for diagnosis."""
import sys, chess, chess.engine, chess.pgn

OWEN = "../engine/owen"
GNU = "gnuchess"

def play(owen_white, movetime, max_plies=200):
    board = chess.Board()
    owen = chess.engine.SimpleEngine.popen_uci(OWEN)
    gnu = chess.engine.SimpleEngine.popen_uci([GNU, "--uci"])
    gnu.configure({"OwnBook": False})  # fair: no opening book for either
    info_log = []
    try:
        limit = chess.engine.Limit(time=movetime)
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            owen_turn = (board.turn == chess.WHITE) == owen_white
            eng = owen if owen_turn else gnu
            res = eng.play(board, limit, info=chess.engine.INFO_SCORE)
            tag = "Owen" if owen_turn else "GNU "
            sc = res.info.get("score")
            info_log.append(f"{board.fullmove_number:>3}{'.' if board.turn else '...'} "
                            f"{tag} {board.san(res.move):7} score={sc}")
            board.push(res.move)
    finally:
        owen.quit(); gnu.quit()
    return board, info_log

def main():
    movetime = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    for owen_white in (True, False):
        board, log = play(owen_white, movetime)
        res = board.result(claim_draw=True)
        who = "Owen=White GNU=Black" if owen_white else "Owen=Black GNU=White"
        owen_pts = (res == "1-0") == owen_white if res != "1/2-1/2" else None
        verdict = "DRAW" if owen_pts is None else ("OWEN WINS" if owen_pts else "OWEN LOSES")
        print(f"\n===== {who} | movetime={movetime}s | result {res} -> {verdict} =====")
        for line in log:
            print(line)
        print("final FEN:", board.fen(), "| over:", board.outcome(claim_draw=True))

if __name__ == "__main__":
    main()
