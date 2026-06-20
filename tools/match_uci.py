# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run a simple match between Owen UCI and another engine such as GNU Chess."""

from __future__ import annotations

import argparse
import math
import os
import shutil

import chess
import chess.engine


def find_engine(explicit: str | None, names: tuple[str, ...]) -> str:
    if explicit:
        return explicit
    for name in names:
        path = shutil.which(name)
        if path:
            return path
        if name.startswith("/") and os.path.exists(name):
            return name
    raise SystemExit(f"engine not found; tried {', '.join(names)}")


def score_to_elo(score: float, games: int) -> str:
    eps = 1e-4
    s = min(max(score, eps), 1.0 - eps)
    elo = -400.0 * math.log10(1.0 / s - 1.0)
    margin = 400.0 / math.log(10) * math.sqrt(max(s * (1.0 - s) / games, eps)) * 1.96
    return f"{elo:+.0f} +/- {margin:.0f}"


def configure_engine(engine: chess.engine.SimpleEngine, options: dict[str, object]):
    supported = set(engine.options.keys())
    filtered = {k: v for k, v in options.items() if k in supported}
    if filtered:
        engine.configure(filtered)


def engine_command(path: str, protocol: str):
    if protocol == "uci" and "gnuchess" in os.path.basename(path):
        return [path, "--uci", "--quiet"]
    return path


def open_engine(path: str, protocol: str) -> chess.engine.SimpleEngine:
    command = engine_command(path, protocol)
    if protocol == "uci":
        return chess.engine.SimpleEngine.popen_uci(command, timeout=10.0)
    if protocol == "xboard":
        return chess.engine.SimpleEngine.popen_xboard(command, timeout=10.0)
    raise SystemExit(f"unknown protocol: {protocol}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owen", default="../engine/owen")
    ap.add_argument("--eval", default="../weights/owen.nnue")
    ap.add_argument("--opponent", default=None)
    ap.add_argument("--opponent-protocol", choices=("uci", "xboard", "auto"), default="auto")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--movetime", type=float, default=0.2)
    ap.add_argument("--opponent-movetime", type=float, default=None)
    args = ap.parse_args()

    opponent_path = find_engine(args.opponent, ("gnuchess", "/usr/games/gnuchess"))
    opponent_protocol = args.opponent_protocol
    if opponent_protocol == "auto":
        opponent_protocol = "uci"
    owen_path = os.path.abspath(args.owen)
    eval_path = os.path.abspath(args.eval)

    owen = open_engine(owen_path, "uci")
    opponent = open_engine(opponent_path, opponent_protocol)
    try:
        configure_engine(owen, {"EvalFile": eval_path})
        configure_engine(opponent, {"Book": False})
        opp_time = args.opponent_movetime if args.opponent_movetime is not None else args.movetime

        wins = draws = losses = 0
        for game in range(args.games):
            board = chess.Board()
            owen_white = (game % 2 == 0)
            while not board.is_game_over(claim_draw=True):
                engine = owen if (board.turn == chess.WHITE) == owen_white else opponent
                limit = chess.engine.Limit(time=args.movetime if engine is owen else opp_time)
                result = engine.play(board, limit)
                if result.move is None:
                    break
                board.push(result.move)

            res = board.result(claim_draw=True)
            if res == "1/2-1/2":
                draws += 1
            elif (res == "1-0") == owen_white:
                wins += 1
            else:
                losses += 1
            print(f"game {game + 1:02d}/{args.games}: {res} Owen {'W' if owen_white else 'B'} "
                  f"| W/D/L {wins}/{draws}/{losses}")
    finally:
        owen.quit()
        opponent.quit()

    score = (wins + 0.5 * draws) / args.games
    print(f"\nOwen vs {opponent_path}: +{wins} ={draws} -{losses} score={score:.1%}")
    print(f"Elo estimate vs opponent: {score_to_elo(score, args.games)}")


if __name__ == "__main__":
    main()
