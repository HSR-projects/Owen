# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
"""
evaluator.py — play a trained OwenNet against Stockfish over a UCI engine link
and report the score / estimated Elo gap.

Usage:
  python evaluator.py --weights ../weights/owen_latest.pt \
      --stockfish /usr/games/stockfish --games 20 --sf-elo 1350 --sims 200

Stockfish strength is throttled with UCI_LimitStrength/UCI_Elo (when supported)
or with `Skill Level`, so a weak fresh net has a meaningful opponent to climb against.
"""

from __future__ import annotations
import argparse
import math
import shutil

import numpy as np
import torch
import chess
import chess.engine

from model import OwenNet
from mcts import MCTS, select_move
import yaml


def load_net(weights: str, channels: int, blocks: int, device: str) -> OwenNet:
    net = OwenNet(channels, blocks).to(device)
    net.load_state_dict(torch.load(weights, map_location=device))
    net.eval()
    return net


def find_stockfish(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for name in ("stockfish", "/usr/games/stockfish", "/usr/local/bin/stockfish"):
        p = shutil.which(name) or (name if name.startswith("/") else None)
        if p and shutil.which(p) or (p and p.startswith("/")):
            return p
    return shutil.which("stockfish")


def elo_from_score(score: float, n: int) -> tuple[float, str]:
    """Logistic Elo estimate from a match score in [0,1]."""
    eps = 1e-4
    s = min(max(score, eps), 1 - eps)
    elo = -400.0 * math.log10(1.0 / s - 1.0)
    margin = 400.0 / math.log(10) * math.sqrt(max(s * (1 - s) / n, eps)) * 1.96
    return elo, f"±{margin:.0f} (95%)"


def play_match(net, engine, cfg, device, games, sims, sf_movetime):
    rng = np.random.default_rng(7)
    wins = draws = losses = 0
    for g in range(games):
        owen_white = (g % 2 == 0)
        board = chess.Board()
        mcts = MCTS(net, {**cfg, "dirichlet_eps": 0.0, "seed": 5000 + g}, device)
        while not board.is_game_over(claim_draw=True):
            if (board.turn == chess.WHITE) == owen_white:
                vc = mcts.run(board, sims, add_noise=False)
                board.push(select_move(vc, 0.0, rng))
            else:
                result = engine.play(board, chess.engine.Limit(time=sf_movetime))
                board.push(result.move)
        res = board.result(claim_draw=True)
        if res == "1/2-1/2":
            draws += 1
        elif (res == "1-0") == owen_white:
            wins += 1
        else:
            losses += 1
        print(f"  game {g+1}/{games}: {res} (Owen {'W' if owen_white else 'B'}) "
              f"| running W/D/L = {wins}/{draws}/{losses}")
    return wins, draws, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stockfish", default=None)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--sf-elo", type=int, default=1350, help="throttle Stockfish to this Elo")
    ap.add_argument("--sf-skill", type=int, default=None, help="alt: Skill Level 0-20")
    ap.add_argument("--sf-movetime", type=float, default=0.1)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu"
    net = load_net(args.weights, cfg["channels"], cfg["blocks"], device)

    sf_path = find_stockfish(args.stockfish)
    if not sf_path:
        print("Stockfish not found. Install it (e.g. `apt install stockfish` or download "
              "from stockfishchess.org) and pass --stockfish /path/to/stockfish.")
        return

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        if args.sf_skill is not None:
            engine.configure({"Skill Level": args.sf_skill})
            label = f"Skill {args.sf_skill}"
        else:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.sf_elo})
            label = f"~{args.sf_elo} Elo"
        print(f"Owen ({net.num_params():,} params) vs Stockfish [{label}] "
              f"| {args.games} games, {args.sims} sims/move")
        w, d, l = play_match(net, engine, cfg, device, args.games, args.sims, args.sf_movetime)
    finally:
        engine.quit()

    n = w + d + l
    score = (w + 0.5 * d) / n
    elo, margin = elo_from_score(score, n)
    print(f"\nResult: +{w} ={d} -{l}  score={score:.1%}")
    print(f"Estimated Elo vs this opponent: {elo:+.0f} {margin}")


if __name__ == "__main__":
    main()
