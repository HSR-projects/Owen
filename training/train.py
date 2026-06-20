# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
"""
train.py — the self-play reinforcement-learning loop.

Each iteration:
  1. generate self-play games with the current network (MCTS),
  2. add the samples to a FIFO replay buffer,
  3. train the network for a number of mini-batch steps,
  4. periodically gate the candidate against the current best (arena match) and
     promote it only if it wins clearly.

Run:  python train.py --config config.yaml
All randomness is seeded for reproducibility.
"""

from __future__ import annotations
import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import chess

from model import OwenNet
from nnue import NNUENet, export_nnue, states_to_nnue_features, wdl_to_scalar
from mcts import MCTS, select_move
from selfplay import play_game
from replay_buffer import ReplayBuffer


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss(net, nnue, states, policies, wdl, device, nnue_weight: float):
    states = torch.from_numpy(states).to(device)
    policies = torch.from_numpy(policies).to(device)
    wdl = torch.from_numpy(wdl).to(device)
    p_logits, v_logits = net(states)
    logp = F.log_softmax(p_logits, dim=1)
    policy_loss = -(policies * logp).sum(dim=1).mean()
    value_loss = F.cross_entropy(v_logits, wdl)
    nnue_loss = torch.zeros((), device=device)
    if nnue is not None and nnue_weight > 0.0:
        nnue_pred = nnue(states_to_nnue_features(states))
        nnue_loss = F.mse_loss(nnue_pred, wdl_to_scalar(wdl))
    return policy_loss + value_loss + nnue_weight * nnue_loss, policy_loss.item(), value_loss.item(), nnue_loss.item()


@torch.no_grad()
def arena(net_a, net_b, cfg, device, games: int) -> float:
    """Play `games` between net_a and net_b (alternating colors). Return net_a score."""
    acfg = dict(cfg); acfg["dirichlet_eps"] = 0.0           # deterministic-ish eval
    sims = cfg.get("eval_simulations", max(16, cfg.get("simulations", 100) // 2))
    rng = np.random.default_rng(12345)
    score = 0.0
    for g in range(games):
        a_white = (g % 2 == 0)
        board = chess.Board()
        m_a = MCTS(net_a, {**acfg, "seed": 1000 + g}, device)
        m_b = MCTS(net_b, {**acfg, "seed": 2000 + g}, device)
        while not board.is_game_over(claim_draw=True) and board.fullmove_number < cfg.get("max_moves", 200):
            mover = m_a if (board.turn == chess.WHITE) == a_white else m_b
            vc = mover.run(board, sims, add_noise=False)
            board.push(select_move(vc, 0.0, rng))
        res = board.result(claim_draw=True)
        if res == "1/2-1/2":
            score += 0.5
        elif (res == "1-0") == a_white:
            score += 1.0
    return score / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seeds(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    os.makedirs(cfg.get("weights_dir", "../weights"), exist_ok=True)

    net = OwenNet(cfg["channels"], cfg["blocks"]).to(device)
    print(f"OwenNet: {net.num_params():,} params | device={device} | policy moves={net.p_fc.out_features}")

    nnue = None
    nnue_weight = float(cfg.get("nnue_loss_weight", 0.25))
    if cfg.get("nnue_enabled", True):
        nnue = NNUENet(int(cfg.get("nnue_hidden", 128))).to(device)
        print(f"NNUE: {nnue.num_params():,} params | training on {device}")

    params = list(net.parameters()) + ([] if nnue is None else list(nnue.parameters()))
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("l2", 1e-4))

    if cfg.get("resume") and os.path.exists(cfg["resume"]):
        net.load_state_dict(torch.load(cfg["resume"], map_location=device))
        print(f"resumed from {cfg['resume']}")
    if nnue is not None and cfg.get("nnue_resume") and os.path.exists(cfg["nnue_resume"]):
        nnue.load_state_dict(torch.load(cfg["nnue_resume"], map_location=device))
        print(f"resumed NNUE from {cfg['nnue_resume']}")

    best_net = copy.deepcopy(net)
    buffer = ReplayBuffer(cfg["buffer_capacity"], seed=cfg.get("seed", 0))

    for it in range(cfg["iterations"]):
        t0 = time.time()

        # ── Self-play ──
        net.eval()
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        total_positions = 0
        for g in range(cfg["games_per_iter"]):
            seed = cfg.get("seed", 0) + it * 100000 + g
            samples, res, n = play_game(net, cfg, device=device, seed=seed)
            buffer.add_game(samples)
            results[res] = results.get(res, 0) + 1
            total_positions += n

        # ── Train ──
        net.train()
        if nnue is not None:
            nnue.train()
        ploss = vloss = nloss = 0.0
        steps = cfg["train_steps"]
        for _ in range(steps):
            states, policies, wdl = buffer.sample(cfg["batch_size"])
            loss, pl, vl, nl = compute_loss(net, nnue, states, policies, wdl, device, nnue_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.get("grad_clip", 5.0))
            if nnue is not None:
                torch.nn.utils.clip_grad_norm_(nnue.parameters(), cfg.get("grad_clip", 5.0))
            opt.step()
            ploss += pl; vloss += vl; nloss += nl

        dt = time.time() - t0
        print(f"[iter {it:03d}] games={cfg['games_per_iter']} pos={total_positions} "
              f"buf={len(buffer)} W/D/L={results['1-0']}/{results['1/2-1/2']}/{results['0-1']} "
              f"| ploss={ploss/steps:.3f} vloss={vloss/steps:.3f} nnue={nloss/steps:.3f} | {dt:.1f}s")

        # ── Checkpoint ──
        ckpt = os.path.join(cfg.get("weights_dir", "../weights"), f"owen_iter{it:03d}.pt")
        torch.save(net.state_dict(), ckpt)
        torch.save(net.state_dict(), os.path.join(cfg.get("weights_dir", "../weights"), "owen_latest.pt"))
        if nnue is not None:
            nnue_pt = os.path.join(cfg.get("weights_dir", "../weights"), f"owen_nnue_iter{it:03d}.pt")
            torch.save(nnue.state_dict(), nnue_pt)
            torch.save(nnue.state_dict(), os.path.join(cfg.get("weights_dir", "../weights"), "owen_nnue_latest.pt"))
            export_nnue(nnue, os.path.join(cfg.get("weights_dir", "../weights"), "owen.nnue"))

        # ── Gating ──
        gi = cfg.get("gate_interval", 0)
        if gi and (it + 1) % gi == 0:
            net.eval()
            sc = arena(net, best_net, cfg, device, cfg.get("gate_games", 20))
            print(f"        gate: candidate vs best = {sc:.2%}")
            if sc >= cfg.get("gate_threshold", 0.55):
                best_net = copy.deepcopy(net)
                torch.save(net.state_dict(),
                           os.path.join(cfg.get("weights_dir", "../weights"), "owen_best.pt"))
                print("        -> promoted to best")

    print("training complete.")


if __name__ == "__main__":
    main()
