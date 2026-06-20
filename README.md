# Owen Chess Engine

**Owen Engine © HSR-Projects** — Licensed under **GPL-3.0-or-later** (see [LICENSE](LICENSE)).

Owen is a UCI chess engine with a **from-scratch C++20 core** — bitboard board
representation, magic-bitboard move generation (perft-validated), and an alpha-beta
search written from the ground up. For evaluation it uses a **Stockfish HalfKP NNUE**
network, runs **multi-threaded (Lazy SMP)**, and probes **Syzygy endgame
tablebases** for perfect endgame play. It beats GNU Chess and converts won games
cleanly.

> **What is original vs third-party.** The board, move generation, and the entire
> search are original Owen work. The neural-network *evaluation* uses Stockfish's
> NNUE network through the `nnue-probe` library, and tablebase probing uses the
> `Fathom` library — both included under GPL/MIT with attribution (see the headers
> in `engine/src/sfnnue/` and `engine/src/syzygy/`). A separate from-scratch
> AlphaZero-style self-play trainer also lives in `training/`.

---

## Features

**Core (from scratch)**
- 64-bit bitboards, magic-bitboard sliding attacks, Zobrist hashing.
- Legal move generation validated by **perft** (13 positions, incl. startpos depth 6
  = 119,060,324 and Kiwipete depth 5 = 193,690,690).
- Iterative deepening alpha-beta with: transposition table, null-move pruning, late
  move reductions (LMR), principal-variation search, aspiration windows, reverse-
  futility / futility / late-move pruning, static exchange evaluation (SEE) for
  capture pruning and ordering, quiescence search, killer + history move ordering,
  check extensions, and repetition + fifty-move draw detection.
- 50-move-aware scoring so won positions are *converted* (no aimless shuffling).
- Robust under any GUI: minimum-depth floor and a `MoveTime` anti-hang cap so the
  engine never plays a depth-1 move or freezes if a GUI gives it little/odd time.

**Evaluation**
- **Stockfish HalfKP NNUE** (network file in `weights/`) — strong, accurate eval.
- Handcrafted **PeSTO-tapered** evaluation (material, piece-square tables, mobility,
  king safety, pawn structure, outposts) as a fallback when no net is present.

**Performance & endgames**
- **Lazy SMP** multithreading (shared transposition table), `Threads` option.
- **Syzygy tablebases** (≤7-man) via Fathom — root DTZ probing for flawless,
  progress-guaranteed endgame conversion.

## Layout

```
owen/
├── engine/                 C++20 engine
│   ├── src/                board, movegen, search, uci, eval dispatch (original)
│   │   ├── sfnnue/         Stockfish HalfKP NNUE probe (third-party, attributed)
│   │   └── syzygy/         Fathom tablebase prober (third-party, attributed)
│   ├── Makefile            build with g++/gcc (no cmake needed)
│   └── CMakeLists.txt
├── training/               PyTorch AlphaZero-style self-play trainer
├── weights/                NNUE net + Syzygy tables (fetched, not in git — see weights/README.md)
└── README.md
```

## Build

```bash
cd engine
make                 # g++ -std=c++20 -O3 -march=native -pthread  (+ gcc for Fathom)
./owen               # start the UCI engine
./owen bench         # run the perft self-test suite
```

Then fetch the assets (one-time) so the engine plays at full strength — see
[weights/README.md](weights/README.md):

```bash
# Stockfish NNUE network (required for full strength)
curl -L -o ../weights/nn-62ef826d1a6d.nnue \
  https://github.com/official-stockfish/networks/raw/master/nn-62ef826d1a6d.nnue
# Syzygy 3-4-5-man tablebases (optional, ~939 MB) into ../weights/syzygy/
```

The engine locates these **relative to its own executable**, so it works from any
working directory / GUI. Without the net it uses the handcrafted fallback eval.

### Windows build

A portable `owen.exe` (statically linked, AVX2) is produced by CI on every push —
download it from the **Actions** tab → latest run → `owen-windows-x64` artifact.

To build it yourself:

```bash
# Cross-compile from Linux with MinGW-w64:
sudo apt install g++-mingw-w64-x86-64 gcc-mingw-w64-x86-64
cd engine && make win          # -> engine/owen.exe

# …or natively on Windows under MSYS2 (MINGW64 shell):
cd engine && make win-native   # -> engine/owen.exe
```

## UCI options

| Option | Default | Meaning |
|--------|---------|---------|
| `Threads` | 1 | Lazy SMP search threads (set to your core count). |
| `EvalFile` | auto | Path to the Stockfish NNUE network. |
| `SyzygyPath` | auto | Directory of Syzygy `.rtbw/.rtbz` files. |
| `MoveTime` | 3000 | Think time (ms) when the GUI gives no clock (anti-hang). |
| `UseGPU` | false | Use the experimental CUDA NNUE backend (if built with CUDA). |

`go` supports `wtime/btime/winc/binc`, `movetime`, `depth`, `nodes`, and `infinite`.
For a fair game in a GUI, set a real time control (e.g. 5 min, or ≥ 2 s/move) and
`Threads` to your CPU's core count.

## Training pipeline (optional, from scratch)

A self-play reinforcement-learning trainer (PyTorch) lives in `training/`:

```bash
cd training
python3.11 -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
python train.py --config config_smoke.yaml   # fast end-to-end smoke test
python train.py --config config.yaml         # real (small) profile
```

It implements MCTS-guided self-play, a residual policy/value network, a replay
buffer, and gated promotion. Reaching top-engine strength purely from self-play
needs large-scale compute; for play strength today, Owen uses the Stockfish NNUE
evaluation above.

## Acknowledgements

- **Stockfish** (GPL-3.0) — the NNUE network and HalfKP architecture.
- **nnue-probe** by Daniel Shawul (GPL-3.0) — NNUE inference.
- **Fathom** by Jon Dart / basil / Ronald de Man (MIT) — Syzygy tablebase probing.

These power Owen's evaluation and endgames; Owen's board, move generation, and
search are original work. Owen Engine © HSR-Projects, GPL-3.0.
