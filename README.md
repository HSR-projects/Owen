# Owen Chess Engine

**Owen Engine © HSR-Projects** — Licensed under **GPL-3.0-or-later** (see [LICENSE](LICENSE)).



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

.

These power Owen's evaluation and endgames; Owen's board, move generation, and
search are original work. Owen Engine © HSR-Projects, GPL-3.0.
