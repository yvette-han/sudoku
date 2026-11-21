# Sudoku with simulated annealing

Course project (CSE 352, Fall 2026).

Solve 9×9 Sudoku puzzles at three levels (easy, medium, hard) with
**simulated annealing (SA)**, and study which design choices make it work:
the **loss function**, the **initialization**, the **move**, the **cooling
schedule** and **restarts**. SA's results are then compared with two other
methods:

- **Genetic algorithm (GA):** a population of grids that breeds and mutates.
- **Backtracking depth-first search (DFS):** a tree search, and an exact solver to
  check SA's answers against.

Results are in [`results/SUMMARY.md`](results/SUMMARY.md) and `figures/`. A web page
replays the recorded runs (see "Demo").

## Running it

Needs Python 3.9 or newer; the `python3` that comes with macOS works. The solvers
use only the standard library. matplotlib is needed only for the figures:

```bash
pip3 install -r requirements.txt
```

Run these from the repository folder. No command line has a comment on it, so each
one pastes straight into zsh (which does not treat `#` as a comment when you paste).

- Run the tests (32 tests, about 10 s):
  ```bash
  python3 tests.py
  ```
- Simulated annealing on one puzzle:
  ```bash
  python3 annealing.py hard-01 --seed 1 --on-stuck reheat --max-steps 1000000
  ```
- The plain GA, then the GA with local search (memetic GA):
  ```bash
  python3 genetic.py easy-01 --seed 1 --max-seconds 10
  ```
  ```bash
  python3 genetic.py hard-01 --seed 1 --population 100 --local-search 500
  ```
- Backtracking DFS, then BFS on the puzzle built against brute force. BFS stops at its
  2,000,000-state frontier cap (about 240 MB, 15 s), which is the point of that test:
  ```bash
  python3 backtracking.py hard-01 --order mrv --forward-check
  ```
  ```bash
  python3 backtracking.py extreme-02 --bfs
  ```
- The whole experiment pipeline at a small size (about 10 min; writes to `results/quick/`
  and `figures/quick/`, which git ignores):
  ```bash
  python3 experiments.py --quick
  ```
- The full run (about 50 min on 4 cores). It rewrites the committed `results/`, `figures/`
  and `traces/`, so expect git to show changed timings afterwards:
  ```bash
  python3 experiments.py
  ```
- The demo: start a local web server, then open http://localhost:8000/demo.html
  (stop the server with Ctrl+C):
  ```bash
  python3 -m http.server
  ```

## Simulated annealing: the formulation

- **State:** a complete 9×9 grid in which every 3×3 box holds 1–9; rows and columns
  may clash. The puzzle's own digits never change.
- **Loss (energy) L:** the digits missing from each row and column, added up.
  L = 0 means solved.
- **Move:** swap two cells in the same 3×3 box (never one of the puzzle's own digits),
  so every box stays a valid permutation of 1–9.
- **Acceptance:** let Δ = L(new) − L(old).
  - Δ ≤ 0 (no worse): always accept.
  - Δ > 0 (worse): accept with probability e^(−Δ/T).
- **Temperature:** T₀ is set from the spread of 200 sample moves; T ← 0.99·T every
  100 moves. High T explores; low T only goes downhill.
- **Stop:** when L = 0, or after 1,000,000 moves.
- **When stuck:** after 20,000 moves with no new best, reheat (T back to T₀) or
  restart from a new random grid.

## Design choices tested

| Choice | Options in the code |
|---|---|
| Loss function | digits missing from each row and column (boxes kept complete by the move); number of conflicting pairs; weighted mix of rows, columns and boxes |
| Initialization | fully random fill; each box filled with its missing digits; fill forced cells first (constraint propagation), then box fill |
| Move | swap within a box; swap only cells that are currently in conflict; change one cell's digit |
| Cooling schedule | T₀; α (0.9 vs 0.99 vs 0.999); steps per temperature |
| Restarts | one long run; reheating or restarting when stuck; random-restart hill climbing (temperature 0) |

## Comparison methods

**Genetic algorithm.** A population of complete grids. Selection by loss
(tournament of 3); crossover that takes whole boxes (or rows of boxes) from two
parents; mutation = the SA move. Uses the same loss functions as SA so the
comparison is fair.

*With local search (memetic GA).* Optionally, every new grid also gets a short
local search: a fixed number of SA moves at temperature 0 (only moves that are no
worse, so it can walk along plateaus), and the improved grid replaces it. E8 tests
whether this makes the GA competitive, and whether the crossover or the local search
does the work, against two controls: the same algorithm with crossover off, and
random-restart hill climbing with no population at all.

**Backtracking DFS (the tree search).**
- State: a partly filled grid. Action: write digit 1–9 into one empty cell.
- Goal: grid full with no repeated digit in any row, column or box.
- Tree depth = number of empty cells; branching ≤ 9.
- Reject a digit as soon as it breaks a rule; optionally pick the cell with the
  fewest options first (MRV) and use forward checking.
- A breadth-first search (BFS) with a frontier cap tests whether tree search runs
  out of memory.

## Experiments

All run by `experiments.py`; numbers in [`results/SUMMARY.md`](results/SUMMARY.md),
one row per run in `results/E*_runs.csv`, charts in `figures/` (`E1.png` … `E8.png`).

| # | Question | Result |
|---|---|---|
| E1 | Which loss function solves the most puzzles with SA? | With box swaps, the loss hardly matters (95% vs 94%); changing single cells solves about 1% |
| E2 | How much does the initialization matter? | Filling forced cells first: 100% (box fill alone: 95%) |
| E3 | How do T₀ and α change solve rate and steps to solve? | Most settings reach about 95%; cooling far too slowly (α = 0.999) drops to 31% |
| E4 | Do restarts or reheating rescue runs that get stuck? | No way out: 26%; reheating 95%; restarting 97%; random-restart hill climbing 96% |
| E5 | How does SA's best setup do on easy vs medium vs hard? | 100% of 600 runs on all 60 puzzles (hill climbing: 94%) |
| E6 | SA vs GA vs backtracking DFS (and BFS) on the same puzzles and time budget | DFS with MRV + forward checking is fastest (about 0.001 s on a hard puzzle); SA 0.08 s; the plain GA never solves one in 10 s |
| E7 | Do the SA loss curves look like the ones in the reference video? | Same shape; a run with no way out flattens just above 0 (curves in `figures/E7_curves.png`; also writes the demo traces) |
| E8 | Does local search make the GA competitive, and is it the crossover or the local search doing the work? | Plain GA 0%; with local search 92% (100% with 2,000 steps); still 90% with crossover off |

- **Puzzles:** `puzzles/easy.txt`, `medium.txt`, `hard.txt`: 20 each, the first
  records of the public-domain [Sudoku Exchange Puzzle Bank](https://github.com/grantm/sudoku-exchange-puzzle-bank)
  (generated with QQWing so each has one solution, graded with Sukaku Explainer). E1–E4
  use the first 10 per level, E5 all 20. `puzzles/extreme.txt` adds two puzzles that
  are hard in other ways (Inkala's 2012 puzzle; Wikipedia's anti-brute-force example),
  used only in E6 and E8.
- **Design:** E1–E4 each change one part of a working SA baseline
  (`missing_rc`, box fill, box swap, T₀ = auto, α = 0.99 every 100 steps, reheat after
  20,000 steps without a new best, 1,000,000-step budget). E5 reruns the single best
  config measured in E1–E4 on all 60 puzzles × 10 seeds. E6 gives every method the same
  10 s per run, and E8 uses that budget for the GA variants.
- **Metrics:** solve rate, steps (or nodes) and wall-clock time to solve, final loss when unsolved.
- **Randomness:** 5 seeds per puzzle (E5: 10, E6 and E8: 3); every run's seed is recorded so any result can be rerun.

## Demo

`demo.html` replays the runs recorded by the Python solvers, so it shows exactly
what the experiments measured; the trace format is in [`traces/README.md`](traces/README.md).

- Pick a puzzle (easy, medium, hard or extreme) and a method (SA, GA, DFS, or all
  three side by side).
- Play, pause, step forward and back, change the speed, or scrub through the run.
- SA: the grid with clashing cells highlighted, plus the loss and temperature curves
  (rejected swaps outlined; restarts and reheats marked).
- GA: the best grid in the population, with the best and average loss per generation.
- DFS: the grid filling in, with a backtrack counter and a search-depth curve.
- The settings and random seed of each run are shown, so it can be repeated.

When a replay ends, the page also checks it: no puzzle digit was changed, the final
grid really is a solution, this matches what the trace claims, and the loss never hit
0 on an unsolved grid.

**Running it.** In the repository root, run `python3 -m http.server`, then
open <http://localhost:8000/demo.html>. Opening `demo.html` straight from disk
also works, but then the trace files have to be loaded with **Open trace files…**.
The fake test traces in `traces/synthetic/` only appear when "Synthetic test traces"
is ticked, and carry a banner.

The page is static (HTML plus the JSON traces), so it can also be served by GitHub
Pages: `index.html` opens `demo.html`, and `.nojekyll` serves the files as they are.

## Files

| File | What it is |
|---|---|
| `puzzles/` | Easy, medium and hard puzzles (20 each) plus 2 extreme ones, with their sources in each file's header |
| `sudoku.py` | Grid representation, rule checks, constraint propagation, loading puzzles |
| `annealing.py` | SA with plug-in losses (`missing_rc`, `pairs`, `weighted`), initializations (`random`, `box`, `propagate`), moves (`swap`, `conflict_swap`, `change`), geometric cooling, restarts and reheating |
| `genetic.py` | GA using the same losses, initializations and moves; box or band crossover; optional local search on every child (memetic GA) |
| `backtracking.py` | DFS with constraint checks, MRV, forward checking; plus a BFS with a frontier cap |
| `tests.py` | 32 tests: solutions are valid; puzzle digits untouched; L = 0 exactly when solved; same seed gives same run; incremental loss = full recompute; acceptance rule; traces replay; local search never makes a grid worse; hill climbing never accepts a worse move; every puzzle file can be looked up |
| `experiments.py` | Runs E1–E8; writes `results/`, `figures/` and the demo traces |
| `results/` | `SUMMARY.md` (all tables), `E*_runs.csv` (one row per run), `E*_summary.csv` |
| `figures/` | One chart per experiment |
| `demo.html` | The demo page (see "Demo") |
| `index.html`, `.nojekyll` | Let GitHub Pages serve the demo: the site's front page opens `demo.html` |
| `traces/` | JSON run logs the demo replays; format in `traces/README.md` |
| `tools/make_sample_traces.py` | Writes the **fake** test traces in `traces/synthetic/`; walks toward the known answer, so it is not a solver |

## Use of AI

- **Code (solvers, experiments, tests and the demo page):** written with AI assistance.
- **Written proposal and report:** my own writing.
