"""Simulated annealing (SA) for Sudoku.

Formulation (see README, "Simulated annealing: the formulation"):
  state    a complete 9x9 grid that may contain conflicts; givens never change
  loss L   how far the grid is from valid; L = 0 means solved
  move     a random neighbour of the grid (e.g. swap two free cells in one box)
  accept   always if delta = L(new) - L(old) <= 0, else with probability e^(-delta/T)
  cooling  T <- alpha * T every `steps_per_temp` steps
  stop     L = 0, or the step (or time) budget runs out
  stuck    optionally restart from a new grid or reheat T when L stops improving

Every part is a plug-in chosen by name in SAConfig, so experiments can swap
one part at a time. All randomness comes from one random.Random(seed), so
the same puzzle, config and seed always give the same run.

Usage:
    python3 annealing.py easy-01 --seed 1
    python3 annealing.py hard-03 --loss pairs --init random --move change
"""

import argparse
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass

from sudoku import BOX, BOXES, CELL_UNITS, COL, ROW, get_puzzle, grid_to_str, is_solution, propagate

TRACE_FORMAT = "sudoku-trace-1"


# ====================================================================== losses
#
# Every loss here has the form  L = sum over units u of  weight(u) * cost(u),
# where cost(u) adds up a per-digit cost of how many times each digit occurs
# in the unit. Keeping counts[unit][digit] up to date lets a move's delta be
# computed from the few units it touches, not the whole grid.

MISSING = [1] + [0] * 9                        # digit absent from the unit costs 1
PAIRS = [k * (k - 1) // 2 for k in range(10)]   # k copies of a digit = k(k-1)/2 clashing pairs


@dataclass(frozen=True)
class Loss:
    name: str
    digit_cost: tuple       # cost of a digit appearing k times in a unit, k = 0..9
    weights: tuple          # (rows, columns, boxes)
    description: str

    @property
    def unit_weight(self):
        wr, wc, wb = self.weights
        return [wr] * 9 + [wc] * 9 + [wb] * 9

    @property
    def counts_boxes(self):
        return self.weights[2] > 0

    def of_counts(self, counts):
        cost, w = self.digit_cost, self.unit_weight
        return sum(w[u] * sum(cost[counts[u][d]] for d in range(1, 10)) for u in range(27))

    def of_grid(self, grid):
        return self.of_counts(unit_counts(grid))


def make_loss(name, weights=None):
    """Build a loss by name. `weights` (rows, cols, boxes) only applies to 'weighted'."""
    if name == "missing_rc":
        return Loss(name, tuple(MISSING), (1, 1, 0),
                    "digits missing from each row and column (boxes kept complete by the move)")
    if name == "pairs":
        return Loss(name, tuple(PAIRS), (1, 1, 1),
                    "pairs of cells in the same row, column or box holding the same digit")
    if name == "weighted":
        w = tuple(weights) if weights else (1, 1, 1)
        return Loss(name, tuple(MISSING), w,
                    f"digits missing from rows, columns and boxes, weighted {w[0]}:{w[1]}:{w[2]}")
    raise ValueError(f"unknown loss {name!r}; use missing_rc, pairs or weighted")


def unit_counts(grid):
    counts = [[0] * 10 for _ in range(27)]
    for c, d in enumerate(grid):
        for u in CELL_UNITS[c]:
            counts[u][d] += 1
    return counts


class State:
    """A complete grid plus digit counts per unit and its current loss."""

    def __init__(self, grid, loss):
        self.grid = list(grid)
        self.loss = loss
        self.cost = loss.digit_cost
        self.weight = loss.unit_weight
        self.counts = unit_counts(self.grid)
        self.L = loss.of_counts(self.counts)

    def delta(self, changes):
        """Change in loss if each (cell, new_digit) in `changes` were applied. Does not apply them."""
        diff = {}
        grid = self.grid
        for c, new in changes:
            old = grid[c]
            if old == new:
                continue
            for u in CELL_UNITS[c]:
                diff[u * 10 + old] = diff.get(u * 10 + old, 0) - 1
                diff[u * 10 + new] = diff.get(u * 10 + new, 0) + 1
        total = 0
        counts, cost, weight = self.counts, self.cost, self.weight
        for key, k in diff.items():
            if k:
                u, d = divmod(key, 10)
                n = counts[u][d]
                total += weight[u] * (cost[n + k] - cost[n])
        return total

    def apply(self, changes, delta):
        grid, counts = self.grid, self.counts
        for c, new in changes:
            old = grid[c]
            for u in CELL_UNITS[c]:
                counts[u][old] -= 1
                counts[u][new] += 1
            grid[c] = new
        self.L += delta

    def in_conflict(self, c):
        d, counts = self.grid[c], self.counts
        r, k, b = CELL_UNITS[c]
        return counts[r][d] > 1 or counts[k][d] > 1 or counts[b][d] > 1


# ====================================================================== initializations
#
# Each returns (grid, fixed): a complete grid, and which cells may never change
# (the givens, plus any cells constraint propagation forced).

def init_random(givens, rng):
    """Fully random fill: every empty cell gets a random digit 1-9 (boxes not complete)."""
    grid = [d if d else rng.randint(1, 9) for d in givens]
    return grid, [d != 0 for d in givens]


def fill_boxes(grid, fixed, rng):
    """Give each box's free cells the digits that box is missing, in random order."""
    for box in BOXES:
        free = [c for c in box if not fixed[c]]
        missing = [d for d in range(1, 10) if d not in {grid[c] for c in box if fixed[c]}]
        rng.shuffle(missing)
        for c, d in zip(free, missing):
            grid[c] = d
    return grid


def init_box(givens, rng):
    """Each box filled with its missing digits, so every box is already a permutation of 1-9."""
    fixed = [d != 0 for d in givens]
    return fill_boxes(list(givens), fixed, rng), fixed


def init_propagate(givens, rng):
    """Fill forced cells first (constraint propagation) and fix them, then box-fill the rest."""
    grid, ok = propagate(givens)
    if not ok:
        raise ValueError("constraint propagation found a contradiction: the puzzle has no solution")
    fixed = [d != 0 for d in grid]
    return fill_boxes(grid, fixed, rng), fixed


INITS = {"random": init_random, "box": init_box, "propagate": init_propagate}
BOX_COMPLETE_INITS = {"box", "propagate"}


# ====================================================================== moves
#
# A move proposes changes [(cell, new_digit), ...] without applying them.

class Neighbourhood:
    """Which cells a move may touch, worked out once per (re)start."""

    def __init__(self, fixed):
        self.free = [c for c in range(81) if not fixed[c]]
        self.box_free = [[c for c in box if not fixed[c]] for box in BOXES]
        self.swap_boxes = [cells for cells in self.box_free if len(cells) >= 2]


def _two(cells, rng):
    i = rng.randrange(len(cells))
    j = rng.randrange(len(cells) - 1)
    if j >= i:
        j += 1
    return cells[i], cells[j]


def move_swap(state, nb, rng):
    """Swap two free cells in the same box (keeps every box a permutation of 1-9)."""
    a, b = _two(rng.choice(nb.swap_boxes), rng)
    g = state.grid
    return [(a, g[b]), (b, g[a])]


def move_conflict_swap(state, nb, rng):
    """Swap within a box, but start from a cell that is currently in conflict.

    The partner is another conflicting cell of the same box when there is one,
    otherwise any free cell of that box.
    """
    conflicted = [c for c in nb.free if len(nb.box_free[BOX[c]]) >= 2 and state.in_conflict(c)]
    if not conflicted:
        return move_swap(state, nb, rng)
    a = rng.choice(conflicted)
    mates = [c for c in nb.box_free[BOX[a]] if c != a]
    hot = [c for c in mates if state.in_conflict(c)]
    b = rng.choice(hot or mates)
    g = state.grid
    return [(a, g[b]), (b, g[a])]


def move_change(state, nb, rng):
    """Change one free cell to a different random digit (can break boxes)."""
    c = rng.choice(nb.free)
    d = rng.randint(1, 8)
    if d >= state.grid[c]:
        d += 1
    return [(c, d)]


MOVES = {"swap": move_swap, "conflict_swap": move_conflict_swap, "change": move_change}
BOX_KEEPING_MOVES = {"swap", "conflict_swap"}


# ====================================================================== acceptance

def accept(delta, T, rng):
    """Metropolis rule. delta = L(new) - L(old), so a worse grid has delta > 0
    and is taken with probability e^(-delta/T) < 1. No worse (delta <= 0): always taken."""
    if delta <= 0:
        return True
    if T <= 0:
        return False
    return rng.random() < math.exp(-delta / T)


# ====================================================================== config + result

@dataclass
class SAConfig:
    loss: str = "missing_rc"       # missing_rc | pairs | weighted
    weights: tuple = None           # (rows, cols, boxes) for loss=weighted
    init: str = "box"               # random | box | propagate
    move: str = "swap"              # swap | conflict_swap | change
    T0: object = "auto"             # a number, or "auto" = std dev of 200 sample move deltas
    alpha: object = 0.99            # a number in (0, 1), or "fit" = cool from T0 to T0/100 over max_steps
    steps_per_temp: int = 100       # steps between two cooling steps
    max_steps: int = 200_000
    max_seconds: float = None       # optional wall-clock budget
    on_stuck: str = "none"          # none | restart | reheat
    patience: int = 20_000          # steps without a new best (since the last reset) that count as stuck

    def check(self):
        loss = make_loss(self.loss, self.weights)
        if self.init not in INITS:
            raise ValueError(f"unknown init {self.init!r}; use {', '.join(INITS)}")
        if self.move not in MOVES:
            raise ValueError(f"unknown move {self.move!r}; use {', '.join(MOVES)}")
        if self.on_stuck not in ("none", "restart", "reheat"):
            raise ValueError("on_stuck must be none, restart or reheat")
        boxes_stay_complete = self.init in BOX_COMPLETE_INITS and self.move in BOX_KEEPING_MOVES
        if not loss.counts_boxes and not boxes_stay_complete:
            # e.g. a Latin square has every row and column complete but broken boxes: L = 0, not solved
            raise ValueError(f"loss {self.loss!r} ignores boxes, so it needs a box-complete init "
                             "(box or propagate) and a box-keeping move (swap or conflict_swap); "
                             "otherwise L = 0 would not mean solved")
        if self.init == "random" and self.move in BOX_KEEPING_MOVES:
            raise ValueError("a random fill leaves boxes with the wrong digits, and a swap inside a box "
                             "can never fix that: use move=change with init=random")
        return loss

    def label(self):
        parts = [self.loss + (f"{list(self.weights)}" if self.weights else ""), self.init, self.move,
                 f"T0={self.T0}", f"alpha={self.alpha}", f"chain={self.steps_per_temp}"]
        if self.on_stuck != "none":
            parts.append(f"{self.on_stuck}@{self.patience}")
        return " ".join(parts)


@dataclass
class SAResult:
    solved: bool
    steps: int
    seconds: float
    final_loss: float
    best_loss: float
    start_loss: float
    T0: float
    alpha: float
    T_final: float
    accepted: int
    worse_accepted: int
    rejected: int
    restarts: int
    reheats: int
    grid: list
    history: list = None    # [(step, loss, T)] every `history_every` steps
    trace: dict = None      # demo trace (traces/README.md) when trace=True


def auto_T0(state, nb, move, rng, samples=200):
    """Initial temperature = std dev of the loss change of `samples` random moves from the start grid."""
    deltas = [state.delta(move(state, nb, rng)) for _ in range(samples)]
    return max(statistics.pstdev(deltas), 1e-3)


# ====================================================================== the solver

def anneal(puzzle, config=None, seed=0, trace=False, history_every=0, trace_limit=300_000):
    """Run SA on `puzzle` (a sudoku.Puzzle). Returns an SAResult."""
    cfg = config or SAConfig()
    loss = cfg.check()
    rng = random.Random(seed)
    init, move = INITS[cfg.init], MOVES[cfg.move]
    t_start = time.perf_counter()

    grid, fixed = init(puzzle.givens, rng)
    nb = Neighbourhood(fixed)
    state = State(grid, loss)
    if state.L > 0 and not (nb.swap_boxes if cfg.move in BOX_KEEPING_MOVES else nb.free):
        raise ValueError("no free cells left for this move to change, but the grid is not solved")

    if cfg.T0 == "auto":
        # (if propagation already solved the puzzle there is nothing to sample)
        T0 = auto_T0(state, nb, move, rng) if state.L > 0 else 1.0
    else:
        T0 = float(cfg.T0)
    if cfg.alpha == "fit":
        alpha = 0.01 ** (cfg.steps_per_temp / max(cfg.max_steps, 1))
    else:
        alpha = float(cfg.alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    T = T0
    start_loss = state.L
    best = best_since_reset = state.L
    last_gain = 0
    accepted = worse = rejected = restarts = reheats = 0
    history = [(0, state.L, T)] if history_every else None

    events = None
    unlogged = 0    # rejected moves left out of the trace once it is full
    if trace:
        events = [{"step": 0, "loss": state.L, "T": round(T, 6), "time": 0.0}]
        trace_start = grid_to_str(state.grid)

    step = 0
    deadline = t_start + cfg.max_seconds if cfg.max_seconds else None
    while state.L > 0 and step < cfg.max_steps:
        step += 1
        changes = move(state, nb, rng)
        delta = state.delta(changes)
        took = accept(delta, T, rng)
        if took:
            state.apply(changes, delta)
            accepted += 1
            if delta > 0:
                worse += 1
        else:
            rejected += 1

        if events is not None:
            if took or len(events) < trace_limit:
                e = {"step": step, "set": [[ROW[c], COL[c], d] for c, d in changes],
                     "accepted": took, "delta": delta, "loss": state.L, "T": float(f"{T:.5g}"),
                     "time": round(time.perf_counter() - t_start, 5)}
                if unlogged:
                    e["skipped"] = unlogged
                    unlogged = 0
                events.append(e)
            else:
                unlogged += 1

        if state.L < best_since_reset:
            best_since_reset = state.L
            last_gain = step
            if state.L < best:
                best = state.L

        if step % cfg.steps_per_temp == 0:
            T *= alpha

        if cfg.on_stuck != "none" and state.L > 0 and step - last_gain >= cfg.patience:
            if cfg.on_stuck == "restart":
                grid, fixed = init(puzzle.givens, rng)
                nb = Neighbourhood(fixed)
                state = State(grid, loss)
                restarts += 1
                if events is not None:
                    events.append({"step": step, "restart": True, "grid": grid_to_str(state.grid),
                                   "loss": state.L, "T": round(T0, 6),
                                   "time": round(time.perf_counter() - t_start, 5)})
            else:
                reheats += 1
                if events is not None:
                    events.append({"step": step, "reheat": True, "loss": state.L, "T": round(T0, 6),
                                   "time": round(time.perf_counter() - t_start, 5)})
            T = T0
            best_since_reset = state.L
            last_gain = step

        if history_every and step % history_every == 0:
            history.append((step, state.L, T))
        if deadline and step % 1024 == 0 and time.perf_counter() > deadline:
            break

    seconds = time.perf_counter() - t_start
    solved = state.L == 0
    if solved and not is_solution(state.grid, puzzle.givens):
        raise AssertionError("loss reached 0 on a grid that is not a solution: the loss misses a rule")
    if history_every and history[-1][0] != step:
        history.append((step, state.L, T))
    if events is not None and unlogged:
        events.append({"step": step, "skipped": unlogged, "loss": state.L, "T": float(f"{T:.5g}"),
                       "time": round(seconds, 5)})

    out = None
    if events is not None:
        out = {
            "format": TRACE_FORMAT, "method": "sa",
            "label": f"SA: {cfg.label()}",
            "puzzle": {"id": puzzle.id, "level": puzzle.level, "givens": grid_to_str(puzzle.givens),
                       "source": puzzle.source, "rating": puzzle.rating},
            "seed": seed,
            "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()} | {
                "T0 used": round(T0, 6), "alpha used": alpha, "loss means": loss.description},
            "command": command_for(puzzle.id, cfg, seed),
            "start": trace_start,
            "events": events,
            "result": {"solved": solved, "grid": grid_to_str(state.grid), "loss": state.L,
                       "steps": step, "time": round(seconds, 4)},
        }

    return SAResult(solved, step, seconds, state.L, best, start_loss, T0, alpha, T, accepted, worse,
                    rejected, restarts, reheats, list(state.grid), history, out)


def command_for(pid, cfg, seed):
    d = SAConfig()
    parts = ["python3 annealing.py", pid, f"--seed {seed}"]
    for field in ("loss", "init", "move", "T0", "alpha", "steps_per_temp", "max_steps", "on_stuck", "patience"):
        v = getattr(cfg, field)
        if v != getattr(d, field):
            parts.append(f"--{field.replace('_', '-')} {v}")
    if cfg.weights:
        parts.append("--weights " + ",".join(str(w) for w in cfg.weights))
    if cfg.max_seconds:
        parts.append(f"--max-seconds {cfg.max_seconds}")
    return " ".join(parts)


def number_or_word(word):
    def parse(text):
        return text if text == word else float(text)
    return parse


def main():
    ap = argparse.ArgumentParser(description="Solve one puzzle with simulated annealing.")
    ap.add_argument("puzzle", help="puzzle id, e.g. easy-01")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", default="missing_rc", choices=["missing_rc", "pairs", "weighted"])
    ap.add_argument("--weights", help="rows,cols,boxes for --loss weighted, e.g. 1,1,2")
    ap.add_argument("--init", default="box", choices=list(INITS))
    ap.add_argument("--move", default="swap", choices=list(MOVES))
    ap.add_argument("--T0", type=number_or_word("auto"), default="auto")
    ap.add_argument("--alpha", type=number_or_word("fit"), default=0.99)
    ap.add_argument("--steps-per-temp", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--on-stuck", default="none", choices=["none", "restart", "reheat"])
    ap.add_argument("--patience", type=int, default=20_000)
    ap.add_argument("--trace", help="write a demo trace JSON to this path")
    args = ap.parse_args()

    cfg = SAConfig(args.loss, tuple(float(w) for w in args.weights.split(",")) if args.weights else None,
                   args.init, args.move, args.T0, args.alpha, args.steps_per_temp, args.max_steps,
                   args.max_seconds, args.on_stuck, args.patience)
    puzzle = get_puzzle(args.puzzle)
    res = anneal(puzzle, cfg, args.seed, trace=bool(args.trace))
    print(f"{puzzle.id}: {'solved' if res.solved else 'not solved'} after {res.steps:,} steps "
          f"in {res.seconds:.2f} s; loss {res.start_loss} -> {res.final_loss} (best {res.best_loss}); "
          f"T0 {res.T0:.3g}; worse moves accepted {res.worse_accepted:,}; "
          f"restarts {res.restarts}, reheats {res.reheats}")
    if args.trace:
        import json
        with open(args.trace, "w") as f:
            json.dump(res.trace, f, separators=(",", ":"))
        print(f"trace written to {args.trace}")


if __name__ == "__main__":
    main()
