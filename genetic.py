"""Genetic algorithm (GA) for Sudoku, using the same losses and moves as annealing.py.

  population  complete grids (givens fixed), built with one of SA's initializations
  fitness     the same loss as SA (lower is better; 0 = solved)
  selection   tournament: the best of `tournament` random individuals
  crossover   swap whole boxes between two parents, so every box stays a permutation:
                box   - each box comes from either parent (uniform)
                band  - the top 1 or 2 bands (rows of boxes) from one parent, the rest from the other
  mutation    SA's move (a swap inside a box, or a one-cell change for init=random)
  local search  optional (local_search > 0): every new grid then gets that many SA moves at
              a fixed temperature ls_T (0 = hill climbing that also takes sideways moves),
              and the improved grid replaces it. A GA with this is called a memetic algorithm.
  elitism     the best `elite` grids are copied unchanged into the next generation
  stop        loss 0, or the generation (or time) budget runs out

Usage:
    python3 genetic.py easy-01 --seed 1
    python3 genetic.py hard-01 --seed 1 --population 100 --local-search 500
(the second one is the memetic GA: the GA with local search)
"""

import argparse
import random
import time
from dataclasses import asdict, dataclass

from annealing import BOX_COMPLETE_INITS, INITS, MOVES, TRACE_FORMAT, Neighbourhood, State, accept, make_loss
from sudoku import BOXES, get_puzzle, grid_to_str, is_solution


@dataclass
class GAConfig:
    loss: str = "missing_rc"
    weights: tuple = None
    init: str = "box"
    population: int = 150
    elite: int = 2
    tournament: int = 3
    crossover: str = "box"          # box | band
    crossover_rate: float = 0.9
    mutation_rate: float = 0.5      # chance that a child gets mutated
    mutation_moves: int = 1         # moves applied to a mutated child
    max_generations: int = 2000
    max_seconds: float = None
    stall_restart: int = 300        # generations without a new best before the non-elite population is rebuilt (0 = never)
    local_search: int = 0           # SA moves applied to every new grid (0 = plain GA)
    ls_T: float = 0.0               # temperature of those moves (0 = take only moves that are no worse)

    def check(self):
        loss = make_loss(self.loss, self.weights)
        if self.init not in INITS:
            raise ValueError(f"unknown init {self.init!r}")
        if self.crossover not in ("box", "band"):
            raise ValueError("crossover must be box or band")
        if not loss.counts_boxes and self.init not in BOX_COMPLETE_INITS:
            raise ValueError(f"loss {self.loss!r} ignores boxes, so it needs init=box or init=propagate")
        if not 0 <= self.elite < self.population:
            raise ValueError("elite must be smaller than the population")
        if self.local_search < 0 or self.ls_T < 0:
            raise ValueError("local_search and ls_T must be >= 0")
        return loss

    def label(self):
        text = (f"{self.loss} {self.init} pop={self.population} {self.crossover}-crossover "
                f"mut={self.mutation_rate}x{self.mutation_moves} elite={self.elite}")
        if self.crossover_rate != 0.9:
            text += f" crossover-rate={self.crossover_rate}"
        if self.local_search:
            text += f" local-search={self.local_search}@T={self.ls_T}"
        return text


@dataclass
class GAResult:
    solved: bool
    generations: int
    evaluations: int
    seconds: float
    best_loss: float
    start_best_loss: float
    restarts: int
    grid: list
    history: list = None    # [(generation, best loss, average loss)]
    trace: dict = None


def mutate(grid, free_boxes, free, rng, use_change):
    """SA's move, applied in place: swap two free cells of one box, or change one cell."""
    if use_change:
        c = rng.choice(free)
        d = rng.randint(1, 8)
        grid[c] = d + 1 if d >= grid[c] else d
        return
    cells = rng.choice(free_boxes)
    i = rng.randrange(len(cells))
    j = rng.randrange(len(cells) - 1)
    if j >= i:
        j += 1
    a, b = cells[i], cells[j]
    grid[a], grid[b] = grid[b], grid[a]


def improve(grid, loss, nb, move, steps, T, rng):
    """Local search: up to `steps` SA moves at fixed temperature T on one grid.

    With T = 0 it takes every move that is no worse, so it can walk along a
    plateau of equal loss. Returns (improved grid, its loss, loss evaluations used).
    """
    state = State(grid, loss)
    used = 1
    for _ in range(steps):
        if state.L == 0:
            break
        changes = move(state, nb, rng)
        delta = state.delta(changes)
        used += 1
        if accept(delta, T, rng):
            state.apply(changes, delta)
    return state.grid, state.L, used


def crossover(p1, p2, kind, rng):
    """Child built from whole boxes of the two parents."""
    child = list(p1)
    if kind == "box":
        for box in BOXES:
            if rng.random() < 0.5:
                for c in box:
                    child[c] = p2[c]
    else:  # band: rows 0..cut*3-1 from p1, the rest from p2
        cut = rng.choice((1, 2))
        for c in range(27 * cut, 81):
            child[c] = p2[c]
    return child


def evolve(puzzle, config=None, seed=0, trace=False, history_every=1):
    cfg = config or GAConfig()
    loss = cfg.check()
    rng = random.Random(seed)
    init = INITS[cfg.init]
    t_start = time.perf_counter()
    deadline = t_start + cfg.max_seconds if cfg.max_seconds else None

    _, fixed = init(puzzle.givens, rng)
    free = [c for c in range(81) if not fixed[c]]
    free_boxes = [[c for c in box if not fixed[c]] for box in BOXES]
    free_boxes = [cells for cells in free_boxes if len(cells) >= 2]
    use_change = cfg.init == "random"
    nb = Neighbourhood(fixed)
    move = MOVES["change" if use_change else "swap"]
    can_move = bool(free_boxes) or (use_change and bool(free))
    evaluations = 0

    def score(grid):
        """Loss of a new grid, after local search if it is switched on."""
        nonlocal evaluations
        if cfg.local_search and can_move:
            grid, s, used = improve(grid, loss, nb, move, cfg.local_search, cfg.ls_T, rng)
            evaluations += used
            return grid, s
        evaluations += 1
        return grid, loss.of_grid(grid)

    def new_individual():
        return score(init(puzzle.givens, rng)[0])

    pop, scores = map(list, zip(*[new_individual() for _ in range(cfg.population)]))

    def pick():
        best = None
        for _ in range(cfg.tournament):
            i = rng.randrange(len(pop))
            if best is None or scores[i] < scores[best]:
                best = i
        return pop[best]

    order = sorted(range(len(pop)), key=scores.__getitem__)
    best_grid, best_loss = list(pop[order[0]]), scores[order[0]]
    start_best = best_loss
    last_gain = 0
    restarts = 0
    history = [(0, best_loss, sum(scores) / len(scores))]
    events = None
    if trace:
        events = [{"step": 0, "grid": grid_to_str(best_grid), "loss": best_loss,
                   "avg_loss": round(history[0][2], 3), "time": 0.0}]

    gen = 0
    while best_loss > 0 and gen < cfg.max_generations:
        gen += 1
        nxt = [list(pop[i]) for i in order[:cfg.elite]]
        nxt_scores = [scores[i] for i in order[:cfg.elite]]
        while len(nxt) < cfg.population:
            a, b = pick(), pick()
            child = crossover(a, b, cfg.crossover, rng) if rng.random() < cfg.crossover_rate else list(a)
            if can_move and rng.random() < cfg.mutation_rate:
                for _ in range(cfg.mutation_moves):
                    mutate(child, free_boxes, free, rng, use_change)
            child, s = score(child)
            nxt.append(child)
            nxt_scores.append(s)
        pop, scores = nxt, nxt_scores
        order = sorted(range(len(pop)), key=scores.__getitem__)

        if scores[order[0]] < best_loss:
            best_loss = scores[order[0]]
            best_grid = list(pop[order[0]])
            last_gain = gen
        elif cfg.stall_restart and gen - last_gain >= cfg.stall_restart:
            # keep the elite, rebuild everyone else from scratch
            for i in order[cfg.elite:]:
                pop[i], scores[i] = new_individual()
            order = sorted(range(len(pop)), key=scores.__getitem__)
            if scores[order[0]] < best_loss:
                best_loss = scores[order[0]]
                best_grid = list(pop[order[0]])
            restarts += 1
            last_gain = gen

        avg = sum(scores) / len(scores)
        if history_every and gen % history_every == 0:
            history.append((gen, best_loss, avg))
        if events is not None:
            events.append({"step": gen, "grid": grid_to_str(best_grid), "loss": best_loss,
                           "avg_loss": round(avg, 3), "time": round(time.perf_counter() - t_start, 4)})
        if deadline and time.perf_counter() > deadline:
            break

    seconds = time.perf_counter() - t_start
    solved = best_loss == 0
    if solved and not is_solution(best_grid, puzzle.givens):
        raise AssertionError("loss reached 0 on a grid that is not a solution")
    if history_every and history[-1][0] != gen:
        history.append((gen, best_loss, sum(scores) / len(scores)))

    out = None
    if events is not None:
        out = {
            "format": TRACE_FORMAT, "method": "ga",
            "label": f"GA: {cfg.label()}",
            "puzzle": {"id": puzzle.id, "level": puzzle.level, "givens": grid_to_str(puzzle.givens),
                       "source": puzzle.source, "rating": puzzle.rating},
            "seed": seed,
            "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()} | {
                "loss means": loss.description},
            "command": f"python3 genetic.py {puzzle.id} --seed {seed}" + cli_flags(cfg),
            "events": events,
            "result": {"solved": solved, "grid": grid_to_str(best_grid), "loss": best_loss,
                       "steps": gen, "time": round(seconds, 4)},
        }
    return GAResult(solved, gen, evaluations, seconds, best_loss, start_best, restarts, best_grid, history, out)


def cli_flags(cfg):
    d, out = GAConfig(), []
    for field in ("loss", "init", "population", "elite", "tournament", "crossover", "crossover_rate",
                  "mutation_rate", "mutation_moves", "max_generations", "max_seconds", "stall_restart",
                  "local_search", "ls_T"):
        v = getattr(cfg, field)
        if v != getattr(d, field):
            out.append(f" --{field.replace('_', '-')} {v}")
    if cfg.weights:
        out.append(" --weights " + ",".join(str(w) for w in cfg.weights))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description="Solve one puzzle with a genetic algorithm.")
    ap.add_argument("puzzle")
    ap.add_argument("--seed", type=int, default=0)
    d = GAConfig()
    ap.add_argument("--loss", default=d.loss, choices=["missing_rc", "pairs", "weighted"])
    ap.add_argument("--weights")
    ap.add_argument("--init", default=d.init, choices=list(INITS))
    ap.add_argument("--population", type=int, default=d.population)
    ap.add_argument("--elite", type=int, default=d.elite)
    ap.add_argument("--tournament", type=int, default=d.tournament)
    ap.add_argument("--crossover", default=d.crossover, choices=["box", "band"])
    ap.add_argument("--crossover-rate", type=float, default=d.crossover_rate)
    ap.add_argument("--mutation-rate", type=float, default=d.mutation_rate)
    ap.add_argument("--mutation-moves", type=int, default=d.mutation_moves)
    ap.add_argument("--max-generations", type=int, default=d.max_generations)
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--stall-restart", type=int, default=d.stall_restart)
    ap.add_argument("--local-search", type=int, default=d.local_search,
                    help="SA moves applied to every new grid (memetic GA); 0 = plain GA")
    ap.add_argument("--ls-T", type=float, default=d.ls_T, help="temperature of the local search (0 = hill climbing)")
    ap.add_argument("--trace")
    a = ap.parse_args()
    cfg = GAConfig(a.loss, tuple(float(w) for w in a.weights.split(",")) if a.weights else None, a.init,
                   a.population, a.elite, a.tournament, a.crossover, a.crossover_rate, a.mutation_rate,
                   a.mutation_moves, a.max_generations, a.max_seconds, a.stall_restart, a.local_search, a.ls_T)
    puzzle = get_puzzle(a.puzzle)
    res = evolve(puzzle, cfg, a.seed, trace=bool(a.trace))
    print(f"{puzzle.id}: {'solved' if res.solved else 'not solved'} after {res.generations:,} generations "
          f"({res.evaluations:,} evaluations) in {res.seconds:.2f} s; best loss {res.start_best_loss} -> "
          f"{res.best_loss}; population rebuilt {res.restarts} times")
    if a.trace:
        import json
        with open(a.trace, "w") as f:
            json.dump(res.trace, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
