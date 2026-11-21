"""Write FAKE traces for testing demo.html.

These are NOT solver runs, and nothing in traces/synthetic/ may be used as a
result in the report. Every trace here is scripted from the puzzle's known
answer: the script already knows where each digit goes and walks toward it,
with some noise, so the viewer has swaps, rejected moves, restarts,
generations and backtracks to show before the real solvers exist.

- The "SA-like" trace's loss is the number of cells that differ from the
  known answer. A real loss can't see the answer.
- Worse moves are let through on a made-up schedule, not by the e^(-delta/T) rule.
- The "GA-like" and "DFS-like" traces are scripted the same way.
- Times are invented.
- Each trace has "synthetic": true, and the page shows a banner on it.

Run from the repository root (standard library only):

    python3 tools/make_sample_traces.py
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "traces" / "synthetic"
COMMAND = "python3 tools/make_sample_traces.py"

# (id, level label, givens, known answer, source)
# The level labels only exercise the page's level picker. They are not ratings.
PUZZLES = [
    ("sample-wiki", "easy",
     "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
     "534678912672195348198342567859761423426853791713924856961537284287419635345286179",
     "Example puzzle from the Wikipedia article 'Sudoku' (level label is a placeholder)"),
    ("sample-euler01", "medium",
     "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
     "483921657967345821251876493548132976729564138136798245372689514814253769695417382",
     "Project Euler problem 96, grid 01 (level label is a placeholder)"),
    ("sample-inkala", "hard",
     "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..",
     "812753649943682175675491283154237896369845721287169534521974368438526917796318452",
     "Arto Inkala's 2012 puzzle (level label is a placeholder)"),
]

BOXES = [[9 * (3 * br + i) + 3 * bc + j for i in range(3) for j in range(3)]
         for br in range(3) for bc in range(3)]
BOX_OF = [0] * 81
for b, cells in enumerate(BOXES):
    for c in cells:
        BOX_OF[c] = b


def to_str(g):
    return "".join(str(d) for d in g)


def rc(c):
    return divmod(c, 9)


def base(kind, pid, level, givens, source, seed, label, settings):
    return {
        "format": "sudoku-trace-1",
        "synthetic": True,
        "method": kind,
        "label": label,
        "puzzle": {"id": pid, "level": level, "givens": givens, "source": source},
        "seed": seed,
        "settings": {"generator": "tools/make_sample_traces.py (fake: walks toward the known answer)", **settings},
        "command": COMMAND,
        "events": [],
    }


class Fake:
    def __init__(self, givens, answer, rng):
        self.rng = rng
        self.given = [ch not in ".0" for ch in givens]
        self.ans = [int(ch) for ch in answer]
        self.free = [[c for c in cells if not self.given[c]] for cells in BOXES]

    def scramble(self):
        """Answer with each box's free cells shuffled (looks like a box-fill start)."""
        g = self.ans[:]
        for cells in self.free:
            vals = [g[c] for c in cells]
            self.rng.shuffle(vals)
            for c, v in zip(cells, vals):
                g[c] = v
        return g

    def wrong(self, g):
        return sum(g[c] != self.ans[c] for c in range(81))

    def fixing_swap(self, g):
        """A swap that puts one wrong cell's correct digit in place."""
        bad = [c for c in range(81) if g[c] != self.ans[c]]
        c = self.rng.choice(bad)
        d = next(x for x in self.free[BOX_OF[c]] if g[x] == self.ans[c])
        return c, d

    def random_swap(self):
        cells = self.rng.choice([f for f in self.free if len(f) >= 2])
        a, b = self.rng.sample(cells, 2)
        return a, b


def fake_sa(pid, level, givens, answer, source, seed, stuck_floor, ramp):
    rng = random.Random(seed)
    f = Fake(givens, answer, rng)
    T0, alpha = 1.5, 0.9985
    tr = base("sa", pid, level, givens, source, seed, "SA-like (fake), box swap",
              {"loss": "FAKE: cells that differ from the known answer", "init": "each box gets its missing digits",
               "move": "swap two free cells in one box", "T0": T0, "alpha": alpha,
               "restarts": "one restart after a plateau"})
    g = f.scramble()
    tr["start"] = to_str(g)
    ev = tr["events"]
    L, T, step = f.wrong(g), T0, 0
    ev.append({"step": 0, "loss": L, "T": T0, "time": 0.0})
    phase, plateau = 1, 0
    while L > 0 and step < 30000:
        step += 1
        T *= alpha
        progress = min(1.0, step / ramp)
        # phase 1 stalls at `stuck_floor`, then restarts; phase 2 goes to 0
        can_fix = L > stuck_floor if phase == 1 else True
        if can_fix and rng.random() < 0.10 + 0.30 * progress:
            a, b = f.fixing_swap(g)
        else:
            a, b = f.random_swap()
        g[a], g[b] = g[b], g[a]
        newL = f.wrong(g)
        delta = newL - L
        # made-up acceptance schedule (NOT the SA rule): worse moves pass less often as T falls
        accepted = delta <= 0 or rng.random() < 0.45 * T / T0
        (ra, ca), (rb, cb) = rc(a), rc(b)
        e = {"step": step, "set": [[ra, ca, g[a]], [rb, cb, g[b]]], "accepted": accepted,
             "delta": delta, "loss": newL if accepted else L, "T": float(f"{T:.4g}"),
             "time": round(step * 3.1e-5, 5)}
        if accepted:
            L = newL
        else:
            g[a], g[b] = g[b], g[a]
        ev.append(e)
        if phase == 1 and L <= stuck_floor:
            plateau += 1
            if plateau > 700:
                step += 1
                g = f.scramble()
                L, T, phase = f.wrong(g), T0, 2
                ev.append({"step": step, "restart": True, "grid": to_str(g), "loss": L, "T": T0,
                           "time": round(step * 3.1e-5, 5)})
    tr["result"] = {"solved": L == 0, "grid": to_str(g), "loss": L, "steps": step, "time": ev[-1]["time"]}
    return tr


def fake_ga(pid, level, givens, answer, source, seed, floor, gens):
    rng = random.Random(seed)
    f = Fake(givens, answer, rng)
    tr = base("ga", pid, level, givens, source, seed, "GA-like (fake), box crossover",
              {"loss": "FAKE: cells that differ from the known answer", "population": 200,
               "crossover": "swap whole boxes", "mutation": "box swap, rate 0.1", "generations": gens})
    best = f.scramble()
    ev = tr["events"]
    L = f.wrong(best)
    for gen in range(gens + 1):
        if gen and L > floor and rng.random() < 0.45:
            for _ in range(rng.choice([1, 1, 2])):
                if f.wrong(best) <= floor:
                    break
                a, b = f.fixing_swap(best)
                best[a], best[b] = best[b], best[a]
        L = f.wrong(best)
        avg = L + 6 + 22 * (0.965 ** gen) + rng.uniform(-1.5, 1.5)
        ev.append({"step": gen, "grid": to_str(best), "loss": L, "avg_loss": round(avg, 2),
                   "time": round(gen * 0.012, 4)})
        if L == 0:
            break
    tr["result"] = {"solved": L == 0, "grid": to_str(best), "loss": L, "steps": ev[-1]["step"], "time": ev[-1]["time"]}
    return tr


def fake_dfs(pid, level, givens, answer, source, seed, p_wrong):
    rng = random.Random(seed)
    f = Fake(givens, answer, rng)
    tr = base("dfs", pid, level, givens, source, seed, "DFS-like (fake), row-major order",
              {"order": "row-major (fake)", "checks": "reject a digit that repeats in its row, column or box"})
    g = [f.ans[c] if f.given[c] else 0 for c in range(81)]
    empties = [c for c in range(81) if not f.given[c]]
    ev = tr["events"]
    node = 0

    def fits(c, d):
        r, col = rc(c)
        for x in range(81):
            if x != c and g[x] == d:
                xr, xc = rc(x)
                if xr == r or xc == col or BOX_OF[x] == BOX_OF[c]:
                    return False
        return True

    def put(c, d, back=False):
        nonlocal node
        g[c] = d
        if not back:
            node += 1
        e = {"step": node, "set": [[*rc(c), d]], "time": round(node * 2e-5, 6)}
        if back:
            e["backtrack"] = True
        ev.append(e)

    i = 0
    while i < len(empties):
        c = empties[i]
        wrong = [d for d in range(1, 10) if d != f.ans[c] and fits(c, d)]
        if wrong and rng.random() < p_wrong:
            # a wrong guess that only fails a few levels further down
            put(c, rng.choice(wrong))
            went = [c]
            for c2 in empties[i + 1:i + 1 + rng.randint(1, 5)]:
                if not fits(c2, f.ans[c2]):
                    break
                put(c2, f.ans[c2])
                went.append(c2)
            for c2 in reversed(went):
                put(c2, 0, back=True)
        put(c, f.ans[c])
        i += 1
    tr["result"] = {"solved": True, "grid": to_str(g), "steps": node, "time": ev[-1]["time"]}
    return tr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    # per level: SA plateau loss, SA pace, GA generations, DFS wrong-guess rate
    plans = {"easy": (8, 2500, 120, 0.10), "medium": (6, 4000, 160, 0.18), "hard": (4, 7000, 220, 0.32)}
    for k, (pid, level, givens, answer, source) in enumerate(PUZZLES):
        stuck, ramp, gens, p_wrong = plans[level]
        traces = [
            fake_sa(pid, level, givens, answer, source, seed=100 + k, stuck_floor=stuck, ramp=ramp),
            # the hard GA run is left unsolved on purpose, to show that state
            fake_ga(pid, level, givens, answer, source, seed=200 + k, floor=2 if level == "hard" else 0, gens=gens),
            fake_dfs(pid, level, givens, answer, source, seed=300 + k, p_wrong=p_wrong),
        ]
        for tr in traces:
            name = f"{tr['method']}_{pid}_seed{tr['seed']}.json"
            (OUT / name).write_text(json.dumps(tr, separators=(",", ":")) + "\n")
            index.append({"file": name, "method": tr["method"], "level": level, "puzzle": pid,
                          "label": tr["label"], "seed": tr["seed"], "synthetic": True})
            print(f"{name}: {len(tr['events'])} events, solved={tr['result']['solved']}")
    (OUT / "index.json").write_text(json.dumps({"format": "sudoku-trace-index-1", "traces": index}, indent=1) + "\n")


if __name__ == "__main__":
    main()
