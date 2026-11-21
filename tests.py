"""Tests for the solvers. Run with:  python3 tests.py   (or: python3 -m unittest tests)

What the README asks for:
  - solutions are valid
  - givens are never changed
  - L = 0 exactly when the grid is solved (for every allowed loss / init / move combination)
  - the same seed gives the same run
Plus: the incremental loss update matches a full recompute, the acceptance rule
has the right sign, and every trace replays to the grid its run ended with.
"""

import math
import random
import unittest

from annealing import (BOX_COMPLETE_INITS, INITS, MOVES, Neighbourhood, SAConfig, State, accept, anneal,
                       make_loss)
from backtracking import DFSConfig, bfs, dfs
from genetic import GAConfig, crossover, evolve, improve
from sudoku import (BOXES, LEVELS, Puzzle, boxes_are_permutations, conflicting_cells, get_puzzle, grid_to_str,
                    is_solution, is_solved, keeps_givens, load_puzzles, parse_grid, propagate)

PUZZLES = load_puzzles()
EASY = [p for p in PUZZLES if p.level == "easy"]
SOLUTIONS = {p.id: dfs(p).grid for p in PUZZLES}
LATIN = [(r + c) % 9 + 1 for r in range(9) for c in range(9)]   # rows and columns fine, boxes not

def allowed(loss, init, move):
    try:
        SAConfig(loss=loss, init=init, move=move).check()
        return True
    except ValueError:
        return False


# every (loss, init, move) combination that SAConfig.check() allows
LOSS_SETUPS = [(loss, init, move) for loss in ("missing_rc", "pairs", "weighted")
               for init in INITS for move in MOVES if allowed(loss, init, move)]


def replay(trace):
    """Rebuild the final grid from a trace, the way demo.html does."""
    givens = parse_grid(trace["puzzle"]["givens"])
    grid = parse_grid(trace["start"]) if trace.get("start") else list(givens)
    last = -1
    for e in trace["events"]:
        assert e["step"] >= last, "step went backwards"
        last = e["step"]
        if e.get("grid"):
            grid = parse_grid(e["grid"])
        if e.get("set") and e.get("accepted", True):
            for r, c, d in e["set"]:
                grid[9 * r + c] = d
        assert keeps_givens(grid, givens), f"a given changed at step {e['step']}"
    return grid


class GridBasics(unittest.TestCase):
    def test_parse_roundtrip(self):
        p = PUZZLES[0]
        self.assertEqual(parse_grid(grid_to_str(p.givens)), list(p.givens))
        self.assertEqual(parse_grid(grid_to_str(p.givens).replace("0", ".")), list(p.givens))
        with self.assertRaises(ValueError):
            parse_grid("12")

    def test_rule_checks(self):
        sol = SOLUTIONS[PUZZLES[0].id]
        self.assertTrue(is_solved(sol))
        self.assertFalse(is_solved(LATIN))
        self.assertTrue(conflicting_cells(LATIN))
        broken = list(sol)
        broken[0], broken[1] = broken[1], broken[0]
        self.assertFalse(is_solved(broken))
        self.assertEqual(conflicting_cells(sol), set())

    def test_get_puzzle_finds_every_level(self):
        for pid in ("easy-01", "medium-20", "hard-07", "extreme-01", "extreme-02"):
            self.assertEqual(get_puzzle(pid).id, pid)
        with self.assertRaises(KeyError):
            get_puzzle("hard-99")

    def test_puzzle_files(self):
        for level in LEVELS:
            self.assertEqual(sum(p.level == level for p in PUZZLES), 20)
        self.assertEqual(len(load_puzzles(["extreme"])), 2)

    def test_propagation_never_contradicts_the_solution(self):
        for p in PUZZLES:
            g, ok = propagate(p.givens)
            self.assertTrue(ok)
            sol = SOLUTIONS[p.id]
            self.assertTrue(all(d == 0 or d == sol[c] for c, d in enumerate(g)), p.id)


class Backtracking(unittest.TestCase):
    def test_dfs_solves_every_puzzle_validly(self):
        for p in PUZZLES + load_puzzles(["extreme"]):
            r = dfs(p, DFSConfig("mrv", True))
            self.assertTrue(r.solved, p.id)
            self.assertTrue(is_solution(r.grid, p.givens), p.id)

    def test_all_dfs_variants_find_the_same_unique_solution(self):
        for p in PUZZLES[::7]:
            grids = {grid_to_str(dfs(p, DFSConfig(order, fc)).grid)
                     for order in ("first", "mrv") for fc in (False, True)}
            self.assertEqual(len(grids), 1, p.id)

    def test_budget_stops_search(self):
        p = load_puzzles(["extreme"])[1]
        r = dfs(p, DFSConfig("first", False, max_nodes=10_000))
        self.assertFalse(r.solved)
        self.assertEqual(r.stopped, "node budget")

    def test_bfs(self):
        r = bfs(EASY[0])
        self.assertTrue(r.solved)
        self.assertEqual(r.grid, SOLUTIONS[EASY[0].id])
        r = bfs(load_puzzles(["extreme"])[1], max_frontier=20_000)
        self.assertEqual(r.stopped, "frontier limit")


class Losses(unittest.TestCase):
    def test_zero_on_solutions(self):
        for name in ("missing_rc", "pairs", "weighted"):
            loss = make_loss(name)
            for p in PUZZLES[::5]:
                self.assertEqual(loss.of_grid(SOLUTIONS[p.id]), 0, name)

    def test_zero_exactly_when_solved(self):
        """For every allowed setup, grids that setup can reach have L = 0 iff solved."""
        rng = random.Random(1)
        for loss_name, init, move in LOSS_SETUPS:
            loss = make_loss(loss_name)
            for p in PUZZLES[::6]:
                grid, fixed = INITS[init](p.givens, rng)
                state, nb = State(grid, loss), Neighbourhood(fixed)
                if not nb.swap_boxes:            # propagation already solved it
                    self.assertTrue(is_solved(state.grid) and state.L == 0)
                    continue
                for _ in range(400):
                    ch = MOVES[move](state, nb, rng)
                    state.apply(ch, state.delta(ch))
                    self.assertEqual(state.L == 0, is_solved(state.grid), (loss_name, init, move))
                # and grids one step away from the solution are never L = 0
                sol = list(SOLUTIONS[p.id])
                free = [c for c in BOXES[4] if p.givens[c] == 0]
                if len(free) >= 2:
                    sol[free[0]], sol[free[1]] = sol[free[1]], sol[free[0]]
                    self.assertGreater(loss.of_grid(sol), 0)

    def test_rows_and_columns_loss_misses_boxes(self):
        """Why missing_rc needs box-keeping moves: a Latin square scores 0 but is not solved."""
        self.assertEqual(make_loss("missing_rc").of_grid(LATIN), 0)
        self.assertFalse(is_solved(LATIN))
        self.assertGreater(make_loss("pairs").of_grid(LATIN), 0)
        self.assertGreater(make_loss("weighted").of_grid(LATIN), 0)
        for init, move in (("random", "change"), ("box", "change")):
            with self.assertRaises(ValueError):
                SAConfig(loss="missing_rc", init=init, move=move).check()
        with self.assertRaises(ValueError):
            SAConfig(loss="pairs", init="random", move="swap").check()

    def test_incremental_delta_matches_full_recompute(self):
        rng = random.Random(2)
        for loss_name, init, move in LOSS_SETUPS + [("weighted", "box", "change")]:
            loss = make_loss(loss_name, (1, 2, 3) if loss_name == "weighted" else None)
            p = PUZZLES[rng.randrange(len(PUZZLES))]
            grid, fixed = INITS[init](p.givens, rng)
            state, nb = State(grid, loss), Neighbourhood(fixed)
            if not nb.swap_boxes:                # propagation already solved it
                continue
            for _ in range(300):
                ch = MOVES[move](state, nb, rng)
                before = loss.of_grid(state.grid)
                d = state.delta(ch)
                after = list(state.grid)
                for c, v in ch:
                    after[c] = v
                self.assertEqual(d, loss.of_grid(after) - before)
                if rng.random() < 0.5:
                    state.apply(ch, d)
                    self.assertEqual(state.L, loss.of_grid(state.grid))


class Moves(unittest.TestCase):
    def test_moves_keep_givens_and_box_moves_keep_boxes(self):
        rng = random.Random(3)
        loss = make_loss("pairs")
        for move in MOVES:
            for p in PUZZLES[::4]:
                grid, fixed = INITS["box"](p.givens, rng)
                state, nb = State(grid, loss), Neighbourhood(fixed)
                for _ in range(200):
                    ch = MOVES[move](state, nb, rng)
                    state.apply(ch, state.delta(ch))
                self.assertTrue(keeps_givens(state.grid, p.givens))
                if move != "change":
                    self.assertTrue(boxes_are_permutations(state.grid), move)

    def test_inits(self):
        rng = random.Random(4)
        for name in INITS:
            for p in PUZZLES[::4]:
                grid, fixed = INITS[name](p.givens, rng)
                self.assertTrue(all(1 <= d <= 9 for d in grid))
                self.assertTrue(keeps_givens(grid, p.givens))
                if name in BOX_COMPLETE_INITS:
                    self.assertTrue(boxes_are_permutations(grid))
                if name == "propagate":
                    sol = SOLUTIONS[p.id]
                    self.assertTrue(all(sol[c] == grid[c] for c in range(81) if fixed[c]))


class Acceptance(unittest.TestCase):
    def test_never_reject_a_move_that_is_no_worse(self):
        rng = random.Random(5)
        self.assertTrue(all(accept(d, T, rng) for d in (-3, -1, 0) for T in (0, 0.1, 10)))

    def test_worse_moves_taken_with_probability_exp_minus_delta_over_T(self):
        rng = random.Random(6)
        for delta, T in ((1, 1.0), (2, 1.0), (1, 0.25), (3, 5.0)):
            n = 40_000
            rate = sum(accept(delta, T, rng) for _ in range(n)) / n
            expected = math.exp(-delta / T)
            self.assertLess(expected, 1)
            self.assertAlmostEqual(rate, expected, delta=4 * math.sqrt(expected * (1 - expected) / n) + 1e-3)

    def test_zero_temperature_is_greedy(self):
        rng = random.Random(7)
        self.assertFalse(any(accept(1, 0, rng) for _ in range(100)))


class Annealing(unittest.TestCase):
    def test_solves_easy_puzzles_with_valid_solutions(self):
        cfg = SAConfig(on_stuck="reheat", max_steps=400_000)
        for p in EASY[:4]:
            r = anneal(p, cfg, seed=1)
            self.assertTrue(r.solved, p.id)
            self.assertTrue(is_solution(r.grid, p.givens))
            self.assertEqual(r.grid, SOLUTIONS[p.id])

    def test_same_seed_same_run(self):
        cfg = SAConfig(max_steps=30_000, on_stuck="restart", patience=5_000)
        p = PUZZLES[25]
        a = anneal(p, cfg, seed=42, history_every=500)
        b = anneal(p, cfg, seed=42, history_every=500)
        c = anneal(p, cfg, seed=43, history_every=500)
        self.assertEqual((a.steps, a.grid, a.history, a.accepted, a.restarts),
                         (b.steps, b.grid, b.history, b.accepted, b.restarts))
        self.assertNotEqual(a.history, c.history)

    def test_unsolved_runs_keep_givens(self):
        for loss, init, move in LOSS_SETUPS:
            p = PUZZLES[45]
            r = anneal(p, SAConfig(loss=loss, init=init, move=move, max_steps=3_000), seed=3)
            self.assertTrue(keeps_givens(r.grid, p.givens))
            self.assertEqual(r.final_loss == 0, is_solved(r.grid))

    def test_hill_climbing_never_accepts_a_worse_move(self):
        r = anneal(PUZZLES[45], SAConfig(T0=0.0, on_stuck="restart", patience=2_000, max_steps=30_000), seed=5)
        self.assertEqual(r.worse_accepted, 0)
        self.assertEqual(r.T_final, 0)
        self.assertTrue(keeps_givens(r.grid, PUZZLES[45].givens))

    def test_every_option_runs(self):
        p = EASY[0]
        for kw in (dict(T0=0.5, alpha=0.9), dict(alpha="fit"), dict(on_stuck="reheat", patience=500),
                   dict(move="conflict_swap"), dict(init="propagate"), dict(steps_per_temp=7)):
            r = anneal(p, SAConfig(max_steps=5_000, **kw), seed=1)
            self.assertLessEqual(r.steps, 5_000)

    def test_trace_replays_to_final_grid(self):
        p = EASY[1]
        r = anneal(p, SAConfig(on_stuck="restart", patience=3_000, max_steps=40_000), seed=2, trace=True)
        self.assertEqual(replay(r.trace), r.grid)
        self.assertEqual(r.trace["result"]["solved"], r.solved)
        self.assertEqual(r.trace["format"], "sudoku-trace-1")

    def test_cut_trace_still_counts_every_rejected_move(self):
        p = PUZZLES[45]
        r = anneal(p, SAConfig(max_steps=30_000), seed=4, trace=True, trace_limit=2_000)
        ev = r.trace["events"]
        logged = sum(1 for e in ev if e.get("accepted") is False)
        self.assertEqual(logged + sum(e.get("skipped", 0) for e in ev), r.rejected)
        self.assertEqual(sum(1 for e in ev if e.get("accepted") is True), r.accepted)
        self.assertEqual(replay(r.trace), r.grid)


class Genetic(unittest.TestCase):
    def test_crossover_keeps_boxes_and_givens(self):
        rng = random.Random(8)
        p = PUZZLES[30]
        a, b = INITS["box"](p.givens, rng)[0], INITS["box"](p.givens, rng)[0]
        for kind in ("box", "band"):
            child = crossover(a, b, kind, rng)
            self.assertTrue(boxes_are_permutations(child))
            self.assertTrue(keeps_givens(child, p.givens))
            for box in BOXES:   # every box comes whole from one parent
                self.assertTrue(all(child[c] == a[c] for c in box) or all(child[c] == b[c] for c in box))

    def test_same_seed_same_run_and_givens_kept(self):
        cfg = GAConfig(population=30, max_generations=40)
        p = PUZZLES[22]
        a, b = evolve(p, cfg, seed=5), evolve(p, cfg, seed=5)
        self.assertEqual((a.grid, a.history), (b.grid, b.history))
        self.assertTrue(keeps_givens(a.grid, p.givens))

    def test_local_search_never_makes_a_grid_worse_at_T0(self):
        rng = random.Random(10)
        loss = make_loss("missing_rc")
        for p in PUZZLES[40::3]:
            grid, fixed = INITS["box"](p.givens, rng)
            before = loss.of_grid(grid)
            out, L, used = improve(list(grid), loss, Neighbourhood(fixed), MOVES["swap"], 300, 0.0, rng)
            self.assertLessEqual(L, before)
            self.assertEqual(L, loss.of_grid(out))
            self.assertTrue(keeps_givens(out, p.givens) and boxes_are_permutations(out))
            self.assertLessEqual(used, 301)

    def test_memetic_ga_solves_and_is_repeatable(self):
        cfg = GAConfig(population=20, local_search=500, max_generations=300)
        for p in (EASY[0], PUZZLES[40]):
            a = evolve(p, cfg, seed=1, trace=True)
            self.assertTrue(a.solved, p.id)
            self.assertTrue(is_solution(a.grid, p.givens))
            self.assertEqual(replay(a.trace), a.grid)
            b = evolve(p, cfg, seed=1)
            self.assertEqual((a.grid, a.history, a.evaluations), (b.grid, b.history, b.evaluations))

    def test_memetic_without_crossover_runs(self):
        r = evolve(PUZZLES[45], GAConfig(population=10, local_search=100, crossover_rate=0.0,
                                         max_generations=30), seed=2)
        self.assertTrue(keeps_givens(r.grid, PUZZLES[45].givens))
        self.assertLessEqual(r.generations, 30)

    def test_solves_a_nearly_finished_grid(self):
        sol = SOLUTIONS[EASY[0].id]
        rng = random.Random(9)
        blanks = set(rng.sample(range(81), 30))
        p = Puzzle("near", "easy", tuple(0 if c in blanks else d for c, d in enumerate(sol)))
        r = evolve(p, GAConfig(max_generations=500), seed=1, trace=True)
        self.assertTrue(r.solved)
        self.assertTrue(is_solution(r.grid, p.givens))
        self.assertEqual(replay(r.trace), r.grid)


class DFSTrace(unittest.TestCase):
    def test_trace_replays(self):
        for cfg in (DFSConfig("first", False), DFSConfig("mrv", True)):
            r = dfs(PUZZLES[50], cfg, trace=True)
            self.assertEqual(replay(r.trace), r.grid)
            backs = sum(1 for e in r.trace["events"] if e.get("backtrack"))
            self.assertEqual(backs, r.backtracks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
