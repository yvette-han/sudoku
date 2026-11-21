"""Tree search for Sudoku: backtracking depth-first search (DFS), plus a breadth-first search (BFS) to test memory.

Formulation (README, "Backtracking DFS"):
  state   a partly filled grid
  action  write a digit 1-9 into one empty cell
  goal    grid full, no repeated digit in any row, column or box
  depth   = number of empty cells; branching <= 9

Options:
  order          "first": the first empty cell in reading order
                 "mrv":   the empty cell with the fewest legal digits (minimum remaining values)
  forward_check  after writing a digit, undo it at once if some empty peer has no legal digit left

A digit that repeats in its row, column or box is never written (the constraint check).
The search is deterministic: digits are tried in order 1-9, so there is no seed.

Usage:
    python3 backtracking.py hard-01 --order mrv --forward-check
    python3 backtracking.py easy-01 --bfs
"""

import argparse
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass

from annealing import TRACE_FORMAT
from sudoku import BOX, COL, PEERS, ROW, get_puzzle, grid_to_str, is_solution

ALL = 0b1111111110   # bits 1..9
# Number of digits in a mask (masks stay below 1024). A table instead of int.bit_count(),
# which needs Python 3.10; macOS's built-in python3 is 3.9.
POPCOUNT = [bin(m).count("1") for m in range(1024)]


@dataclass
class DFSConfig:
    order: str = "mrv"            # first | mrv
    forward_check: bool = True
    max_nodes: int = None
    max_seconds: float = None

    def label(self):
        return f"DFS {self.order}{' + forward checking' if self.forward_check else ''}"


@dataclass
class DFSResult:
    solved: bool
    nodes: int          # digits written (search-tree nodes below the root)
    backtracks: int     # digits taken back out
    max_depth: int
    seconds: float
    grid: list
    stopped: str = ""   # "" | "node budget" | "time budget" | "no solution"
    trace: dict = None


class Board:
    """Grid plus bitmasks of the digits already used in each row, column and box."""

    def __init__(self, givens):
        self.grid = list(givens)
        self.rows = [0] * 9
        self.cols = [0] * 9
        self.boxes = [0] * 9
        for c, d in enumerate(self.grid):
            if d:
                bit = 1 << d
                if (self.rows[ROW[c]] | self.cols[COL[c]] | self.boxes[BOX[c]]) & bit:
                    raise ValueError("the givens already repeat a digit")
                self.rows[ROW[c]] |= bit
                self.cols[COL[c]] |= bit
                self.boxes[BOX[c]] |= bit

    def legal(self, c):
        """Bitmask of digits that can go in cell c."""
        return ALL & ~(self.rows[ROW[c]] | self.cols[COL[c]] | self.boxes[BOX[c]])

    def put(self, c, d):
        bit = 1 << d
        self.grid[c] = d
        self.rows[ROW[c]] |= bit
        self.cols[COL[c]] |= bit
        self.boxes[BOX[c]] |= bit

    def take(self, c):
        bit = ~(1 << self.grid[c])
        self.rows[ROW[c]] &= bit
        self.cols[COL[c]] &= bit
        self.boxes[BOX[c]] &= bit
        self.grid[c] = 0


class Budget(Exception):
    pass


def dfs(puzzle, config=None, trace=False, trace_limit=300_000):
    cfg = config or DFSConfig()
    if cfg.order not in ("first", "mrv"):
        raise ValueError("order must be first or mrv")
    board = Board(puzzle.givens)
    empties = [c for c in range(81) if board.grid[c] == 0]
    t_start = time.perf_counter()
    deadline = t_start + cfg.max_seconds if cfg.max_seconds else None
    stats = {"nodes": 0, "backtracks": 0, "max_depth": 0}
    events = [] if trace else None

    def log(c, d, back=False):
        if events is not None and len(events) < trace_limit:
            e = {"step": stats["nodes"], "set": [[ROW[c], COL[c], d]], "time": round(time.perf_counter() - t_start, 6)}
            if back:
                e["backtrack"] = True
            events.append(e)

    def choose(open_cells):
        """Index (in open_cells) of the next cell to fill, and its legal-digit mask."""
        if cfg.order == "first":
            return 0, board.legal(open_cells[0])
        best_i, best_mask, best_n = 0, 0, 10
        for i, c in enumerate(open_cells):
            mask = board.legal(c)
            n = POPCOUNT[mask]
            if n < best_n:
                best_i, best_mask, best_n = i, mask, n
                if n <= 1:
                    break
        return best_i, best_mask

    def peers_still_possible(c):
        for p in PEERS[c]:
            if board.grid[p] == 0 and board.legal(p) == 0:
                return False
        return True

    def search(open_cells, depth):
        if not open_cells:
            return True
        i, mask = choose(open_cells)
        c = open_cells[i]
        rest = open_cells[:i] + open_cells[i + 1:]
        for d in range(1, 10):
            if not mask & (1 << d):
                continue                      # d already in this row, column or box
            stats["nodes"] += 1
            if cfg.max_nodes and stats["nodes"] > cfg.max_nodes:
                raise Budget("node budget")
            if deadline and stats["nodes"] % 2048 == 0 and time.perf_counter() > deadline:
                raise Budget("time budget")
            board.put(c, d)
            log(c, d)
            stats["max_depth"] = max(stats["max_depth"], depth + 1)
            if (not cfg.forward_check or peers_still_possible(c)) and search(rest, depth + 1):
                return True
            board.take(c)
            stats["backtracks"] += 1
            log(c, 0, back=True)
        return False

    stopped = ""
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 1000))
    try:
        solved = search(empties, 0)
        if not solved:
            stopped = "no solution"
    except Budget as b:
        solved, stopped = False, str(b)
    finally:
        sys.setrecursionlimit(old_limit)
    seconds = time.perf_counter() - t_start
    if solved and not is_solution(board.grid, puzzle.givens):
        raise AssertionError("DFS returned a grid that is not a solution")

    out = None
    if events is not None:
        out = {
            "format": TRACE_FORMAT, "method": "dfs", "label": cfg.label(),
            "puzzle": {"id": puzzle.id, "level": puzzle.level, "givens": grid_to_str(puzzle.givens),
                       "source": puzzle.source, "rating": puzzle.rating},
            "seed": None,
            "settings": asdict(cfg) | {"note": "deterministic: digits are tried in order 1-9"},
            "command": f"python3 backtracking.py {puzzle.id} --order {cfg.order}"
                       + (" --forward-check" if cfg.forward_check else ""),
            "events": events,
            "result": {"solved": solved, "grid": grid_to_str(board.grid), "steps": stats["nodes"],
                       "time": round(seconds, 5)},
        }
        if len(events) >= trace_limit:
            out["result"]["note"] = f"trace cut at {trace_limit} events"
    return DFSResult(solved, stats["nodes"], stats["backtracks"], stats["max_depth"], seconds,
                     list(board.grid), stopped, out)


# ---------------------------------------------------------------------- BFS

@dataclass
class BFSResult:
    solved: bool
    nodes: int            # states generated
    peak_frontier: int    # most states held at once
    depth_reached: int
    seconds: float
    peak_bytes: int       # rough memory for the frontier (81 bytes per state + list overhead)
    stopped: str = ""
    grid: list = None


def bfs(puzzle, max_frontier=2_000_000, max_seconds=None):
    """Breadth-first search with the same constraint check, filling cells in reading order.

    Every state at depth k has the same first k empty cells filled, so the frontier
    is one level of the tree. It stops when the frontier would exceed `max_frontier`
    states, standing in for running out of memory.
    """
    board = Board(puzzle.givens)
    empties = [c for c in range(81) if board.grid[c] == 0]
    t_start = time.perf_counter()
    deadline = t_start + max_seconds if max_seconds else None
    frontier = deque([bytes(board.grid)])
    nodes = peak = 0
    per_state = 81 + 33 + 8   # bytes object + its header + the list slot pointing to it
    for depth, c in enumerate(empties):
        nxt = deque()
        for state in frontier:
            used = 0
            for p in PEERS[c]:
                used |= 1 << state[p]
            for d in range(1, 10):
                if not used & (1 << d):
                    child = bytearray(state)
                    child[c] = d
                    nxt.append(bytes(child))
                    nodes += 1
            if len(nxt) > max_frontier:
                return BFSResult(False, nodes, len(nxt), depth, time.perf_counter() - t_start,
                                 len(nxt) * per_state, "frontier limit")
            if deadline and nodes % 4096 == 0 and time.perf_counter() > deadline:
                return BFSResult(False, nodes, max(peak, len(nxt)), depth, time.perf_counter() - t_start,
                                 max(peak, len(nxt)) * per_state, "time budget")
        frontier = nxt
        peak = max(peak, len(frontier))
        if not frontier:
            return BFSResult(False, nodes, peak, depth, time.perf_counter() - t_start, peak * per_state,
                             "no solution")
    grid = list(frontier[0])
    solved = is_solution(grid, puzzle.givens)
    return BFSResult(solved, nodes, peak, len(empties), time.perf_counter() - t_start, peak * per_state, "", grid)


def main():
    ap = argparse.ArgumentParser(description="Solve one puzzle with backtracking DFS (or BFS).")
    ap.add_argument("puzzle")
    ap.add_argument("--order", default="mrv", choices=["first", "mrv"])
    ap.add_argument("--forward-check", action="store_true")
    ap.add_argument("--max-nodes", type=int)
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--bfs", action="store_true", help="run breadth-first search instead")
    ap.add_argument("--max-frontier", type=int, default=2_000_000)
    ap.add_argument("--trace")
    a = ap.parse_args()
    puzzle = get_puzzle(a.puzzle)
    if a.bfs:
        r = bfs(puzzle, a.max_frontier, a.max_seconds)
        print(f"{puzzle.id} BFS: {'solved' if r.solved else 'not solved (' + r.stopped + ')'}; "
              f"{r.nodes:,} states, peak frontier {r.peak_frontier:,} (~{r.peak_bytes / 1e6:.0f} MB), "
              f"depth {r.depth_reached}/{puzzle.empties}, {r.seconds:.2f} s")
        return
    r = dfs(puzzle, DFSConfig(a.order, a.forward_check, a.max_nodes, a.max_seconds), trace=bool(a.trace))
    print(f"{puzzle.id} {DFSConfig(a.order, a.forward_check).label()}: "
          f"{'solved' if r.solved else 'not solved (' + r.stopped + ')'}; {r.nodes:,} nodes, "
          f"{r.backtracks:,} backtracks, {r.seconds:.3f} s")
    if a.trace:
        import json
        with open(a.trace, "w") as f:
            json.dump(r.trace, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
