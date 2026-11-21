"""Sudoku basics shared by every solver: grid layout, rule checks, puzzle loading.

A grid is a list of 81 ints, row by row (cell index = 9 * row + col), with 0
for an empty cell. The 27 "units" are the 9 rows, 9 columns and 9 boxes; a
grid is solved when every cell is filled and no unit repeats a digit.
"""

from dataclasses import dataclass
from pathlib import Path

PUZZLE_DIR = Path(__file__).resolve().parent / "puzzles"
LEVELS = ("easy", "medium", "hard")
DIGITS = range(1, 10)

ROW = [c // 9 for c in range(81)]
COL = [c % 9 for c in range(81)]
BOX = [(c // 27) * 3 + (c % 9) // 3 for c in range(81)]

ROWS = [[9 * r + c for c in range(9)] for r in range(9)]
COLS = [[9 * r + c for r in range(9)] for c in range(9)]
BOXES = [[9 * (3 * (b // 3) + i) + 3 * (b % 3) + j for i in range(3) for j in range(3)] for b in range(9)]

# Units are numbered 0-8 (rows), 9-17 (columns), 18-26 (boxes).
UNITS = ROWS + COLS + BOXES
CELL_UNITS = [(ROW[c], 9 + COL[c], 18 + BOX[c]) for c in range(81)]
PEERS = [sorted(set(ROWS[ROW[c]] + COLS[COL[c]] + BOXES[BOX[c]]) - {c}) for c in range(81)]


@dataclass(frozen=True)
class Puzzle:
    id: str
    level: str
    givens: tuple  # 81 ints, 0 = empty
    rating: float = None
    source: str = ""

    @property
    def empties(self):
        return sum(1 for d in self.givens if d == 0)


# ---------------------------------------------------------------- conversion

def parse_grid(text):
    """81 characters (digits, with 0 or . for empty; whitespace ignored) -> list of 81 ints."""
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) != 81:
        raise ValueError(f"a grid needs 81 cells, got {len(chars)}")
    grid = []
    for ch in chars:
        if ch in ".0":
            grid.append(0)
        elif ch in "123456789":
            grid.append(int(ch))
        else:
            raise ValueError(f"bad cell {ch!r}: use 1-9, and 0 or . for empty")
    return grid


def grid_to_str(grid):
    return "".join(str(d) for d in grid)


def format_grid(grid):
    """Human-readable grid with box lines, for printing."""
    lines = []
    for r in range(9):
        if r in (3, 6):
            lines.append("------+-------+------")
        row = [str(d) if d else "." for d in grid[9 * r: 9 * r + 9]]
        lines.append(" ".join(row[0:3]) + " | " + " ".join(row[3:6]) + " | " + " ".join(row[6:9]))
    return "\n".join(lines)


# ---------------------------------------------------------------- rule checks

def conflicting_cells(grid):
    """Set of filled cells whose digit repeats somewhere in their row, column or box."""
    bad = set()
    for unit in UNITS:
        seen = {}
        for c in unit:
            d = grid[c]
            if d:
                seen.setdefault(d, []).append(c)
        for cells in seen.values():
            if len(cells) > 1:
                bad.update(cells)
    return bad


def is_solved(grid):
    """Every cell filled and every unit holds 1-9 exactly once."""
    if len(grid) != 81 or any(d not in DIGITS for d in grid):
        return False
    return all(len({grid[c] for c in unit}) == 9 for unit in UNITS)


def keeps_givens(grid, givens):
    return all(g == 0 or g == d for g, d in zip(givens, grid))


def is_solution(grid, givens):
    """A valid solved grid that also agrees with every given."""
    return is_solved(grid) and keeps_givens(grid, givens)


def candidates(grid, c):
    """Digits that cell c could take without repeating a digit in its row, column or box."""
    used = {grid[p] for p in PEERS[c]}
    return [d for d in DIGITS if d not in used]


def boxes_are_permutations(grid):
    """True if every box holds 1-9 exactly once (rows and columns may still clash)."""
    return all(sorted(grid[c] for c in box) == list(DIGITS) for box in BOXES)


# ---------------------------------------------------------------- constraint propagation

def propagate(grid):
    """Fill every cell that the rules force, repeating until nothing changes.

    Two rules:
      - naked single: an empty cell with only one candidate digit;
      - hidden single: a digit that fits in only one empty cell of some unit.
    Returns (new_grid, ok); ok is False if some cell or digit has no place left
    (the grid has no solution). Forced digits are always part of the solution,
    so the result never rules out the answer.
    """
    g = list(grid)
    changed = True
    while changed:
        changed = False
        for c in range(81):
            if g[c] == 0:
                cand = candidates(g, c)
                if not cand:
                    return g, False
                if len(cand) == 1:
                    g[c] = cand[0]
                    changed = True
        for unit in UNITS:
            present = {g[c] for c in unit}
            for d in DIGITS:
                if d in present:
                    continue
                spots = [c for c in unit if g[c] == 0 and d in candidates(g, c)]
                if not spots:
                    return g, False
                if len(spots) == 1:
                    g[spots[0]] = d
                    present.add(d)
                    changed = True
    return g, True


# ---------------------------------------------------------------- puzzles

def load_puzzles(levels=LEVELS, folder=PUZZLE_DIR, per_level=None):
    """Read puzzles/<level>.txt files. Each line: id, 81 digits, rating, bank hash."""
    puzzles = []
    for level in ([levels] if isinstance(levels, str) else levels):
        path = Path(folder) / f"{level}.txt"
        source = ""
        found = []
        for line in path.read_text().splitlines():
            if line.startswith("#"):
                if "github.com" in line and not source:
                    source = line.lstrip("# ").strip().rstrip(".")
                continue
            if not line.strip():
                continue
            fields = line.split()
            pid, digits = fields[0], fields[1]
            rating = float(fields[2]) if len(fields) > 2 and fields[2] != "-" else None
            tag = fields[3] if len(fields) > 3 else ""
            where = f"{source}, record {tag}" if source else tag
            found.append(Puzzle(pid, level, tuple(parse_grid(digits)), rating, where))
        puzzles.extend(found[:per_level] if per_level else found)
    return puzzles


def get_puzzle(pid, folder=PUZZLE_DIR):
    """Find a puzzle by id, e.g. "hard-03" or "extreme-02" (looked up in puzzles/<level>.txt)."""
    level = pid.split("-")[0]
    files = [level] if (Path(folder) / f"{level}.txt").exists() else sorted(f.stem for f in Path(folder).glob("*.txt"))
    for p in load_puzzles(files, folder):
        if p.id == pid:
            return p
    raise KeyError(f"no puzzle {pid!r} in {folder}")


if __name__ == "__main__":
    for p in load_puzzles(per_level=1):
        print(f"{p.id} (rating {p.rating}, {p.empties} empty cells)")
        print(format_grid(p.givens), end="\n\n")
