# Trace files for `demo.html`

The demo page only **replays** runs. A solver writes one JSON file per run,
and the page plays it back. It never solves anything itself, so whatever the
page shows is exactly what the Python run did.

```
traces/
  index.json            list of real runs (written by experiments.py)
  <run>.json            one file per run
  synthetic/            FAKE traces for testing the page (tools/make_sample_traces.py)
```

`synthetic/` holds scripted fakes: they walk toward the known answer, so they
are not results. The page shows them only when there are no real runs yet, or
when the "Synthetic test traces" box is ticked, and it marks each one with a
banner.

## Viewing

Browsers block a page opened from disk (`file://`) from reading other files.
Pick one of these:

- Serve the folder: run `python3 -m http.server` in the repository root, then open
  <http://localhost:8000/demo.html>.
- Or open `demo.html` directly and use **Open trace files…** (or drop the
  `.json` files onto the page). Pick several files of the same puzzle to see
  them side by side.
- Put it on a website: any static host works (e.g. GitHub Pages for this repo),
  as long as `demo.html` and `traces/` sit next to each other.

The URL remembers the selection, e.g.
`demo.html#level=hard&puzzle=hard-03&method=all`.

## Run file (`sudoku-trace-1`)

```json
{
  "format": "sudoku-trace-1",
  "method": "sa",
  "label": "SA, box swap, alpha=0.999",
  "puzzle": {"id": "easy-01", "level": "easy", "givens": "53..7....6..195...", "source": "where it came from"},
  "seed": 12345,
  "settings": {"loss": "missing digits in rows and columns", "init": "box fill", "move": "box swap", "T0": 1.0, "alpha": 0.999},
  "command": "python3 experiments.py --method sa --puzzle easy-01 --seed 12345",
  "start": "534678912672195348...",
  "events": [ ... ],
  "result": {"solved": true, "grid": "534678912...", "loss": 0, "steps": 5321, "time": 0.84}
}
```

| Field | Required | Meaning |
|---|---|---|
| `format` | yes | Always `"sudoku-trace-1"`. |
| `method` | yes | `"sa"`, `"ga"` or `"dfs"`. Picks the panel layout and charts. |
| `puzzle.givens` | yes | 81 characters, row by row: `1`–`9`, and `0` or `.` for empty. A 9×9 list of lists also works. |
| `puzzle.id`, `puzzle.level` | recommended | Used by the puzzle and level pickers. |
| `events` | yes | The run, in order (below). |
| `start` | no | The grid before the first event. Defaults to the givens. SA should give its initial filled grid here. |
| `seed`, `settings`, `command` | recommended | Shown under "Settings and seed" so the run can be repeated. `settings` can hold any keys. |
| `label` | no | Short name for the run, shown in the panel header. |
| `result` | no | `solved`, `grid`, `loss`, `steps`, `time`. The page checks these against the grid it rebuilds. |
| `synthetic` | no | `true` only for fake test traces. |

Cells are written as `[row, col, digit]` with `row` and `col` from 0 to 8, and
digit `0` meaning empty.

### Events

Every event has a `step`, which must never decrease. The other keys are
optional, and a value that is left out carries over from the previous event.

| Key | Meaning |
|---|---|
| `step` | x-axis value: SA step, GA generation, or DFS node count. |
| `set` | `[[row, col, digit], ...]`: the cells this event writes. |
| `grid` | 81-character string that replaces the whole grid (SA restart, GA best grid). |
| `accepted` | SA: `false` means `set` was **proposed and rejected**. The grid stays the same, and the page outlines the two cells. |
| `delta` | SA: Δ = L(new) − L(old) of the proposed move. An accepted move with `delta > 0` counts as a worse move accepted. Without `delta`, the page uses the change in `loss`. |
| `loss` | Loss of the current grid **after** this event. For a rejected move, that is the unchanged loss. For GA, it is the best loss in the population. |
| `T` | SA temperature. |
| `avg_loss` | GA: average loss of the population. |
| `restart`, `reheat`, `backtrack` | `true` to count a restart (send the new grid in `grid`), a reheat, or a DFS backtrack. |
| `time` | Seconds since the run started. |
| `skipped` | SA: how many rejected moves since the previous event were left out of the trace (they don't change the grid). The page adds them to its "rejected moves" counter. |

What each method should write:

- **SA:** a `step: 0` event with the starting `loss` and `T`, then one event per
  step: `{"step", "set": [two swapped cells with their new digits], "accepted", "delta", "loss", "T"}`.
  For a restart: `{"step", "restart": true, "grid": "...", "loss", "T"}`.
- **GA:** one event per generation: `{"step": generation, "grid": best grid, "loss": best loss, "avg_loss"}`.
- **DFS:** one event per assignment: `{"step": nodes so far, "set": [[r, c, d]]}`.
  For an undo: `{"step", "set": [[r, c, 0]], "backtrack": true}`.

### Size

Each event takes about 80 bytes, so 200,000 events make a file of about 16 MB, which still loads
fine. For longer runs, log only accepted moves and put the number of rejected
moves you left out in `skipped` (that is what `annealing.py` does after
40,000 events). The x-axis uses `step`, so gaps are fine. Rounding `T` to 4
significant digits also helps.

### Writing one from Python

```python
import json

trace = {"format": "sudoku-trace-1", "method": "sa", "seed": seed,
         "puzzle": {"id": pid, "level": level, "givens": givens, "source": source},
         "settings": settings, "start": grid_to_string(grid), "events": []}
trace["events"].append({"step": 0, "loss": L, "T": T})
# ... inside the loop, once the move has been accepted or rejected:
trace["events"].append({"step": t, "set": [[r1, c1, d1], [r2, c2, d2]],
                        "accepted": accepted, "delta": delta, "loss": L, "T": round(T, 6)})
# ... after the loop:
trace["result"] = {"solved": L == 0, "grid": grid_to_string(grid), "loss": L, "steps": t}
with open(f"traces/sa_{pid}_seed{seed}.json", "w") as f:
    json.dump(trace, f, separators=(",", ":"))
```

`d1` and `d2` are the digits the two cells hold *after* the swap. For a
rejected move, they are the digits the swap *would* have put there.

## Index file (`traces/index.json`)

This is the list the page's pickers read. Paths are relative to the index file.

```json
{
  "format": "sudoku-trace-index-1",
  "traces": [
    {"file": "sa_easy-01_seed1.json", "method": "sa", "level": "easy", "puzzle": "easy-01", "label": "SA, box swap", "seed": 1}
  ]
}
```

## What the page checks

When a replay ends, the page rebuilds the final grid from the events and
reports:

- whether any event changed a given;
- whether the final grid is a valid solution (no empty cell, no repeat in a
  row, column or box);
- whether that agrees with `result.solved` and `result.grid`;
- whether `loss` ever reached 0 while the grid was **not** solved. This means
  the loss misses a rule, e.g. a row-and-column-only loss used with a move that
  can break boxes.

Format problems (bad grid strings, cells out of range, `step` going backwards)
are shown at the top of the panel.
