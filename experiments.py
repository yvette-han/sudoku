"""Run the README's experiments E1-E8 and write results/, figures/ and demo traces.

Everything, full size (about 50 minutes on 4 cores):
    python3 experiments.py
Only some experiments:
    python3 experiments.py E1 E3
A small version for checking the pipeline (about 10 minutes):
    python3 experiments.py --quick
Only rebuild SUMMARY.md and the figures from saved results:
    python3 experiments.py --report

Design: E1-E4 each change ONE part of a fixed, working SA baseline (one factor
at a time) and keep everything else the same. E5 takes the single best config
measured in E1-E4 (highest solve rate over all levels; ties go to the lower
median steps) and runs it on more puzzles and seeds. It does not combine the
winners into a new, untested config, because the parts interact (very slow
cooling plus reheating, for example, never cools down). E6 compares methods on
the same puzzles with the same wall-clock budget.
Each run's seed is recorded, so any row of results/E*_runs.csv can be rerun.

Outputs:
  results/E*_runs.csv      one row per run
  results/E*_summary.csv   solve rate etc. per config and level
  results/SUMMARY.md       all summary tables (numbers only)
  figures/E*.png           charts
  traces/*.json + index.json   runs the demo page replays (E7, full mode only)
"""

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

from annealing import SAConfig, anneal
from backtracking import DFSConfig, bfs, dfs
from genetic import GAConfig, evolve
from sudoku import LEVELS, load_puzzles

ROOT = Path(__file__).resolve().parent
ALL_LEVELS = LEVELS + ("extreme",)
PUZZLES = {p.id: p for p in load_puzzles(ALL_LEVELS)}

SIZES = {
    "full": dict(puzzles=10, seeds=5, e5_puzzles=20, e5_seeds=10, e6_seconds=10.0, e6_seeds=3, e7_seeds=9),
    "quick": dict(puzzles=3, seeds=2, e5_puzzles=4, e5_seeds=3, e6_seconds=2.0, e6_seeds=2, e7_seeds=3),
}

# The SA baseline that E1-E4 each vary one part of: box swaps, geometric cooling,
# and a reheat whenever the best loss has not improved for 20,000 steps.
BASE = asdict(SAConfig(loss="missing_rc", init="box", move="swap", T0="auto", alpha=0.99,
                       steps_per_temp=100, max_steps=1_000_000, on_stuck="reheat", patience=20_000))
# Random-restart hill climbing, the no-temperature control used in E4, E5, E6 and E8: SA at
# temperature 0 (only moves that are no worse, sideways included), restarting from a new box
# fill after 2,000 steps without a new best.
HILL = {"T0": 0.0, "on_stuck": "restart", "patience": 2_000}
SA_STEPS_PER_SECOND = 300_000   # rough speed, used to size step budgets for E6's time budget

# Memetic GA (GA + local search) settings.
# MEMETIC: picked in a pilot scan (3 puzzles per level x 2 seeds, 10 s), where population 100
#   with 500 local-search steps per child solved 18/18. E8's variants start from it.
# MEMETIC_BEST: the best memetic setting measured in E8 (2,000 steps per child), used in E6
#   and for the demo traces (E7).
MEMETIC = {"population": 100, "local_search": 500}
MEMETIC_BEST = {"population": 100, "local_search": 2000}

QUESTIONS = {
    "E1": "Which loss function solves the most puzzles with SA?",
    "E2": "How much does the initialization matter?",
    "E3": "How do T0 and alpha change solve rate and steps to solve?",
    "E4": "Do restarts or reheating rescue runs that get stuck?",
    "E5": "How does SA's best setup do on easy vs medium vs hard?",
    "E6": "SA vs GA vs backtracking DFS (and BFS) on the same puzzles and time budget",
    "E7": "Do my SA loss curves look like the ones in the video? (curves + demo traces; comparing with the video is manual)",
    "E8": "Does local search make the GA competitive, and is it the crossover or the local search doing the work?",
}


# ====================================================================== experiment definitions
# Each returns (description, [(block, label, method, config), ...], puzzle ids, seeds)

def sa(**changes):
    return BASE | changes


def puzzle_ids(n, levels=LEVELS):
    return [p.id for p in PUZZLES.values() if p.level in levels and int(p.id.split("-")[1]) <= n]


def e1(size):
    configs = [
        ("box moves", "missing_rc / box / swap", "sa", sa()),
        ("box moves", "pairs / box / swap", "sa", sa(loss="pairs")),
        ("cell changes", "pairs / random / change", "sa", sa(loss="pairs", init="random", move="change")),
        ("cell changes", "weighted 1:1:1 / random / change", "sa",
         sa(loss="weighted", weights=(1, 1, 1), init="random", move="change")),
        ("cell changes", "weighted 1:1:3 / random / change", "sa",
         sa(loss="weighted", weights=(1, 1, 3), init="random", move="change")),
    ]
    note = ("Two blocks. With box moves the boxes are always complete, so a box term would always be 0 "
            "(weighted = missing_rc there, not run twice). With one-cell changes the loss must count boxes, "
            "so missing_rc is not allowed.")
    return note, configs, puzzle_ids(size["puzzles"]), range(1, size["seeds"] + 1)


def e2(size):
    configs = [
        ("box moves", "box / missing_rc / swap", "sa", sa(init="box")),
        ("box moves", "propagate / missing_rc / swap", "sa", sa(init="propagate")),
        ("cell changes", "random / weighted / change", "sa",
         sa(init="random", loss="weighted", weights=(1, 1, 1), move="change")),
        ("cell changes", "box / weighted / change", "sa",
         sa(init="box", loss="weighted", weights=(1, 1, 1), move="change")),
        ("cell changes", "propagate / weighted / change", "sa",
         sa(init="propagate", loss="weighted", weights=(1, 1, 1), move="change")),
    ]
    note = "A swap inside a box can't repair a random fill, so random init only appears with one-cell changes."
    return note, configs, puzzle_ids(size["puzzles"]), range(1, size["seeds"] + 1)


def e3(size):
    configs = []
    for T0 in ("auto", 0.3, 3.0):
        for alpha in (0.9, 0.99, 0.999):
            configs.append(("T0 x alpha (chain 100)", f"T0={T0} alpha={alpha} chain=100", "sa",
                            sa(T0=T0, alpha=alpha)))
    for chain in (10, 1000):
        configs.append(("steps per temperature", f"T0=auto alpha=0.99 chain={chain}", "sa",
                        sa(steps_per_temp=chain)))
    note = ("'chain' = steps per temperature; 'auto' T0 = std dev of 200 sample move deltas. The baseline's "
            "reheat stays on, so a reheat returns T to the T0 under test.")
    return note, configs, puzzle_ids(size["puzzles"]), range(1, size["seeds"] + 1)


def e4(size):
    configs = [
        ("stuck handling", "one run, alpha=0.99", "sa", sa(on_stuck="none")),
        ("stuck handling", "one long run, alpha fit to budget", "sa", sa(on_stuck="none", alpha="fit")),
        ("stuck handling", "restart after 20k steps stuck", "sa", sa(on_stuck="restart", patience=20_000)),
        ("stuck handling", "reheat after 20k steps stuck (baseline)", "sa", sa()),
        ("stuck handling", "reheat after 5k steps stuck", "sa", sa(on_stuck="reheat", patience=5_000)),
        ("no temperature", "random-restart hill climbing (T=0, restart after 2k steps stuck)", "sa",
         sa(**HILL)),
    ]
    note = ("Same budget for all: 1,000,000 steps. 'Stuck' = no new best loss for that many steps. A restart "
            "starts again from a new random box fill; a reheat sets T back to T0 and keeps the grid. "
            "'alpha fit' cools from T0 to T0/100 over exactly the budget. Random-restart hill climbing is SA "
            "with the temperature at 0 (only moves that are no worse, sideways included), restarting after "
            "2,000 steps without a new best; it is the same control as in E8.")
    return note, configs, puzzle_ids(size["puzzles"]), range(1, size["seeds"] + 1)


def e5(size, out):
    best, why = best_config(out)
    configs = [("best", "best SA setup", "sa", best),
               ("no temperature", "random-restart hill climbing", "sa", sa(**HILL))]
    why += (" For comparison, random-restart hill climbing (SA at temperature 0, restarting after 2,000 steps "
            "without a new best, as in E4) runs on the same puzzles and seeds with the same 1,000,000-step budget.")
    return why, configs, puzzle_ids(size["e5_puzzles"]), range(1, size["e5_seeds"] + 1)


def hill_climbing(secs):
    """Random-restart hill climbing on a wall-clock budget: SA at temperature 0 (only moves that are
    no worse, sideways included), restarting from a new box fill after 2,000 steps without a new best."""
    return sa(**HILL, max_steps=int(SA_STEPS_PER_SECOND * secs), max_seconds=secs)


def e6(size, out):
    secs = size["e6_seconds"]
    best, _ = best_config(out)
    sa_cfg = best | {"max_steps": int(SA_STEPS_PER_SECOND * secs), "max_seconds": secs}
    ga = asdict(GAConfig(max_generations=10 ** 9, max_seconds=secs))
    configs = [
        ("local search", "SA (best setup)", "sa", sa_cfg),
        ("local search", "GA, box crossover", "ga", ga),
        ("local search", "GA, band crossover", "ga", ga | {"crossover": "band"}),
        ("local search", "GA + local search (memetic, 2000 steps)", "ga", ga | MEMETIC_BEST),
        ("local search", "random-restart hill climbing", "sa", hill_climbing(secs)),
        ("tree search", "DFS, reading order", "dfs", asdict(DFSConfig("first", False, max_seconds=secs))),
        ("tree search", "DFS, reading order + forward checking", "dfs",
         asdict(DFSConfig("first", True, max_seconds=secs))),
        ("tree search", "DFS, MRV + forward checking", "dfs", asdict(DFSConfig("mrv", True, max_seconds=secs))),
        ("tree search", "BFS, reading order", "bfs", dict(max_frontier=2_000_000, max_seconds=secs)),
    ]
    note = (f"Every method gets {secs:g} s of wall-clock time per run. SA and GA use several seeds; DFS and BFS "
            "are deterministic (one run). BFS stops when its frontier passes 2,000,000 states (~240 MB), "
            "standing in for running out of memory. Random-restart hill climbing is SA at temperature 0, restarting "
            "after 2,000 steps without a new best (the same control as in E4 and E8). "
            "'extreme' = puzzles/extreme.txt (E6 and E8 only). A time budget makes the stopping point depend on machine speed, so a rerun with the same seed follows the same path but may stop a little earlier or later.")
    ids = puzzle_ids(size["puzzles"]) + [p for p in PUZZLES if p.startswith("extreme")]
    return note, configs, ids, range(1, size["e6_seeds"] + 1)


def e8(size, out):
    secs = size["e6_seconds"]
    ga = asdict(GAConfig(max_generations=10 ** 9, max_seconds=secs))
    mem = ga | MEMETIC
    configs = [
        ("GA", "GA, no local search", "ga", ga),
        ("memetic GA", "memetic: pop 100, 500 LS steps", "ga", mem),
        ("memetic GA", "memetic: pop 30, 500 LS steps", "ga", mem | {"population": 30}),
        ("memetic GA", "memetic: pop 100, 100 LS steps", "ga", mem | {"local_search": 100}),
        ("memetic GA", "memetic: pop 100, 2000 LS steps", "ga", mem | {"local_search": 2000}),
        ("memetic GA", "memetic, band crossover", "ga", mem | {"crossover": "band"}),
        ("controls", "memetic, no crossover", "ga", mem | {"crossover_rate": 0.0}),
        ("controls", "random-restart hill climbing", "sa", hill_climbing(secs)),
    ]
    note = (f"Every config gets {secs:g} s per run, as in E6. 'LS steps' = SA moves of local search applied to every "
            "new grid, at temperature 0 (only moves that are no worse). 'No crossover': children are mutated copies "
            "of tournament winners, then local search, so selection and local search without crossover. "
            "'Random-restart hill climbing': no population at all; the same temperature-0 moves on one grid, "
            "restarting from a new box fill after 2,000 steps without a new best.")
    ids = puzzle_ids(size["puzzles"]) + [p for p in PUZZLES if p.startswith("extreme")]
    return note, configs, ids, range(1, size["e6_seeds"] + 1)


# ====================================================================== running

def run_job(job):
    exp, block, label, method, cfg, pid, seed = job
    p = PUZZLES[pid]
    row = {"experiment": exp, "block": block, "config": label, "method": method, "puzzle": pid,
           "level": p.level, "rating": p.rating, "seed": seed}
    if method == "sa":
        r = anneal(p, SAConfig(**cfg), seed)
        row |= {"solved": r.solved, "steps": r.steps, "seconds": round(r.seconds, 4),
                "start_loss": r.start_loss, "final_loss": r.final_loss, "best_loss": r.best_loss,
                "restarts": r.restarts, "reheats": r.reheats, "worse_accepted": r.worse_accepted,
                "accepted": r.accepted, "rejected": r.rejected, "T0_used": round(r.T0, 4)}
    elif method == "ga":
        r = evolve(p, GAConfig(**cfg), seed, history_every=0)
        row |= {"solved": r.solved, "steps": r.generations, "evaluations": r.evaluations,
                "seconds": round(r.seconds, 4), "start_loss": r.start_best_loss, "final_loss": r.best_loss,
                "best_loss": r.best_loss, "restarts": r.restarts}
    elif method == "dfs":
        r = dfs(p, DFSConfig(**cfg))
        row |= {"seed": "", "solved": r.solved, "steps": r.nodes, "backtracks": r.backtracks,
                "max_depth": r.max_depth, "seconds": round(r.seconds, 5), "stopped": r.stopped}
    elif method == "bfs":
        r = bfs(p, **cfg)
        row |= {"seed": "", "solved": r.solved, "steps": r.nodes, "peak_frontier": r.peak_frontier,
                "peak_mb": round(r.peak_bytes / 1e6, 1), "max_depth": r.depth_reached,
                "seconds": round(r.seconds, 4), "stopped": r.stopped}
    row["cfg"] = json.dumps(cfg, default=list)
    return row


def run_experiment(exp, note, configs, ids, seeds, out, workers):
    jobs = []
    for block, label, method, cfg in configs:
        for pid in ids:
            for seed in (seeds if method in ("sa", "ga") else [0]):
                jobs.append((exp, block, label, method, cfg, pid, seed))
    print(f"\n{exp}: {QUESTIONS[exp]}\n  {len(configs)} configs x {len(ids)} puzzles -> {len(jobs)} runs", flush=True)
    t0 = time.time()
    rows = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_job, jobs), 1):
            rows.append(row)
            if i % max(1, len(jobs) // 10) == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs, {time.time() - t0:.0f} s", flush=True)
    order = {(c[1]): k for k, c in enumerate(configs)}
    rows.sort(key=lambda r: (order[r["config"]], ALL_LEVELS.index(r["level"]), r["puzzle"], str(r["seed"])))
    write_csv(out / f"{exp}_runs.csv", rows)
    meta = load_meta(out)
    meta[exp] = {"note": note, "configs": [c[1] for c in configs], "runs": len(rows),
                 "wall_seconds": round(time.time() - t0), "finished": time.strftime("%Y-%m-%d %H:%M"),
                 "commit": git_commit()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return rows


# ====================================================================== summaries

def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["solved"] = r["solved"] == "True"
        for k in ("steps", "seconds", "final_loss", "best_loss", "restarts", "reheats", "peak_mb", "evaluations"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
    return rows


def load_meta(out):
    path = out / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def summarize(rows):
    """Per (block, config, level), plus an 'all' row per config pooled over the bank levels."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["block"], r["config"], r["method"], r["level"])].append(r)
        if r["level"] in LEVELS:
            groups[(r["block"], r["config"], r["method"], "all")].append(r)
    out = []
    for (block, config, method, level), rs in groups.items():
        ok = [r for r in rs if r["solved"]]
        bad = [r for r in rs if not r["solved"]]
        med = lambda xs: round(statistics.median(xs), 4) if xs else ""
        out.append({
            "block": block, "config": config, "method": method, "level": level, "runs": len(rs),
            "solved": len(ok), "solve_rate": round(len(ok) / len(rs), 3),
            "median_steps_solved": med([r["steps"] for r in ok]),
            "median_seconds_solved": med([r["seconds"] for r in ok]),
            "mean_final_loss_unsolved": round(statistics.mean(r["final_loss"] for r in bad), 2)
            if bad and all(r.get("final_loss") not in (None, "") for r in bad) else "",
            "mean_best_loss_unsolved": round(statistics.mean(r["best_loss"] for r in bad), 2)
            if bad and all(r.get("best_loss") not in (None, "") for r in bad) else "",
            "mean_restarts": round(statistics.mean(r.get("restarts") or 0 for r in rs), 2),
            "mean_reheats": round(statistics.mean(r.get("reheats") or 0 for r in rs), 2),
            "peak_mb_max": max((r["peak_mb"] for r in rs if r.get("peak_mb") not in (None, "")), default=""),
        })
    return out


def config_order(rows):
    seen = []
    for r in rows:
        if r["config"] not in seen:
            seen.append(r["config"])
    return seen


def winner(rows, keep=None):
    """Config with the best pooled solve rate; ties -> lower median steps to solve."""
    by = defaultdict(list)
    for r in rows:
        if r["level"] in LEVELS and (keep is None or keep(r)):
            by[r["config"]].append(r)
    if not by:
        return None, None

    def score(item):
        rs = item[1]
        ok = [r["steps"] for r in rs if r["solved"]]
        return (-len(ok) / len(rs), statistics.median(ok) if ok else float("inf"))

    name, rs = min(by.items(), key=score)
    return name, json.loads(rs[0]["cfg"])


def best_config(out):
    """The single best SA config measured in E1-E4 (all configs share the same budget and puzzles)."""
    rows = []
    for e in ("E1", "E2", "E3", "E4"):
        path = out / f"{e}_runs.csv"
        if path.exists():
            for r in read_csv(path):
                r["config"] = f"{e}: {r['config']}"
                rows.append(r)
    if not rows:
        return dict(BASE), "E1-E4 have not been run, so this is the baseline."
    name, cfg = winner(rows)
    if isinstance(cfg.get("weights"), list):
        cfg["weights"] = tuple(cfg["weights"])
    return cfg, (f"Best setup = the config with the highest solve rate over all levels in E1-E4 "
                 f"(ties: lower median steps): {name}, i.e. `{SAConfig(**cfg).label()}`, "
                 f"budget {cfg['max_steps']:,} steps.")


def fmt_cell(s):
    if not s:
        return ""
    txt = f"{100 * s['solve_rate']:.0f}%"
    if s["median_steps_solved"] != "":
        v = s["median_steps_solved"]
        txt += f" · {v:,.0f}" if v >= 10 else f" · {v:g}"
    return txt


def markdown_report(out, size_name):
    meta = load_meta(out)
    lines = [f"# Experiment results ({size_name} run)", "",
             "Generated by `experiments.py`. Numbers only; what they mean is for the report.", "",
             f"- Machine: {os.cpu_count()} cores, Python {platform.python_version()}, {platform.system()}",
             f"- Code version the runs used: `{', '.join(sorted({m.get('commit', '?') for m in meta.values()})) or git_commit()}`"
             + (f" (E7 traces: `{json.loads((out / 'E7.json').read_text()).get('commit', '?')}`)" if (out / "E7.json").exists() else ""),
             f"- SA baseline for E1-E4: `{SAConfig(**BASE).label()}`, budget {BASE['max_steps']:,} steps unless noted",
             "- Cells: solve rate · median steps to solve (SA steps, GA generations, DFS/BFS nodes), over solved runs only.",
             "- Puzzles: puzzles/*.txt (Sudoku Exchange Puzzle Bank, graded easy/medium/hard).", ""]
    for exp in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"):
        if exp == "E7":
            if (out / "E7.json").exists():
                e7 = json.loads((out / "E7.json").read_text())
                lines += ["## E7. " + QUESTIONS["E7"], "", e7["note"], ""]
                for t in e7.get("traces", []):
                    lines.append(f"- `{t['file']}`: {t['label']} on {t['puzzle']}: {t['outcome']}")
                lines.append("")
            continue
        path = out / f"{exp}_runs.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        summ = summarize(rows)
        write_csv(out / f"{exp}_summary.csv", summ)
        levels = [lv for lv in ALL_LEVELS if any(r["level"] == lv for r in rows)]
        cols = levels + (["all"] if len([lv for lv in levels if lv in LEVELS]) > 1 else [])
        info = meta.get(exp, {})
        lines += [f"## {exp}. {QUESTIONS[exp]}", "", info.get("note", ""), "",
                  f"{info.get('runs', len(rows))} runs; finished {info.get('finished', '?')}.", ""]
        lines.append("| config | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        index = {(s["config"], s["level"]): s for s in summ}
        for cfg in config_order(rows):
            lines.append(f"| {cfg} | " + " | ".join(fmt_cell(index.get((cfg, c))) for c in cols) + " |")
        if exp in ("E6", "E8"):
            lines += ["", "Median seconds to solve (solved runs) and, for BFS, the largest frontier in MB:", "",
                      "| config | " + " | ".join(levels) + " |", "|---|" + "---|" * len(levels)]
            for cfg in config_order(rows):
                cells = []
                for lv in levels:
                    s = index.get((cfg, lv))
                    if not s:
                        cells.append("")
                        continue
                    t = s["median_seconds_solved"]
                    cell = f"{t:.3g} s" if t != "" else "—"
                    if s["peak_mb_max"] != "":
                        cell += f" (peak {s['peak_mb_max']:g} MB)"
                    cells.append(cell)
                lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
        if exp in ("E1", "E2", "E3", "E4"):
            unsolved = [s for s in summ if s["level"] == "all" and s["mean_final_loss_unsolved"] != ""]
            if unsolved:
                unsolved.sort(key=lambda s: config_order(rows).index(s["config"]))
                lines += ["", "Unsolved runs (all levels), mean loss at the end of the budget / mean best loss "
                          "they reached (a run that reheats or restarts can end far above its best):", ""]
                lines += [f"- {s['config']}: {s['mean_final_loss_unsolved']} / {s['mean_best_loss_unsolved']}"
                          for s in unsolved]
        lines.append("")
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print(f"wrote {out / 'SUMMARY.md'}")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ====================================================================== E7: curves + demo traces

E7_CURVES = [("baseline (reheat when stuck)", "baseline", BASE),
             ("baseline without reheating", "noreheat", BASE | {"on_stuck": "none"})]
# Random-restart hill climbing (no temperature), drawn in its own row: its frequent restarts
# make a dense sawtooth that would hide the SA curves on a shared panel.
E7_HILL = ("random-restart hill climbing", "hill", BASE | HILL)


def e7_curves(figs):
    """Loss and temperature of one run (seed 1) per level: SA with and without reheating, and hill climbing."""
    curves = {}
    for level in LEVELS:
        for name, _, cfg in E7_CURVES + [E7_HILL]:
            r = anneal(PUZZLES[f"{level}-01"], SAConfig(**cfg), seed=1, history_every=200)
            curves[(level, name)] = (r.history, r.solved, r.steps, r.restarts)
    plot_e7(curves, [c[0] for c in E7_CURVES], figs, hill=E7_HILL[0])


def e7(size, out, figs, write_traces):
    best, _ = best_config(out)
    sa_setups = list(E7_CURVES)
    if SAConfig(**best).label() != SAConfig(**BASE).label():
        sa_setups.append(("best setup from E1-E4", "best", best))
    sa_setups.append(E7_HILL)
    e7_curves(figs)

    note = ("Loss and temperature curves: figures/E7_curves.png (puzzle <level>-01, seed 1: the baseline with and "
            "without reheating, and random-restart hill climbing in its own row). Demo traces: for each level's "
            "first puzzle, each SA setup (hill climbing included) is run with seeds "
            f"1..{size['e7_seeds']} and the run with the median number of steps is saved (a typical run, not the "
            "luckiest). SA traces log every step up to 40,000 events, then only accepted moves.")
    record = {"note": note, "traces": []}
    if write_traces:
        tdir = ROOT / "traces"
        index = []

        def save(trace, name, outcome):
            path = tdir / name
            path.write_text(json.dumps(trace, separators=(",", ":")) + "\n")
            index.append({"file": name, "method": trace["method"], "level": trace["puzzle"]["level"],
                          "puzzle": trace["puzzle"]["id"], "label": trace["label"], "seed": trace["seed"]})
            record["traces"].append({"file": f"traces/{name}", "label": trace["label"],
                                     "puzzle": trace["puzzle"]["id"], "outcome": outcome})
            print(f"  trace {name}: {len(trace['events']):,} events, {path.stat().st_size / 1e6:.1f} MB", flush=True)

        for old in tdir.glob("*.json"):
            if old.name != "index.json":
                old.unlink()
        for level in LEVELS:
            p = PUZZLES[f"{level}-01"]
            for name, tag, cfg in sa_setups:
                runs = sorted((anneal(p, SAConfig(**cfg), s).steps, s) for s in range(1, size["e7_seeds"] + 1))
                steps, seed = runs[len(runs) // 2]
                r = anneal(p, SAConfig(**cfg), seed, trace=True, trace_limit=40_000)
                r.trace["label"] = f"SA, {name} (median of {len(runs)} seeds)"
                save(r.trace, f"sa_{tag}_{p.id}_seed{seed}.json",
                     f"{'solved' if r.solved else 'not solved'} in {r.steps:,} steps"
                     + ("" if r.solved else f" (final loss {r.final_loss})"))
            g = evolve(p, GAConfig(max_seconds=10, max_generations=3000), 1, trace=True)
            g.trace["label"] = "GA, box crossover"
            save(g.trace, f"ga_{p.id}_seed1.json",
                 f"{'solved' if g.solved else 'not solved'} after {g.generations:,} generations (best loss {g.best_loss})")
            # two memetic runs: the best setting (2,000 steps, usually solved by local search in
            # generation 0) and the 500-step one, which takes a few generations to evolve a solution
            for mem in (MEMETIC_BEST, MEMETIC):
                steps = mem["local_search"]
                g = evolve(p, GAConfig(max_seconds=10, max_generations=3000, **mem), 1, trace=True)
                g.trace["label"] = f"GA + local search (memetic, {steps} steps)"
                save(g.trace, f"ga_memetic{steps}_{p.id}_seed1.json",
                     f"{'solved' if g.solved else 'not solved'} after {g.generations:,} generations (best loss {g.best_loss})")
            for cfg, tag in ((DFSConfig("mrv", True), "mrv_fc"), (DFSConfig("first", False), "first")):
                d = dfs(p, cfg, trace=True)
                d.trace["label"] = cfg.label()
                save(d.trace, f"dfs_{tag}_{p.id}.json", f"solved with {d.nodes:,} nodes, {d.backtracks:,} backtracks")
        d = dfs(PUZZLES["extreme-01"], DFSConfig("mrv", True), trace=True)
        save(d.trace, "dfs_mrv_fc_extreme-01.json", f"solved with {d.nodes:,} nodes, {d.backtracks:,} backtracks")
        (tdir / "index.json").write_text(json.dumps({"format": "sudoku-trace-index-1", "traces": index}, indent=1) + "\n")
    record["commit"] = git_commit()
    (out / "E7.json").write_text(json.dumps(record, indent=1))


# ====================================================================== figures

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def plt_or_none():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: skipping figures (pip install matplotlib)")
        return None
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
        "text.color": INK, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })
    return plt


def compact(v):
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}K"
    return f"{v:g}"


def log_axis(ax, which="y"):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis = ax.yaxis if which == "y" else ax.xaxis
    (ax.set_yscale if which == "y" else ax.set_xscale)("log")
    lo, hi = ax.get_ylim() if which == "y" else ax.get_xlim()
    few = lo > 0 and hi / lo < 300   # under ~2.5 decades: label 1, 2 and 5 of each decade
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if few else (1.0,)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: compact(v)))
    axis.set_minor_formatter(NullFormatter())


def grouped_bars(ax, groups, series, value, colors, pct=False, log=False, label_tips=False, zero_marks=False):
    import numpy as np
    n = len(series)
    width = min(0.8 / max(n, 1), 0.22)
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        vals = [value(s, g) for g in groups]
        pos = x + (i - (n - 1) / 2) * width
        h = [v if v not in (None, "") else float("nan") for v in vals]
        ax.bar(pos, h, width * 0.92, color=colors[i % len(colors)], label=s, edgecolor=SURFACE, linewidth=0.6)
        for px, v in zip(pos, vals):
            if v in (None, ""):
                continue
            if label_tips or (zero_marks and v == 0):
                ax.text(px, v + (2 if pct else 0), f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, groups)
    ax.tick_params(axis="x", length=0)
    if pct:
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
    if log:
        log_axis(ax)


def bar_figure(plt, exp, rows, figs, title):
    summ = summarize(rows)
    idx = {(s["config"], s["level"]): s for s in summ}
    configs = config_order(rows)
    levels = [lv for lv in ALL_LEVELS if any(r["level"] == lv for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rate = lambda c, lv: 100 * idx[(c, lv)]["solve_rate"] if (c, lv) in idx else None
    steps = lambda c, lv: idx[(c, lv)]["median_steps_solved"] if (c, lv) in idx else None
    grouped_bars(axes[0], levels, configs, rate, PALETTE, pct=True, label_tips=len(configs) <= 6)
    axes[0].set_title("Solve rate (%)", loc="left")
    grouped_bars(axes[1], levels, configs, steps, PALETTE, log=True)
    axes[1].set_title("Median steps to solve (solved runs, log scale)", loc="left")
    fig.suptitle(f"{exp}. {title}", x=0.01, ha="left", fontsize=12, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(configs), 3), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1 + 0.05 * ((len(configs) - 1) // 3), 1, 0.95))
    fig.savefig(figs / f"{exp}.png", dpi=150)
    plt.close(fig)


def heatmap_e3(plt, rows, figs):
    from matplotlib.colors import LinearSegmentedColormap
    summ = summarize(rows)
    idx = {(s["config"], s["level"]): s for s in summ}
    T0s, alphas = ["auto", "0.3", "3.0"], ["0.9", "0.99", "0.999"]
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    for ax, level in zip(axes, list(LEVELS) + ["all"]):
        grid = [[100 * idx.get((f"T0={t} alpha={a} chain=100", level), {"solve_rate": float("nan")})["solve_rate"]
                 for a in alphas] for t in T0s]
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100)
        for i in range(len(T0s)):
            for j in range(len(alphas)):
                v = grid[i][j]
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                        color="#ffffff" if v >= 55 else INK)
        ax.set_xticks(range(len(alphas)), [f"α={a}" for a in alphas])
        ax.set_yticks(range(len(T0s)), [f"T0={t}" for t in T0s])
        ax.grid(False)
        ax.set_title(level, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes[-1], fraction=0.05, label="solve rate (%)")
    fig.suptitle("E3. Solve rate by start temperature and cooling rate (100 steps per temperature)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figs / "E3.png", dpi=150)
    plt.close(fig)


def figure_e5(plt, rows, figs):
    import numpy as np
    from matplotlib.colors import to_rgba
    summ = summarize(rows)
    idx = {(s["config"], s["level"]): s for s in summ}
    configs = config_order(rows)
    levels = list(LEVELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    grouped_bars(axes[0], levels, configs,
                 lambda c, lv: 100 * idx[(c, lv)]["solve_rate"] if (c, lv) in idx else None,
                 PALETTE, pct=True, label_tips=True)
    axes[0].set_title("Solve rate (%)", loc="left")
    n = len(configs)
    width = min(0.8 / n, 0.32)
    for i, c in enumerate(configs):
        data = [[max(r["steps"], 1) for r in rows if r["config"] == c and r["level"] == lv and r["solved"]]
                or [float("nan")] for lv in levels]
        pos = np.arange(len(levels)) + (i - (n - 1) / 2) * width
        bp = axes[1].boxplot(data, positions=pos, widths=width * 0.8, patch_artist=True, showfliers=True,
                             manage_ticks=False)
        color = PALETTE[i % len(PALETTE)]
        for b in bp["boxes"]:
            b.set(facecolor=to_rgba(color, 0.22), edgecolor=color)
        for m in bp["medians"]:
            m.set(color=color, linewidth=2)
        for k in ("whiskers", "caps"):
            for m in bp[k]:
                m.set(color=AXIS)
        for f in bp["fliers"]:
            f.set(marker="o", markersize=3, markerfacecolor=color, markeredgecolor=SURFACE, alpha=0.6)
    axes[1].set_xticks(range(len(levels)), levels)
    axes[1].tick_params(axis="x", length=0)
    log_axis(axes[1])
    axes[1].set_title("Steps to solve, solved runs (log scale)", loc="left")
    axes[1].text(0, -0.16, "A run plotted at 1 was solved by the start grid (constraint propagation), before any SA step.",
                 transform=axes[1].transAxes, fontsize=8, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(configs), bbox_to_anchor=(0.5, 0))
    fig.suptitle("E5. Best SA setup by level" + (", with random-restart hill climbing for comparison" if n > 1 else ""),
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    fig.savefig(figs / "E5.png", dpi=150)
    plt.close(fig)


def figure_e6(plt, rows, figs, exp="E6", title="Local search vs tree search, same wall-clock budget"):
    summ = summarize(rows)
    idx = {(s["config"], s["level"]): s for s in summ}
    configs = config_order(rows)
    levels = [lv for lv in ALL_LEVELS if any(r["level"] == lv for r in rows)]
    rate = lambda c, lv: 100 * idx[(c, lv)]["solve_rate"] if (c, lv) in idx else None
    secs = lambda c, lv: idx[(c, lv)]["median_seconds_solved"] or None if (c, lv) in idx else None

    if len(configs) > len(PALETTE):
        # More configs than colours that can be told apart: one row of charts per block
        # (small multiples), each with its own legend, and the same seconds scale in every row.
        blocks = []
        for c in configs:
            b = next(r["block"] for r in rows if r["config"] == c)
            if not blocks or blocks[-1][0] != b:
                blocks.append((b, []))
            blocks[-1][1].append(c)
        fig, axes = plt.subplots(len(blocks), 2, figsize=(14, 3.7 * len(blocks) + 0.8), squeeze=False)
        for i, (block, cfgs) in enumerate(blocks):
            grouped_bars(axes[i][0], levels, cfgs, rate, PALETTE, pct=True, zero_marks=True)
            grouped_bars(axes[i][1], levels, cfgs, secs, PALETTE, log=True)
            axes[i][0].set_title(f"{block.capitalize()}: solve rate within the time budget (%)", loc="left")
            axes[i][1].set_title(f"{block.capitalize()}: median seconds to solve (log scale)", loc="left")
            handles, labels = axes[i][0].get_legend_handles_labels()
            axes[i][1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5)
        lo = min(a[1].get_ylim()[0] for a in axes)
        hi = max(a[1].get_ylim()[1] for a in axes)
        for a in axes:
            a[1].set_ylim(lo, hi)
        fig.suptitle(f"{exp}. {title}", x=0.01, ha="left", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(figs / f"{exp}.png", dpi=150)
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4))
    grouped_bars(axes[0], levels, configs, rate, PALETTE, pct=True, zero_marks=True)
    axes[0].set_title("Solve rate within the time budget (%)", loc="left")
    grouped_bars(axes[1], levels, configs, secs, PALETTE, log=True)
    axes[1].set_title("Median seconds to solve (solved runs, log scale)", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
    fig.suptitle(f"{exp}. {title}", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(figs / f"{exp}.png", dpi=150)
    plt.close(fig)


def plot_e7(curves, names, figs, hill=None):
    plt = plt_or_none()
    if not plt:
        return
    from matplotlib.ticker import FuncFormatter
    rows = 3 if hill else 2
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3 * rows))
    styles = [(PALETTE[0], 1.3, 3), (PALETTE[1], 2.6, 2)]   # (color, width, z): first name drawn thin on top
    hill_color = PALETTE[2]

    def outcome(name, hist, solved, steps, restarts=None):
        short = name.split(" (")[0]
        extra = f", {restarts:,} restarts" if restarts else ""
        if solved:
            return f"{short}: solved at {steps:,}{extra}", steps
        last = hist[-1][1]
        flat = next((h[0] for h in hist if all(x[1] == last for x in hist[hist.index(h):])), steps)
        return f"{short}: stuck at loss {last:g} from ~{flat:,} to {steps:,}{extra}", flat

    for j, level in enumerate(LEVELS):
        notes, ends = [], []
        for name, (color, width, z) in zip(names, styles):
            hist, solved, steps, _ = curves[(level, name)]
            xs = [h[0] for h in hist]
            axes[0][j].plot(xs, [h[1] for h in hist], color=color, linewidth=width, zorder=z, label=name)
            axes[1][j].plot(xs, [h[2] for h in hist], color=color, linewidth=width, zorder=z, label=name)
            note, end = outcome(name, hist, solved, steps)
            notes.append(note)
            ends.append(end)
        # the SA rows share one step scale; hill climbing (often far longer) gets its own
        limits = {0: max(ends) * 1.25, 1: max(ends) * 1.25}
        if hill:
            hist, solved, steps, restarts = curves[(level, hill)]
            axes[2][j].plot([h[0] for h in hist], [h[1] for h in hist], color=hill_color, linewidth=1.0,
                            label=f"{hill} (temperature 0)")
            note, end = outcome(hill, hist, solved, steps, restarts)
            limits[2] = end * 1.1
            axes[2][j].set_title(f"{level}-01: hill climbing loss (own step scale)", loc="left", pad=18)
            axes[2][j].text(0, 1.03, note, transform=axes[2][j].transAxes, fontsize=7.5, color=INK2,
                            va="bottom", ha="left")
        for i in range(rows):
            ax = axes[i][j]
            ax.set_ylim(bottom=0)
            ax.set_xlim(0, limits[i])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: compact(v) if v else "0"))
            if i == 0:
                ax.tick_params(labelbottom=False)
        axes[0][j].set_title(f"{level}-01: loss", loc="left", pad=30)
        axes[0][j].text(0, 1.03, "\n".join(notes), transform=axes[0][j].transAxes, fontsize=7.5, color=INK2,
                        va="bottom", ha="left")
        axes[1][j].set_title("temperature", loc="left")
        axes[1][j].set_xlabel("step")
        if hill:
            axes[2][j].set_xlabel("step")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if hill:
        h2, l2 = axes[2][0].get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0))
    fig.suptitle("E7. SA loss (energy) and temperature during one run (seed 1)"
                 + (", and random-restart hill climbing" if hill else ""), x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(figs / "E7_curves.png", dpi=150)
    plt.close(fig)


def make_figures(out, figs):
    plt = plt_or_none()
    if not plt:
        return
    titles = {"E1": "Loss functions", "E2": "Initializations", "E4": "Restarts and reheating"}
    for exp, title in titles.items():
        if (out / f"{exp}_runs.csv").exists():
            bar_figure(plt, exp, read_csv(out / f"{exp}_runs.csv"), figs, title)
    if (out / "E3_runs.csv").exists():
        heatmap_e3(plt, read_csv(out / "E3_runs.csv"), figs)
    if (out / "E5_runs.csv").exists():
        figure_e5(plt, read_csv(out / "E5_runs.csv"), figs)
    if (out / "E6_runs.csv").exists():
        figure_e6(plt, read_csv(out / "E6_runs.csv"), figs)
    if (out / "E8_runs.csv").exists():
        figure_e6(plt, read_csv(out / "E8_runs.csv"), figs, "E8", "GA with local search, and what does the work (same budget)")
    if (out / "E7.json").exists():
        e7_curves(figs)
    print(f"wrote figures to {figs}")


# ====================================================================== main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", help="E1 ... E7 (default: all)")
    ap.add_argument("--quick", action="store_true", help="small sizes; writes to results/quick and figures/quick")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--report", action="store_true", help="only rebuild the summary and figures")
    args = ap.parse_args()

    size_name = "quick" if args.quick else "full"
    size = SIZES[size_name]
    out = ROOT / "results" / ("quick" if args.quick else "")
    figs = ROOT / "figures" / ("quick" if args.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    wanted = [e.upper() for e in args.experiments] or ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
    bad = [e for e in wanted if e not in QUESTIONS]
    if bad:
        sys.exit(f"unknown experiment(s): {', '.join(bad)}")

    if not args.report:
        t0 = time.time()
        for exp in wanted:
            if exp == "E7":
                print(f"\nE7: {QUESTIONS['E7']}", flush=True)
                e7(size, out, figs, write_traces=not args.quick)
                continue
            maker = {"E1": e1, "E2": e2, "E3": e3, "E4": e4}.get(exp)
            note, configs, ids, seeds = maker(size) if maker else {"E5": e5, "E6": e6, "E8": e8}[exp](size, out)
            run_experiment(exp, note, configs, ids, list(seeds), out, args.workers)
        print(f"\nall done in {(time.time() - t0) / 60:.1f} min")
    markdown_report(out, size_name)
    make_figures(out, figs)


if __name__ == "__main__":
    main()
