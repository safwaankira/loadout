from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .catalog import load_tasks, rememberable
from .engine import FIND_TOOL_TOKENS, Loadout
from .graph import ContextGraph, open_store

OUT = Path(__file__).resolve().parent.parent / "web" / "results.json"


def fake_value(tool: str, task: str) -> str:
    return f"{tool}#{hashlib.sha1((tool + task).encode()).hexdigest()[:6]}"


def required(lo: Loadout, main: str, golden: set[str], remembered: set[str]) -> set[str]:
    """The ground-truth tools a run still needs once remembered answers stand in for the tools that fed them.

    @param lo (Loadout): the engine, for the dependency edges.
    @param main (str): the task's main tool.
    @param golden (set[str]): every tool the ground-truth run used.
    @param remembered (set[str]): tools whose answers are remembered fresh.
    @return (set[str]): the tools still needed, remembered ones included.
    """
    def walk(start: str, stop: set[str]) -> set[str]:
        seen, todo = {start}, [start]
        while todo:
            cur = todo.pop()
            if cur in stop:
                continue
            for d in lo.tools[cur].depends_on:
                if d["name"] not in seen:
                    seen.add(d["name"])
                    todo.append(d["name"])
        return seen

    full = walk(main, set())
    reach = walk(main, remembered)
    return {g for g in golden if g in reach or g not in full}


async def run(n_tasks: int = 600, round_size: int = 25, seed: int = 7, out: Path = OUT) -> dict:
    """Replay one agent session of ToolLinkOS tasks and measure each round against three baselines.

    A task counts as covered when every tool its ground-truth successful run used is loaded or already
    remembered fresh. A miss is recovered through find_tool at one extra call. Calls count the tools the
    run still had to execute: remembered answers are not called again.

    @param n_tasks (int): how many tasks the session replays.
    @param round_size (int): tasks per measured round.
    @param seed (int): shuffle seed, so runs are reproducible.
    @param out (Path): where to write the results.
    @return (dict): the results.
    """
    store = open_store()
    graph = ContextGraph(store, ns=f"eval{seed}")
    await graph.init()
    lo = Loadout(graph)
    if await graph.tool_count() < len(lo.tools):
        await graph.load_catalog(lo.tools)
    agent = f"eval-{seed}-{int(time.time())}"
    tasks = load_tasks()
    random.Random(seed).shuffle(tasks)
    tasks = tasks[:n_tasks]
    clock = datetime(2026, 11, 15, 9, 0, tzinfo=timezone.utc)
    rounds = []
    acc = {k: {"covered": 0, "calls": 0, "tokens": 0, "queries": 0} for k in ("all", "bm25", "graph", "loadout")}
    hits = lookups = 0
    started = time.perf_counter()
    plan_ms = []
    for idx, t in enumerate(tasks):
        clock += timedelta(minutes=3)
        golden = set(t["golden_function_names"])
        q = t["user_query"]
        acc["all"]["covered"] += 1
        acc["all"]["calls"] += len(golden)
        acc["all"]["tokens"] += lo.tokens_all
        top = {n for n, _ in lo.bm25.search(q, k=12)}
        acc["bm25"]["covered"] += golden <= top
        acc["bm25"]["calls"] += len(golden) + (0 if golden <= top else 1)
        acc["bm25"]["tokens"] += sum(lo.tools[n].tokens for n in top)
        base = await lo.plan(q, agent, clock, memory=False)
        ok = golden <= base.loaded_names
        acc["graph"]["covered"] += ok
        acc["graph"]["calls"] += len(golden) + (0 if ok else 1)
        acc["graph"]["tokens"] += base.tokens_loaded
        t0 = time.perf_counter()
        plan = await lo.plan(q, agent, clock)
        plan_ms.append((time.perf_counter() - t0) * 1000)
        need = required(lo, t["main_golden_function_name"], golden, plan.remembered_names)
        ok = need <= plan.loaded_names | plan.remembered_names
        calls = len(need - plan.remembered_names) + (0 if ok else 1)
        acc["loadout"]["covered"] += ok
        acc["loadout"]["calls"] += calls
        acc["loadout"]["tokens"] += plan.tokens_loaded + (0 if ok else FIND_TOOL_TOKENS * 4)
        results = {g: fake_value(g, q) for g in need - plan.remembered_names}
        looked = {g for g in golden if rememberable(g)}
        lookups += len(looked)
        hits += len(looked & plan.remembered_names)
        await lo.report(agent, q, results, clock)
        if (idx + 1) % round_size == 0:
            row = {"round": (idx + 1) // round_size, "tasks": idx + 1}
            for k, v in acc.items():
                row[k] = {
                    "covered": round(v["covered"] / round_size, 3),
                    "calls": round(v["calls"] / round_size, 2),
                    "tokens": round(v["tokens"] / round_size),
                }
                acc[k] = {"covered": 0, "calls": 0, "tokens": 0, "queries": 0}
            row["memory_hit_rate"] = round(hits / lookups, 3) if lookups else 0
            hits = lookups = 0
            rounds.append(row)
            print(json.dumps(row), flush=True)
    plan_ms.sort()
    res = {
        "dataset": "ToolLinkOS (573 tools, 1,569 tasks with ground-truth tool sets)",
        "backend": type(store).__name__,
        "tasks": n_tasks,
        "round_size": round_size,
        "tools": len(lo.tools),
        "tokens_all": lo.tokens_all,
        "plan_ms_p50": round(plan_ms[len(plan_ms) // 2], 1),
        "plan_ms_p95": round(plan_ms[int(len(plan_ms) * 0.95)], 1),
        "seconds": round(time.perf_counter() - started, 1),
        "rounds": rounds,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    await graph.forget(agent)
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    asyncio.run(run(n))
