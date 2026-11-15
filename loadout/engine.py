from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .catalog import BM25, Tool, invalidates, load_tools, rememberable
from .graph import ContextGraph, parse

FIND_TOOL = {
    "name": "find_tool",
    "description": "Search the full tool catalog when none of the loaded tools fits; returns matching tool schemas.",
    "parameters": [{"name": "query", "type": "string", "description": "What the tool should do.", "required": True}],
}
FIND_TOOL_TOKENS = len(json.dumps(FIND_TOOL)) // 4


@dataclass
class Item:
    name: str
    reason: str
    hop: int
    via: str | None = None
    param: str | None = None
    tokens: int = 0
    value: object = None
    resolved_at: str | None = None


@dataclass
class Plan:
    task: str
    loaded: list[Item] = field(default_factory=list)
    remembered: list[Item] = field(default_factory=list)
    tokens_loaded: int = 0
    tokens_all: int = 0

    @property
    def loaded_names(self) -> set[str]:
        return {i.name for i in self.loaded}

    @property
    def remembered_names(self) -> set[str]:
        return {i.name for i in self.remembered}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["saved_pct"] = round(100 * (1 - self.tokens_loaded / self.tokens_all), 1) if self.tokens_all else 0
        return d


class Loadout:
    def __init__(self, graph: ContextGraph, tools: dict[str, Tool] | None = None):
        self.graph = graph
        self.tools = tools or load_tools()
        self.bm25 = BM25(self.tools)
        self.tokens_all = sum(t.tokens for t in self.tools.values())

    async def plan(self, task: str, agent: str, now: datetime, *, memory: bool = True, seeds: int = 3) -> Plan:
        """Pick what the agent needs for one task: the target tools, the tools they depend on, minus what it remembers.

        @param task (str): the task in plain words.
        @param agent (str): the agent or session id whose memory to use.
        @param now (datetime): the current time, for freshness.
        @param memory (bool): use remembered answers; False gives the no-memory baseline.
        @param seeds (int): how many target tools to start from.
        @return (Plan): what to load and what is already known.
        """
        targets = [n for n, _ in self.bm25.search(task, k=seeds)]
        reached = await self.graph.closure(targets)
        known = {}
        if memory:
            lookups = [r["name"] for r in reached if rememberable(r["name"])]
            for tool, m in (await self.graph.memories(agent, lookups)).items():
                if parse(m["fresh_until"]) > now:
                    known[tool] = m
        plan = Plan(task=task, tokens_all=self.tokens_all + FIND_TOOL_TOKENS)
        needed = set()
        for r in reached:
            name = r["name"]
            if r["via"] is not None:
                live = [p for p in r["parents"] if p in needed and p not in known]
                if not live:
                    continue
                r = {**r, "via": live[0]}
            needed.add(name)
            if name in known:
                m = known[name]
                plan.remembered.append(
                    Item(name, "remembered", r["hop"], r["via"], r["param"], 0, json.loads(m["value"]), m["resolved_at"])
                )
            else:
                reason = "target" if r["hop"] == 0 else "dependency"
                plan.loaded.append(Item(name, reason, r["hop"], r["via"], r["param"], self.tools[name].tokens))
        plan.tokens_loaded = sum(i.tokens for i in plan.loaded) + FIND_TOOL_TOKENS
        return plan

    def schemas(self, plan: Plan) -> list[dict]:
        return [self.tools[i.name].schema() for i in plan.loaded] + [FIND_TOOL]

    async def report(self, agent: str, task: str, results: dict[str, object], now: datetime) -> dict:
        """Write back what a finished run learned: answers worth remembering, and what its setters made stale.

        @param agent (str): the agent or session id.
        @param task (str): the task.
        @param results (dict[str, object]): every tool the run called, with what it returned.
        @param now (datetime): when the run finished.
        @return (dict): what was remembered and what was invalidated.
        """
        keep = {t: v for t, v in results.items() if rememberable(t)}
        stale = {invalidates(t): t for t in results if invalidates(t)}
        stale = {g: s for g, s in stale.items() if g in self.tools}
        await self.graph.remember(agent, keep, now)
        await self.graph.invalidate(agent, stale, now)
        await self.graph.record_run(agent, f"run:{uuid.uuid4().hex[:12]}", task, list(results), now)
        return {"remembered": sorted(keep), "invalidated": sorted(stale)}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
