from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from selfgraph._ops import _paged

from .catalog import TTL_MINUTES, Tool

NS = os.environ.get("LOADOUT_NS", "loadout")
L_TOOL, L_MEMORY, L_RUN = "Tool", "Memory", "Run"
DEPENDS_ON, USED = "DEPENDS_ON", "USED"
FOREVER = datetime(9999, 1, 1, tzinfo=timezone.utc)


def open_store():
    """Open HydraDB when LOADOUT_HYDRADB_URI is set, otherwise an embedded Kuzu file for local work.

    @return (GraphStore): the store.
    """
    uri = os.environ.get("LOADOUT_HYDRADB_URI")
    if uri:
        from selfgraph import HydraDBStore

        return HydraDBStore(uri, auth=("neo4j", os.environ["LOADOUT_HYDRADB_TOKEN"]))
    from selfgraph import KuzuStore

    return KuzuStore(os.environ.get("LOADOUT_KUZU", "/tmp/loadout.kuzu"))


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat()


def parse(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class ContextGraph:
    """The agent's context layer: tools, their dependencies and the answers it already resolved, all in the graph."""

    def __init__(self, store, ns: str = NS):
        self.store, self.ns = store, ns

    async def init(self) -> None:
        await self.store.initialize()

    async def tool_count(self) -> int:
        return len(await _paged(lambda skip, limit: self.store.match_nodes(self.ns, L_TOOL, skip=skip, limit=limit)))

    async def load_catalog(self, tools: dict[str, Tool]) -> None:
        """Write every tool as a node and every dependency as an edge; safe to rerun.

        @param tools (dict[str, Tool]): the catalog.
        @return (None): nothing.
        """
        rows = [
            (
                t.name,
                {
                    "id": t.name,
                    "name": t.name,
                    "description": t.description,
                    "kind": t.kind,
                    "params_json": json.dumps(t.parameters),
                    "tokens": t.tokens,
                },
            )
            for t in tools.values()
        ]
        await self.store.upsert_nodes(self.ns, L_TOOL, rows)
        edges = []
        for t in tools.values():
            for d in t.depends_on:
                edges.append(
                    (
                        DEPENDS_ON,
                        t.name,
                        d["name"],
                        {
                            "id": f"{t.name}>{d['name']}",
                            "revision": 1,
                            "is_current": True,
                            "param": d.get("parameter_name") or "",
                            "how": d.get("dependence_type") or "",
                        },
                    )
                )
        for i in range(0, len(edges), 500):
            await self.store.create_edges(self.ns, L_TOOL, L_TOOL, edges[i : i + 500])

    async def closure(self, seeds: list[str], hops: int = 3) -> list[dict]:
        """Walk DEPENDS_ON from the seed tools, hop by hop, inside the graph.

        @param seeds (list[str]): the tools the task points at.
        @param hops (int): how far to follow dependencies.
        @return (list[dict]): one entry per tool reached: name, hop, the first tool that needed it, the parameter,
            and every tool in the set that depends on it.
        """
        out = {s: {"name": s, "hop": 0, "via": None, "param": None, "parents": []} for s in seeds}
        frontier = list(seeds)
        for hop in range(1, hops + 1):
            if not frontier:
                break
            edges = await _paged(
                lambda skip, limit, f=frontier: self.store.match_edges(
                    self.ns, from_label=L_TOOL, to_label=L_TOOL, type=DEPENDS_ON, from_in=f, skip=skip, limit=limit
                )
            )
            nxt = []
            for e in sorted(edges, key=lambda e: (frontier.index(e["from_id"]), e["props"]["id"])):
                to = e["to_id"]
                if to not in out:
                    out[to] = {"name": to, "hop": hop, "via": e["from_id"], "param": e["props"].get("param"), "parents": []}
                    nxt.append(to)
                if e["from_id"] not in out[to]["parents"]:
                    out[to]["parents"].append(e["from_id"])
            frontier = nxt
        return list(out.values())

    async def tools(self, names: list[str]) -> dict[str, dict]:
        if not names:
            return {}
        rows = await self.store.match_nodes(self.ns, L_TOOL, any_of={"id": names}, limit=len(names) + 1)
        return {r["id"]: r for r in rows}

    async def memories(self, agent: str, names: list[str]) -> dict[str, dict]:
        """Read what this agent already resolved for the given lookup tools.

        @param agent (str): the agent or session id.
        @param names (list[str]): lookup tool names.
        @return (dict[str, dict]): tool name to its memory node.
        """
        if not names:
            return {}
        ids = [f"mem:{agent}:{n}" for n in names]
        rows = await self.store.match_nodes(self.ns, L_MEMORY, any_of={"id": ids}, limit=len(ids) + 1)
        return {r["tool"]: r for r in rows}

    async def remember(self, agent: str, results: dict[str, Any], at: datetime) -> None:
        """Store answers the agent resolved, each with how long it stays fresh, keeping prior values as history.

        @param agent (str): the agent or session id.
        @param results (dict[str, Any]): lookup tool name to the value it returned.
        @param at (datetime): when it was resolved.
        @return (None): nothing.
        """
        if not results:
            return
        prior = await self.memories(agent, list(results))
        rows = []
        for tool, value in results.items():
            ttl = TTL_MINUTES.get(tool)
            until = FOREVER if ttl is None else at + timedelta(minutes=ttl)
            old = prior.get(tool)
            history = json.loads(old.get("history_json") or "[]") if old else []
            if old:
                history = (history + [{"value": old.get("value"), "at": old.get("resolved_at")}])[-5:]
            rows.append(
                (
                    f"mem:{agent}:{tool}",
                    {
                        "id": f"mem:{agent}:{tool}",
                        "agent": agent,
                        "tool": tool,
                        "value": json.dumps(value),
                        "resolved_at": iso(at),
                        "fresh_until": iso(until),
                        "invalidated_by": "",
                        "history_json": json.dumps(history),
                    },
                )
            )
        await self.store.upsert_nodes(self.ns, L_MEMORY, rows)

    async def invalidate(self, agent: str, pairs: dict[str, str], at: datetime) -> None:
        """Mark remembered answers stale because a setter just changed what they describe.

        @param agent (str): the agent or session id.
        @param pairs (dict[str, str]): lookup tool name to the setter that changed it.
        @param at (datetime): when.
        @return (None): nothing.
        """
        prior = await self.memories(agent, list(pairs))
        rows = [
            (m["id"], {**m, "fresh_until": iso(at), "invalidated_by": pairs[tool]})
            for tool, m in prior.items()
        ]
        if rows:
            await self.store.upsert_nodes(self.ns, L_MEMORY, rows)

    async def record_run(self, agent: str, run_id: str, task: str, used: list[str], at: datetime) -> None:
        await self.store.upsert_nodes(
            self.ns, L_RUN, [(run_id, {"id": run_id, "agent": agent, "task": task, "at": iso(at)})]
        )
        await self.store.create_edges(
            self.ns,
            L_RUN,
            L_TOOL,
            [(USED, run_id, t, {"id": f"{run_id}>{t}", "revision": 1, "is_current": True}) for t in used],
        )

    async def forget(self, agent: str) -> int:
        return await self.store.delete_nodes_by(self.ns, L_MEMORY, equals={"agent": agent})
