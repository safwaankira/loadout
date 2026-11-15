from __future__ import annotations

import collections
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
STOP = set(
    "a an the to of for and or in on my me i is it be can you please with from this that what at by as are do your "
    "want need would like could should will".split()
)

# Lookups whose answer changes from minute to minute: never served from memory.
VOLATILE = {
    "get_current_time",
    "get_battery_status",
    "get_notifications",
    "get_system_uptime",
    "get_steps_per_day",
    "get_current_weather",
    "get_bpm",
    "get_screen_time_data",
    "get_current_storage_space",
    "get_clock_timers",
    "search_file_system",
    "get_network_ip_address",
}
# Minutes a remembered answer stays fresh; anything not listed lasts until a setter changes it.
TTL_MINUTES = {"get_current_date": 600}


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def rememberable(tool: str) -> bool:
    return tool.startswith("get_") and tool not in VOLATILE


def invalidates(tool: str) -> str | None:
    return "get_" + tool[4:] if tool.startswith("set_") else None


@dataclass
class Tool:
    name: str
    description: str
    parameters: list
    kind: str
    depends_on: list = field(default_factory=list)

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    @property
    def tokens(self) -> int:
        return len(json.dumps(self.schema())) // 4


def load_tools() -> dict[str, Tool]:
    raw = json.loads((DATA / "core_tools.json").read_text()) + json.loads((DATA / "regular_tools.json").read_text())
    names = {t["name"] for t in raw}
    return {
        t["name"]: Tool(
            name=t["name"],
            description=t["description"],
            parameters=t.get("parameters", []),
            kind=t.get("func_type", "regular"),
            depends_on=[d for d in t.get("depends_on", []) if d["name"] in names],
        )
        for t in raw
    }


def load_tasks() -> list[dict]:
    return json.loads((DATA / "instances.json").read_text())


class BM25:
    def __init__(self, tools: dict[str, Tool]):
        self.docs = {n: tokenize(n.replace("_", " ") + " " + t.description) for n, t in tools.items()}
        self.tf = {n: collections.Counter(d) for n, d in self.docs.items()}
        self.n = len(self.docs)
        self.df = collections.Counter(w for d in self.docs.values() for w in set(d))
        self.avg = sum(map(len, self.docs.values())) / self.n

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        terms = tokenize(query)
        scores: dict[str, float] = {}
        for name, tf in self.tf.items():
            s = 0.0
            for w in terms:
                if w in tf:
                    idf = math.log(1 + (self.n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(self.docs[name]) / self.avg))
            if s:
                scores[name] = s
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
