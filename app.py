from __future__ import annotations

import contextlib
import json
import random
import re
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from loadout.catalog import load_tasks
from loadout.engine import Loadout, now_utc
from loadout.evaluate import fake_value
from loadout.graph import ContextGraph, open_store

WEB = Path(__file__).resolve().parent / "web"
STATE: dict = {}
AGENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


async def engine() -> Loadout:
    if "lo" not in STATE:
        graph = ContextGraph(open_store())
        await graph.init()
        lo = Loadout(graph)
        if await graph.tool_count() < len(lo.tools):
            await graph.load_catalog(lo.tools)
        STATE["lo"] = lo
        STATE["tasks"] = load_tasks()
    return STATE["lo"]


def agent_of(value: object) -> str:
    agent = str(value or "demo")
    return agent if AGENT_RE.match(agent) else "demo"


def task_of(value: object) -> str:
    task = str(value or "").strip()[:400]
    if not task:
        raise ValueError("task is empty")
    return task


async def plan_payload(task: str, agent: str) -> dict:
    lo = await engine()
    plan = await lo.plan(task, agent, now_utc())
    return {**plan.to_dict(), "tools_total": len(lo.tools), "schemas": lo.schemas(plan)}


async def api_plan(request: Request) -> JSONResponse:
    body = await request.json()
    try:
        return JSONResponse(await plan_payload(task_of(body.get("task")), agent_of(body.get("agent"))))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def api_run(request: Request) -> JSONResponse:
    """Plan a task, run the loaded tools against a simulated device, and write back what the run learned."""
    body = await request.json()
    try:
        task, agent = task_of(body.get("task")), agent_of(body.get("agent"))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    lo = await engine()
    payload = await plan_payload(task, agent)
    results = {
        i["name"]: fake_value(i["name"], task)
        for i in payload["loaded"]
        if not i["name"].startswith("set_") or i["hop"] == 0
    }
    learned = await lo.report(agent, task, results, now_utc())
    return JSONResponse({**payload, "calls": len(results), "learned": learned})


async def api_reset(request: Request) -> JSONResponse:
    body = await request.json()
    lo = await engine()
    return JSONResponse({"forgot": await lo.graph.forget(agent_of(body.get("agent")))})


async def api_examples(request: Request) -> JSONResponse:
    await engine()
    good = json.loads((WEB / "examples.json").read_text())
    return JSONResponse(random.Random().sample(good, 8))


async def api_results(request: Request) -> JSONResponse:
    return JSONResponse(json.loads((WEB / "results.json").read_text()))


async def health(request: Request) -> JSONResponse:
    lo = await engine()
    return JSONResponse({"ok": True, "backend": type(lo.graph.store).__name__, "tools": len(lo.tools)})


async def index(request: Request) -> FileResponse:
    return FileResponse(WEB / "index.html")


mcp = MCPServer("loadout", instructions="Ask get_context before a task; call report_run after it.")


@mcp.tool()
async def get_context(task: str, agent: str = "default") -> dict:
    """Return only the tool schemas this task needs, plus answers this agent already resolved.

    @param task (str): the task in plain words.
    @param agent (str): your agent or session id; memory is kept per agent.
    @return (dict): tools to load, remembered answers, and token counts.
    """
    payload = await plan_payload(task_of(task), agent_of(agent))
    return {
        "tools": payload["schemas"],
        "remembered": {i["name"]: i["value"] for i in payload["remembered"]},
        "tokens_loaded": payload["tokens_loaded"],
        "tokens_if_all_tools": payload["tokens_all"],
    }


@mcp.tool()
async def report_run(task: str, results: dict, agent: str = "default") -> dict:
    """Write back what a finished run returned, so later tasks can skip lookups that are still fresh.

    @param task (str): the task that ran.
    @param results (dict): tool name to the value it returned.
    @param agent (str): your agent or session id.
    @return (dict): what was remembered and what a setter made stale.
    """
    lo = await engine()
    return await lo.report(agent_of(agent), task_of(task), results, now_utc())


mcp_app = mcp.streamable_http_app()


@contextlib.asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        await engine()
        yield


app = Starlette(
    routes=[
        Route("/", index),
        Route("/healthz", health),
        Route("/api/plan", api_plan, methods=["POST"]),
        Route("/api/run", api_run, methods=["POST"]),
        Route("/api/reset", api_reset, methods=["POST"]),
        Route("/api/examples", api_examples),
        Route("/api/results", api_results),
        Mount("/static", StaticFiles(directory=WEB)),
        Mount("/", mcp_app),
    ],
    lifespan=lifespan,
)
