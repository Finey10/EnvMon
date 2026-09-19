"""
Orchestrator — Pipeline with LangGraph (preferred) or plain-Python fallback.

Tries to import LangGraph. If LangGraph / Pydantic is incompatible with the
running Python version (e.g. Python 3.14+), falls back to a simple sequential
function-call pipeline that produces the identical PipelineState output.

Run order: run_air → run_water → run_litter → run_coordinator
"""

from __future__ import annotations

from typing import TypedDict, Optional

from agents import air_agent, water_agent, litter_agent, coordinator_agent

# ── Canonical zone list (matches mock_zones.csv) ─────────────────────────────
ZONES = [
    "Zone 1 - Riverside Industrial",
    "Zone 2 - Central Market",
    "Zone 3 - Lakeside Residential",
    "Zone 4 - Outer Ring Road",
]


# ── Shared pipeline state ────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    zone:               str
    image_path:         str
    air_result:         Optional[dict]
    water_result:       Optional[dict]
    litter_result:      Optional[dict]
    coordinator_result: Optional[dict]
    errors:             list[str]


# ── Node logic (shared by both execution paths) ──────────────────────────────

def run_air(state: PipelineState) -> PipelineState:
    try:
        return {**state, "air_result": air_agent.run(state["zone"])}
    except Exception as exc:
        errors = state.get("errors", []) + [f"Air agent error: {exc}"]
        fallback = {"signal": "air", "severity": "low", "value": 0,
                    "note": f"⚠️ Air agent unavailable: {exc}", "components": {}}
        return {**state, "air_result": fallback, "errors": errors}


def run_water(state: PipelineState) -> PipelineState:
    try:
        return {**state, "water_result": water_agent.run(state["zone"])}
    except Exception as exc:
        errors = state.get("errors", []) + [f"Water agent error: {exc}"]
        fallback = {"signal": "water", "severity": "low", "value": 0,
                    "note": f"⚠️ Water agent unavailable: {exc}"}
        return {**state, "water_result": fallback, "errors": errors}


def run_litter(state: PipelineState) -> PipelineState:
    try:
        return {**state, "litter_result": litter_agent.run(state["image_path"])}
    except Exception as exc:
        errors = state.get("errors", []) + [f"Litter agent error: {exc}"]
        fallback = {"signal": "litter", "severity": "low", "value": 0,
                    "note": f"⚠️ Litter agent unavailable: {exc}", "boxes": []}
        return {**state, "litter_result": fallback, "errors": errors}


def run_coordinator(state: PipelineState) -> PipelineState:
    try:
        result = coordinator_agent.run(
            air_result=state["air_result"],
            water_result=state["water_result"],
            litter_result=state["litter_result"],
        )
        return {**state, "coordinator_result": result}
    except Exception as exc:
        errors = state.get("errors", []) + [f"Coordinator agent error: {exc}"]
        fallback = {"overall_risk": "unknown",
                    "reasoning": f"Coordinator agent encountered an error: {exc}",
                    "recommended_action": "Check API credentials and retry."}
        return {**state, "coordinator_result": fallback, "errors": errors}


# ── Execution backend selection ───────────────────────────────────────────────

def _try_build_langgraph():
    """
    Attempt to build the LangGraph pipeline.
    Returns compiled graph on success, None on failure (e.g. Python 3.14+
    Pydantic V1 incompatibility).
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress Pydantic V1 warning
            from langgraph.graph import StateGraph, END

        graph = StateGraph(PipelineState)
        graph.add_node("run_air",         run_air)
        graph.add_node("run_water",       run_water)
        graph.add_node("run_litter",      run_litter)
        graph.add_node("run_coordinator", run_coordinator)
        graph.set_entry_point("run_air")
        graph.add_edge("run_air",         "run_water")
        graph.add_edge("run_water",       "run_litter")
        graph.add_edge("run_litter",      "run_coordinator")
        graph.add_edge("run_coordinator", END)
        compiled = graph.compile()
        # Quick smoke-test: make sure .invoke is callable
        _ = compiled.invoke   # noqa
        return compiled
    except Exception as e:
        import sys
        print(f"[orchestrator] LangGraph unavailable ({e}), using plain-Python fallback.",
              file=sys.stderr)
        return None


_langgraph_pipeline = _try_build_langgraph()


def _run_plain_python(state: PipelineState) -> PipelineState:
    """Fallback sequential pipeline — identical logic, no LangGraph dependency."""
    state = run_air(state)
    state = run_water(state)
    state = run_litter(state)
    state = run_coordinator(state)
    return state


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(zone: str, image_path: str) -> PipelineState:
    """
    Execute the full monitoring pipeline end-to-end.

    Args:
        zone:       Zone name from mock_zones.csv
        image_path: Path to image file for litter detection

    Returns:
        Populated PipelineState with all four agent results.
    """
    initial: PipelineState = {
        "zone":               zone,
        "image_path":         image_path,
        "air_result":         None,
        "water_result":       None,
        "litter_result":      None,
        "coordinator_result": None,
        "errors":             [],
    }

    if _langgraph_pipeline is not None:
        return _langgraph_pipeline.invoke(initial)
    else:
        return _run_plain_python(initial)


if __name__ == "__main__":
    import json, sys
    zone  = sys.argv[1] if len(sys.argv) > 1 else ZONES[0]
    image = sys.argv[2] if len(sys.argv) > 2 else "sample_litter_1.jpg"

    backend = "LangGraph" if _langgraph_pipeline else "plain-Python"
    print(f"Backend: {backend}  |  zone={zone!r}  |  image={image!r}\n")
    state = run_pipeline(zone, image)

    for key in ("air_result", "water_result", "coordinator_result"):
        print(f"--- {key} ---")
        print(json.dumps(state[key], indent=2))
        print()

    litter = state["litter_result"]
    print("--- litter_result (no boxes) ---")
    print(json.dumps({k: v for k, v in litter.items() if k != "boxes"}, indent=2))
    print(f"  box_count: {len(litter.get('boxes', []))}")

    if state.get("errors"):
        print("\n--- Pipeline Errors ---")
        for e in state["errors"]:
            print(f"  * {e}")
