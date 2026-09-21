"""FastAPI and WebSocket Server for Laya Neural Snake Game."""

import asyncio
from collections import deque
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import laya
from pydantic import BaseModel
import torch

from snake_core import (
    DecisionArbiter,
    MOVES,
    Point,
    PromptBuilder,
    SafetyAnalyzer,
)

app = FastAPI(title="Laya Neural Snake")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[*] Initializing Laya Model on device: {device}...")
agent = laya.load("convaiinnovations/laya", device=device)
print("[*] Laya Model ready!")


class GameStateRequest(BaseModel):
    head: List[int]
    food: List[int]
    body: List[List[int]]
    grid_size: List[int] = [10, 10]
    cur_dir: Optional[str] = None
    current_direction: Optional[str] = None
    steps_since_food: int = 0
    recent_heads: Optional[List[List[int]]] = None


def build_laya_prompt(
    head: List[int],
    food: List[int],
    body: List[List[int]],
    grid_size: List[int],
    cur_dir: str,
    steps_since_food: int = 0,
    recent_heads: Optional[List[List[int]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Builds Laya prompt using snake_core modular analyzers."""
    head_pt: Point = (head[0], head[1])
    food_pt: Point = (food[0], food[1])
    body_pts: List[Point] = [tuple(b) for b in body]
    grid_sz: Tuple[int, int] = (grid_size[0], grid_size[1])
    recent_pts = [tuple(p) for p in recent_heads] if recent_heads else None

    analysis, verified_food_dir, food_path = SafetyAnalyzer.analyze(
        head=head_pt,
        food=food_pt,
        body=body_pts,
        grid_size=grid_sz,
        cur_dir=cur_dir,
        recent_heads=recent_pts,
    )

    state, questions = PromptBuilder.build(
        head=head_pt,
        food=food_pt,
        body=body_pts,
        grid_size=grid_sz,
        cur_dir=cur_dir,
        steps_since_food=steps_since_food,
        analysis=analysis,
        verified_food_dir=verified_food_dir,
        food_path=food_path,
    )

    return state, questions, analysis


def select_safe_action(
    choice: str,
    analysis: Dict[str, Any],
    probs: Dict[str, float],
    steps_since_food: int = 0,
) -> Tuple[str, bool]:
    """Delegates to DecisionArbiter for safety and anti-looping arbitration."""
    return DecisionArbiter.select_action(choice, analysis, probs, steps_since_food)


@app.post("/api/predict")
async def predict_direction(req: GameStateRequest):
    t0 = time.perf_counter()
    recent = req.recent_heads or []
    direction = req.cur_dir or req.current_direction or "RIGHT"
    state, questions, analysis = build_laya_prompt(
        req.head, req.food, req.body, req.grid_size, direction, req.steps_since_food, recent
    )

    results = agent.predict(state, questions)
    direction_ans = results["answers"]["direction"]
    raw_choice = direction_ans["choice"]
    probs = direction_ans["probabilities"]
    confidence = direction_ans.get("confidence", 0.0)

    final_choice, overridden = select_safe_action(raw_choice, analysis, probs, req.steps_since_food)
    t1 = time.perf_counter()

    return {
        "choice": final_choice,
        "raw_model_choice": raw_choice,
        "overridden": overridden,
        "probabilities": probs,
        "confidence": confidence,
        "inference_ms": round((t1 - t0) * 1000, 1),
        "is_safe": analysis.get(final_choice, {}).get("safe", False),
        "analysis": analysis,
        "state_sent": state,
        "criteria_sent": questions["direction"]["criteria"],
    }


@app.websocket("/ws/play")
async def websocket_play(websocket: WebSocket):
    await websocket.accept()
    recent_heads: deque = deque(maxlen=24)
    steps_since_food = 0

    try:
        while True:
            data = await websocket.receive_text()
            t0 = time.perf_counter()
            game_state = json.loads(data)

            head = game_state["head"]
            food = game_state["food"]
            body = game_state["body"]
            grid_size = game_state.get("grid_size", [10, 10])
            cur_dir = game_state.get("cur_dir") or game_state.get("current_direction") or "RIGHT"
            steps_since_food = game_state.get("steps_since_food", steps_since_food)
            step_id = game_state.get("step_id")

            state, questions, analysis = build_laya_prompt(
                head, food, body, grid_size, cur_dir, steps_since_food, list(recent_heads)
            )

            results = agent.predict(state, questions)
            direction_ans = results["answers"]["direction"]
            raw_choice = direction_ans["choice"]
            probs = direction_ans["probabilities"]
            confidence = direction_ans.get("confidence", 0.0)

            final_choice, overridden = select_safe_action(raw_choice, analysis, probs, steps_since_food)
            t1 = time.perf_counter()

            recent_heads.append(tuple(head))
            if head[0] == food[0] and head[1] == food[1]:
                steps_since_food = 0
                recent_heads.clear()
            else:
                steps_since_food += 1

            await websocket.send_text(
                json.dumps(
                    {
                        "step_id": step_id,
                        "choice": final_choice,
                        "direction": final_choice,
                        "raw_choice": raw_choice,
                        "raw_model_choice": raw_choice,
                        "overridden": overridden,
                        "probabilities": probs,
                        "confidence": confidence,
                        "inference_ms": round((t1 - t0) * 1000, 1),
                        "is_safe": analysis.get(final_choice, {}).get("safe", False),
                        "analysis": analysis,
                        "state_sent": state,
                        "criteria_sent": questions["direction"]["criteria"],
                    }
                )
            )
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket Error: {e}")


# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def get_index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Laya Neural Snake API server running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8088)
