import asyncio
from collections import deque
import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import laya
from pydantic import BaseModel
import torch

app = FastAPI(title="Laya Neural Snake")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device selection: Apple Silicon Metal (MPS) or CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[*] Initializing Laya Model on device: {device}...")
t_start = time.time()
agent = laya.load("convaiinnovations/laya", device=device)
print(f"[*] Laya Model ready in {time.time() - t_start:.2f}s!")

def bfs_shortest_path(start, target, obstacles, w, h):
    """BFS to find the shortest obstacle-avoiding path length between start and target."""
    if start == target:
        return 0
    if start in obstacles or not (0 <= start[0] < w and 0 <= start[1] < h):
        return None
    visited = {start}
    q = deque([(start[0], start[1], 0)])
    while q:
        cx, cy, dist = q.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if nx == target[0] and ny == target[1]:
                return dist + 1
            if 0 <= nx < w and 0 <= ny < h:
                if (nx, ny) not in obstacles and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny, dist + 1))
    return None

def get_flood_fill_space(start_cell, obstacles, w, h, max_depth=None):
    """BFS to count reachable free cells from start_cell."""
    if max_depth is None:
        max_depth = max(150, w * h)
    if start_cell in obstacles or not (0 <= start_cell[0] < w and 0 <= start_cell[1] < h):
        return 0
    visited = {start_cell}
    q = deque([start_cell])
    count = 0
    while q and count < max_depth:
        cx, cy = q.popleft()
        count += 1
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                if (nx, ny) not in obstacles and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
    return count

def build_laya_prompt(head: List[int], food: List[int], body: List[List[int]], grid_size: List[int], cur_dir: str):
    w, h = grid_size
    moves = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    body_set = set(map(tuple, body))
    tail = tuple(body[-1]) if body else tuple(head)

    cur_manhattan = abs(head[0] - food[0]) + abs(head[1] - food[1])
    cur_bfs_food = bfs_shortest_path(tuple(head), tuple(food), body_set, w, h)

    criteria = {}
    analysis = {}
    safe_moves = []
    danger_moves = []

    # 180-degree opposite direction is physically forbidden in Snake
    forbidden_reverse = opposite.get(cur_dir) if len(body) >= 1 else None

    for d, (dx, dy) in moves.items():
        # Completely exclude the reverse direction from decision space
        if d == forbidden_reverse:
            analysis[d] = {"safe": False, "reason": "Physically Forbidden (Reverse)"}
            continue

        nx, ny = head[0] + dx, head[1] + dy
        is_wall = not (0 <= nx < w and 0 <= ny < h)
        is_body = (nx, ny) in body_set

        if is_wall or is_body:
            reason = "Wall" if is_wall else "Body Collision"
            danger_moves.append(d)
            criteria[d] = f"DANGER: Fatal collision with {reason}. Snake dies."
            analysis[d] = {"safe": False, "reason": reason}
        else:
            is_food = (nx == food[0] and ny == food[1])
            new_pos = (nx, ny)
            
            # Virtual next step obstacle simulation for tail reachability
            if is_food:
                # Eating food: tail does NOT pop, body grows
                sim_obstacles = body_set | {tuple(head)}
                target_tail = tail
            else:
                # Moving: tail will vacate unless snake is tiny
                sim_obstacles = (body_set | {tuple(head)}) - {tail}
                target_tail = tuple(body[-2]) if len(body) >= 2 else tuple(head)

            # Check if escape route to tail exists
            tail_dist = bfs_shortest_path(new_pos, target_tail, sim_obstacles, w, h)
            can_escape_to_tail = (tail_dist is not None)

            # True BFS path to food from new position
            food_path_len = bfs_shortest_path(new_pos, tuple(food), sim_obstacles, w, h)
            space = get_flood_fill_space(new_pos, sim_obstacles, w, h, max_depth=w * h)

            safe_moves.append(d)
            analysis[d] = {
                "safe": True,
                "space": space,
                "is_food": is_food,
                "can_reach_tail": can_escape_to_tail,
                "tail_dist": tail_dist,
                "food_path_len": food_path_len
            }

            # Generate smart criteria
            if is_food:
                if can_escape_to_tail:
                    criteria[d] = f"BEST: Eats food directly! Escape route to tail is open and safe ({space} free cells)."
                else:
                    criteria[d] = f"TRAP WARNING: Eats food but gets sealed inside coils with no exit to tail! High risk."
            elif can_escape_to_tail:
                if food_path_len is not None and (cur_bfs_food is None or food_path_len < cur_bfs_food):
                    criteria[d] = f"BEST: Safe move directly closer to food (path: {food_path_len} steps), escape route to tail guaranteed ({space} free cells)."
                else:
                    criteria[d] = f"SAFE WANDER: Safe open path ({space} free cells) with guaranteed route to tail. Good maneuvering."
            else:
                if space > len(body) * 1.5:
                    criteria[d] = f"PASSABLE: Open area ({space} cells), but no direct path to tail."
                else:
                    criteria[d] = f"CAUTION: Pocket dead-end ({space} cells) with NO route to tail. Likely suicide trap."

    # Situation summary for Laya
    situation = []
    if safe_moves:
        best_opts = [d for d in safe_moves if "BEST" in criteria[d]]
        if best_opts:
            situation.append(f"Recommended safe direction(s) towards food: {', '.join(best_opts)}.")
        else:
            safe_wander = [d for d in safe_moves if "SAFE" in criteria[d]]
            if safe_wander:
                situation.append(f"Safe wandering / tail-chasing direction(s): {', '.join(safe_wander)}.")
            else:
                situation.append(f"Safe directions: {', '.join(safe_moves)}.")
    if danger_moves:
        situation.append(f"Fatal directions to avoid: {', '.join(danger_moves)}.")

    state = {
        "snake_game": {
            "grid_size": f"{w}x{h}",
            "snake_head": head,
            "food_location": food,
            "snake_length": len(body),
            "current_direction": cur_dir,
            "safe_directions": safe_moves,
            "danger_directions": danger_moves,
            "situation": " ".join(situation)
        }
    }

    questions = {
        "direction": {
            "type": "choice",
            "instructions": "Which direction should the snake move to safely eat food and survive without hitting walls or getting trapped?",
            "criteria": criteria
        }
    }

    return state, questions, analysis

class PredictRequest(BaseModel):
    head: List[int]
    food: List[int]
    body: List[List[int]]
    grid_size: List[int] = [14, 14]
    current_direction: str = "RIGHT"

@app.post("/api/predict")
def predict_move(req: PredictRequest):
    state, questions, analysis = build_laya_prompt(
        req.head, req.food, req.body, req.grid_size, req.current_direction
    )

    t0 = time.perf_counter()
    res = agent.predict(state, questions)
    t1 = time.perf_counter()

    ans = res["answers"]["direction"]
    choice = ans["choice"]
    probs = ans["probabilities"]
    confidence = ans.get("confidence", 0.0)
    inference_ms = round((t1 - t0) * 1000, 1)

    # Fallback safety check: if chosen move is fatal and there's a safe move, alert
    is_safe = analysis.get(choice, {}).get("safe", False)

    return {
        "choice": choice,
        "probabilities": probs,
        "confidence": confidence,
        "inference_ms": inference_ms,
        "is_safe": is_safe,
        "analysis": analysis,
        "state_sent": state,
        "criteria_sent": questions["direction"]["criteria"]
    }

@app.websocket("/ws/play")
async def websocket_play(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            head = payload["head"]
            food = payload["food"]
            body = payload["body"]
            grid_size = payload.get("grid_size", [14, 14])
            cur_dir = payload.get("current_direction", "RIGHT")

            state, questions, analysis = build_laya_prompt(head, food, body, grid_size, cur_dir)

            t0 = time.perf_counter()
            # Run inference in worker thread to prevent blocking event loop
            res = await asyncio.to_thread(agent.predict, state, questions)
            t1 = time.perf_counter()

            ans = res["answers"]["direction"]
            choice = ans["choice"]
            probs = ans["probabilities"]
            confidence = ans.get("confidence", 0.0)
            inference_ms = round((t1 - t0) * 1000, 1)

            resp = {
                "step_id": payload.get("step_id", 0),
                "choice": choice,
                "probabilities": probs,
                "confidence": confidence,
                "inference_ms": inference_ms,
                "analysis": analysis,
                "state_sent": state,
                "criteria_sent": questions["direction"]["criteria"]
            }
            await websocket.send_text(json.dumps(resp))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def get_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)

