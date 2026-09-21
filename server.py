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

def bfs_path(start, target, obstacles, w, h):
    """BFS to find the shortest obstacle-avoiding coordinate path between start and target."""
    if start == target:
        return [start]
    if start in obstacles or not (0 <= start[0] < w and 0 <= start[1] < h):
        return None
    parent = {start: None}
    q = deque([start])
    found = False
    while q:
        curr = q.popleft()
        if curr == target:
            found = True
            break
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = curr[0] + dx, curr[1] + dy
            nxt = (nx, ny)
            if 0 <= nx < w and 0 <= ny < h and nxt not in obstacles and nxt not in parent:
                parent[nxt] = curr
                q.append(nxt)
    if not found:
        return None
    curr = target
    path = []
    while curr is not None:
        path.append(curr)
        curr = parent[curr]
    path.reverse()
    return path

def bfs_shortest_path(start, target, obstacles, w, h):
    """BFS to find the shortest obstacle-avoiding path length between start and target."""
    p = bfs_path(start, target, obstacles, w, h)
    return len(p) - 1 if p else None

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

def simulate_path_safety(head, food, body, w, h):
    """Virtual snake simulation: checks if shortest path to food allows safe escape to tail upon eating."""
    body_set = set(map(tuple, body))
    path = bfs_path(tuple(head), tuple(food), body_set, w, h)
    if not path or len(path) < 2:
        return None, False
    
    snake = [tuple(head)] + [tuple(b) for b in body]
    for step in path[1:]:
        snake = [step] + snake[:-1]
    
    virtual_head = snake[0]
    virtual_tail = snake[-1]
    virtual_obstacles = set(snake[1:-1])
    tail_path = bfs_shortest_path(virtual_head, virtual_tail, virtual_obstacles, w, h)
    return path, (tail_path is not None)

def build_laya_prompt(head: List[int], food: List[int], body: List[List[int]], grid_size: List[int], cur_dir: str, steps_since_food: int = 0, recent_heads: list = None):
    w, h = grid_size
    moves = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    body_set = set(map(tuple, body))
    tail = tuple(body[-1]) if body else tuple(head)
    recent_set = set(recent_heads) if recent_heads else set()

    # Pre-simulate shortest food path safety
    food_path, is_food_safe = simulate_path_safety(head, food, body, w, h)
    verified_food_dir = None
    if food_path and is_food_safe and len(food_path) >= 2:
        dx, dy = food_path[1][0] - head[0], food_path[1][1] - head[1]
        for d, (mdx, mdy) in moves.items():
            if (dx, dy) == (mdx, mdy):
                verified_food_dir = d
                break

    cur_bfs_food = bfs_shortest_path(tuple(head), tuple(food), body_set, w, h)

    criteria = {}
    analysis = {}
    safe_moves = []
    danger_moves = []

    # 180-degree opposite direction is physically forbidden in Snake
    forbidden_reverse = opposite.get(cur_dir) if len(body) >= 1 else None

    for d, (dx, dy) in moves.items():
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
                sim_obstacles = body_set | {tuple(head)}
                target_tail = tail
            else:
                sim_obstacles = (body_set | {tuple(head)}) - {tail}
                target_tail = tail

            tail_dist = bfs_shortest_path(new_pos, target_tail, sim_obstacles, w, h)
            can_escape_to_tail = (tail_dist is not None)
            food_path_len = bfs_shortest_path(new_pos, tuple(food), sim_obstacles, w, h)
            space = get_flood_fill_space(new_pos, sim_obstacles, w, h, max_depth=w * h)

            # Pocket Trap Detection:
            is_deadly_trap = (not can_escape_to_tail) and (space < len(body)) and (len(body) >= 4)
            is_revisit = (new_pos in recent_set)

            analysis[d] = {
                "safe": not is_deadly_trap,
                "is_trap": is_deadly_trap,
                "space": space,
                "is_food": is_food,
                "can_reach_tail": can_escape_to_tail,
                "tail_dist": tail_dist,
                "food_path_len": food_path_len,
                "verified_food_safe": (d == verified_food_dir),
                "is_revisit": is_revisit,
                "snake_len": len(body)
            }

            if is_deadly_trap:
                reason = f"Dead-end pocket ({space} cells < length {len(body)})"
                analysis[d]["reason"] = reason
                danger_moves.append(d)
                criteria[d] = f"DANGER: Suicide trap! Only {space} cells available (snake length is {len(body)}). Fatal body crash."
            else:
                safe_moves.append(d)
                has_safe_space = can_escape_to_tail or (space >= len(body) * 1.5)
                
                # Smart criteria: Prioritize Safe Food Hunting and Breaking Tail Loops
                if is_food:
                    if has_safe_space:
                        criteria[d] = f"CRITICAL BEST: Direct hit! Eats food at ({nx}, {ny}) with guaranteed safe space ({space} cells)."
                    else:
                        criteria[d] = f"TRAP WARNING: Eats food but risks being trapped in coils ({space} cells). High risk."
                elif d == verified_food_dir:
                    criteria[d] = f"CRITICAL BEST: 100% verified safe shortest path to FOOD ({len(food_path) - 1} steps). Guaranteed exit to tail after eating. MUST take to eat fruit and break loops!"
                elif food_path_len is not None and has_safe_space:
                    criteria[d] = f"BEST: Safe approach to food (path: {food_path_len} steps, {space} free cells). Breaks out of tail loop."
                elif can_escape_to_tail or space >= len(body) * 2:
                    if verified_food_dir is not None:
                        criteria[d] = f"AVOID LOOP: Circles tail away from food ({space} free cells). Food is safely accessible via {verified_food_dir}; do not loop!"
                    else:
                        criteria[d] = f"DEFENSIVE: Safe tail-following survival ({space} free cells) until food path clears."
                else:
                    criteria[d] = f"CAUTION: Narrow territory ({space} cells) with no direct tail route."

    # If all moves were classified as deadly traps, pick the one with maximum space
    if not safe_moves and danger_moves:
        best_trap_move = max(danger_moves, key=lambda d: analysis[d].get("space", 0))
        criteria[best_trap_move] = f"EMERGENCY: Maximum reachable free space ({analysis[best_trap_move].get('space', 0)} cells). Best survival chance."
        safe_moves.append(best_trap_move)

    # Situation summary for Laya
    situation = []
    if safe_moves:
        if verified_food_dir:
            situation.append(f"CRITICAL: Direct safe path to fruit is OPEN in direction: {verified_food_dir}! Advance towards food.")
        else:
            best_opts = [d for d in safe_moves if "BEST" in criteria.get(d, "")]
            if best_opts:
                situation.append(f"Recommended safe direction(s) towards food: {', '.join(best_opts)}.")
            else:
                situation.append(f"Safe defensive moves: {', '.join(safe_moves)}.")
    if danger_moves:
        situation.append(f"Fatal directions to avoid: {', '.join(danger_moves)}.")

    state = {
        "snake_game": {
            "grid_size": f"{w}x{h}",
            "snake_head": head,
            "food_location": food,
            "snake_length": len(body),
            "current_direction": cur_dir,
            "steps_since_food": steps_since_food,
            "safe_directions": safe_moves,
            "danger_directions": danger_moves,
            "situation": " ".join(situation)
        }
    }

    questions = {
        "direction": {
            "type": "choice",
            "instructions": "Which direction should the snake move to safely eat food and advance without getting trapped or looping in its own body?",
            "criteria": criteria
        }
    }

    return state, questions, analysis

def select_safe_action(choice: str, analysis: Dict[str, Any], probs: Dict[str, float], steps_since_food: int = 0) -> tuple[str, bool]:
    """Safety Reflex Arbiter:
    Ensures the snake avoids suicide, but aggressively hunts food when safe to prevent infinite tail loops.
    """
    safe_candidates = [
        d for d, info in analysis.items()
        if info.get("safe", False) and not info.get("is_trap", False)
    ]

    # If no 100% clean safe candidate exists, fallback to move with largest space
    if not safe_candidates:
        valid_candidates = [d for d in analysis.keys() if analysis[d].get("reason") != "Physically Forbidden (Reverse)"]
        if not valid_candidates:
            return choice, False
        best_fallback = max(
            valid_candidates,
            key=lambda d: (
                analysis[d].get("space", 0),
                probs.get(d, 0)
            )
        )
        return best_fallback, True

    # Candidate scoring: prioritize verified food advancement over endless circling
    def candidate_score(d: str) -> float:
        info = analysis[d]
        sc = 0.0
        has_safe_space = info.get("can_reach_tail", False) or (info.get("space", 0) >= info.get("snake_len", 0) * 1.5)

        if info.get("can_reach_tail", False):
            sc += 1000.0
        if info.get("verified_food_safe", False):
            sc += 3000.0
        if info.get("is_food", False) and has_safe_space:
            sc += 5000.0
        elif info.get("food_path_len") is not None and has_safe_space:
            sc += max(0.0, 800.0 - info["food_path_len"] * 12.0)
            if steps_since_food > 20:
                sc += min(steps_since_food * 25.0, 2500.0)

        # Anti-Looping Urgency: penalize revisiting recent cells if we have stalled without food
        if steps_since_food > 20 and info.get("is_revisit", False):
            sc -= 400.0

        sc += min(info.get("space", 0), 250) * 1.0
        sc += probs.get(d, 0.0) * 120.0
        return sc

    best_safe = max(safe_candidates, key=candidate_score)

    # Check if raw model choice is safe and of comparable quality
    choice_info = analysis.get(choice, {})
    if choice_info.get("safe", False) and not choice_info.get("is_trap", False):
        choice_score = candidate_score(choice)
        best_score = candidate_score(best_safe)
        if choice == best_safe or (best_score - choice_score < 200.0):
            return choice, False

    return best_safe, True

class PredictRequest(BaseModel):
    head: List[int]
    food: List[int]
    body: List[List[int]]
    grid_size: List[int] = [14, 14]
    current_direction: str = "RIGHT"
    steps_since_food: int = 0

@app.post("/api/predict")
def predict_move(req: PredictRequest):
    state, questions, analysis = build_laya_prompt(
        req.head, req.food, req.body, req.grid_size, req.current_direction, req.steps_since_food
    )

    t0 = time.perf_counter()
    res = agent.predict(state, questions)
    t1 = time.perf_counter()

    ans = res["answers"]["direction"]
    raw_choice = ans["choice"]
    probs = ans["probabilities"]
    confidence = ans.get("confidence", 0.0)
    inference_ms = round((t1 - t0) * 1000, 1)

    # Apply Safety Reflex Arbiter
    final_choice, overridden = select_safe_action(raw_choice, analysis, probs, req.steps_since_food)
    is_safe = analysis.get(final_choice, {}).get("safe", False)

    return {
        "choice": final_choice,
        "raw_model_choice": raw_choice,
        "overridden": overridden,
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
    recent_heads = deque(maxlen=24)
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            head = payload["head"]
            food = payload["food"]
            body = payload["body"]
            grid_size = payload.get("grid_size", [14, 14])
            cur_dir = payload.get("current_direction", "RIGHT")
            steps_since_food = payload.get("steps_since_food", 0)

            if steps_since_food == 0:
                recent_heads.clear()
            recent_heads.append(tuple(head))

            state, questions, analysis = build_laya_prompt(
                head, food, body, grid_size, cur_dir, steps_since_food, list(recent_heads)
            )

            t0 = time.perf_counter()
            # Run inference in worker thread to prevent blocking event loop
            res = await asyncio.to_thread(agent.predict, state, questions)
            t1 = time.perf_counter()

            ans = res["answers"]["direction"]
            raw_choice = ans["choice"]
            probs = ans["probabilities"]
            confidence = ans.get("confidence", 0.0)
            inference_ms = round((t1 - t0) * 1000, 1)

            # Apply Safety Reflex Arbiter
            final_choice, overridden = select_safe_action(raw_choice, analysis, probs, steps_since_food)

            resp = {
                "step_id": payload.get("step_id", 0),
                "choice": final_choice,
                "raw_model_choice": raw_choice,
                "overridden": overridden,
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

