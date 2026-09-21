from collections import deque
import laya
import torch
import time

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading Laya on {device}...")
agent = laya.load("convaiinnovations/laya", device=device)

def get_flood_fill_space(start_cell, obstacles, w, h, max_depth=60):
    """Calculate accessible empty cells from start_cell using BFS."""
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

def evaluate_moves(head, food, body, grid_size, cur_dir):
    moves = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    w, h = grid_size
    body_set = set(map(tuple, body))
    cur_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])

    criteria = {}
    analysis = {}
    safe_moves = []
    danger_moves = []

    for d, (dx, dy) in moves.items():
        nx, ny = head[0] + dx, head[1] + dy
        is_wall = not (0 <= nx < w and 0 <= ny < h)
        is_body = (nx, ny) in body_set
        is_rev = (d == opposite.get(cur_dir)) and len(body) > 1

        if is_wall or is_body or is_rev:
            reason = "Wall" if is_wall else ("Neck snap" if is_rev else "Body")
            danger_moves.append(d)
            criteria[d] = f"DANGER: Fatal collision with {reason}. Snake dies."
            analysis[d] = {"safe": False, "reason": reason}
        else:
            space = get_flood_fill_space((nx, ny), body_set, w, h)
            new_dist = abs(nx - food[0]) + abs(ny - food[1])
            is_food = (nx == food[0] and ny == food[1])
            
            safe_moves.append(d)
            analysis[d] = {
                "safe": True,
                "dist": new_dist,
                "space": space,
                "is_food": is_food
            }

            if is_food:
                criteria[d] = f"BEST: Eats food directly at ({nx},{ny})! High safety space ({space} cells)."
            elif space < min(len(body), 10) and len(body) > 3:
                criteria[d] = f"CAUTION: Trapped pocket! Only {space} cells of free space. High risk of suicide."
            elif new_dist < cur_dist:
                criteria[d] = f"BEST: Safe move closer to food (dist {new_dist}), wide open space ({space} cells)."
            else:
                criteria[d] = f"SAFE: Safe move with {space} open cells, but moves away from food (dist {new_dist})."

    # Situation summary for Laya
    situation = []
    if safe_moves:
        best_opts = [d for d in safe_moves if "BEST" in criteria[d]]
        if best_opts:
            situation.append(f"Recommended safe direction(s) towards food: {', '.join(best_opts)}.")
        else:
            situation.append(f"Safe directions: {', '.join(safe_moves)}.")
    if danger_moves:
        situation.append(f"Fatal directions to avoid: {', '.join(danger_moves)}.")

    state = {
        "snake_game": {
            "grid_size": f"{w}x{h}",
            "snake_head": head,
            "food": food,
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
            "instructions": "Which direction should the snake move to safely eat food and survive without hitting walls or itself?",
            "criteria": criteria
        }
    }

    return state, questions, analysis

# Quick test
state, q, analysis = evaluate_moves([5, 5], [5, 7], [[5, 4], [5, 3]], (12, 12), "DOWN")
res = agent.predict(state, q)
print("Decided:", res["answers"]["direction"])
