import laya
import torch
import time

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading Laya on {device}...")
agent = laya.load("convaiinnovations/laya", device=device)
print("Model loaded successfully.")

def get_criteria(head, food, body_set, grid_w, grid_h, cur_dir):
    moves = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0)
    }
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    
    cur_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
    crit = {}
    analysis = {}

    for d, (dx, dy) in moves.items():
        nx, ny = head[0] + dx, head[1] + dy
        is_wall = not (0 <= nx < grid_w and 0 <= ny < grid_h)
        is_body = (nx, ny) in body_set
        is_reverse = (d == opposite.get(cur_dir))
        is_deadly = is_wall or is_body or is_reverse

        if is_deadly:
            reason = "CRITICAL COLLISION: " + ("Wall hit" if is_wall else ("180-degree neck snap" if is_reverse else "Self body crash"))
            crit[d] = f"Fatal move! {reason}. Immediate game over."
            analysis[d] = {"safe": False, "reason": reason}
        else:
            new_dist = abs(nx - food[0]) + abs(ny - food[1])
            is_food = (nx == food[0] and ny == food[1])
            if is_food:
                crit[d] = f"EXCELLENT: Directly eats food at ({nx}, {ny})! Safe and maximum reward."
                analysis[d] = {"safe": True, "dist": 0, "status": "Eats food"}
            elif new_dist < cur_dist:
                crit[d] = f"GOOD: Safe open path, moves closer to food (dist {new_dist} vs {cur_dist})."
                analysis[d] = {"safe": True, "dist": new_dist, "status": "Closer to food"}
            else:
                crit[d] = f"PASSABLE: Safe open move, but moves farther from food (dist {new_dist} vs {cur_dist})."
                analysis[d] = {"safe": True, "dist": new_dist, "status": "Away from food"}
                
    return crit, analysis

# Test a mock situation
head = [5, 5]
food = [5, 8]
body = [[5, 4], [5, 3], [5, 2]]
cur_dir = "DOWN"
crit, analysis = get_criteria(head, food, set(map(tuple, body)), 12, 12, cur_dir)

state = {
    "grid_size": "12x12",
    "snake_head": head,
    "food_position": food,
    "current_direction": cur_dir,
    "directions_evaluation": analysis,
    "goal": "Navigate snake towards food while avoiding all walls and body segments."
}

questions = {
    "next_move": {
        "type": "choice",
        "instructions": "Determine the single best move direction for the snake to survive and approach the food.",
        "criteria": crit
    }
}

t0 = time.time()
res = agent.predict(state, questions)
t1 = time.time()
print(f"Inference in {(t1-t0)*1000:.1f}ms:")
print("Chosen:", res["answers"]["next_move"]["choice"])
print("Probabilities:", res["answers"]["next_move"]["probabilities"])
print("Confidence:", res["answers"]["next_move"]["confidence"])
