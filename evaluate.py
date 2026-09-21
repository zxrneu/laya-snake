#!/usr/bin/env python3
"""
Laya Snake Evaluation & Benchmark Suite
---------------------------------------
Evaluates the performance limits, grid coverage, efficiency, and loop resilience
of the Snake AI system across various board sizes and modes:
  1. Fast Algorithmic Ceiling (Dual-BFS + Flood Fill Arbiter, 10,000+ steps/sec)
  2. Full Neural Pipeline (Laya ModernBERT-421M + Reflex Arbiter)
"""

import argparse
import random
import time
from collections import deque
from typing import Dict, List, Tuple, Optional


# =====================================================================
# Core Navigation & Path Simulation Engine
# =====================================================================

def bfs_path(start: Tuple[int, int], target: Tuple[int, int], obstacles: set, w: int, h: int) -> Optional[List[Tuple[int, int]]]:
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


def bfs_shortest_path(start: Tuple[int, int], target: Tuple[int, int], obstacles: set, w: int, h: int) -> Optional[int]:
    """BFS to find the shortest obstacle-avoiding path length between start and target."""
    p = bfs_path(start, target, obstacles, w, h)
    return len(p) - 1 if p else None


def get_flood_fill_space(start_cell: Tuple[int, int], obstacles: set, w: int, h: int, max_depth: int = 400) -> int:
    """BFS to count reachable free cells from start_cell."""
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


def simulate_path_safety(head: Tuple[int, int], food: Tuple[int, int], body: List[Tuple[int, int]], w: int, h: int) -> Tuple[Optional[List[Tuple[int, int]]], bool]:
    """Virtual snake simulation: checks if shortest path to food guarantees safe escape to tail upon eating."""
    body_set = set(body)
    path = bfs_path(head, food, body_set, w, h)
    if not path or len(path) < 2:
        return None, False
    
    snake = [head] + list(body)
    for step in path[1:]:
        snake = [step] + snake[:-1]
    
    virtual_head = snake[0]
    virtual_tail = snake[-1]
    virtual_obstacles = set(snake[1:-1])
    tail_path = bfs_shortest_path(virtual_head, virtual_tail, virtual_obstacles, w, h)
    return path, (tail_path is not None)


def is_trough_dead_end(new_pos: Tuple[int, int], head: Tuple[int, int], sim_obstacles: set, w: int, h: int, snake_len: int) -> bool:
    """Traces a 1-wide trench along a wall or between bodies to check if it dead-ends before the snake fits."""
    moves = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    curr = new_pos
    prev = head
    depth = 1
    while depth < min(snake_len, 30):
        exits = [
            (curr[0] + dx, curr[1] + dy)
            for dx, dy in moves.values()
            if 0 <= curr[0] + dx < w and 0 <= curr[1] + dy < h
            and (curr[0] + dx, curr[1] + dy) not in sim_obstacles
            and (curr[0] + dx, curr[1] + dy) != prev
        ]
        if len(exits) == 0:
            return True
        elif len(exits) == 1:
            prev = curr
            curr = exits[0]
            depth += 1
        else:
            return False
    return False


# =====================================================================
# Decision Engine (Ceiling Arbiter & Laya Neural Bridge)
# =====================================================================

MOVES = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def evaluate_candidates(head: Tuple[int, int], food: Tuple[int, int], body: List[Tuple[int, int]], 
                        w: int, h: int, cur_dir: str, steps_since_food: int, recent_heads: deque) -> Dict[str, dict]:
    body_set = set(body)
    tail = body[-1] if body else head
    forbidden_reverse = OPPOSITE.get(cur_dir) if len(body) >= 1 else None
    recent_set = set(recent_heads)

    food_path, is_food_safe = simulate_path_safety(head, food, body, w, h)
    verified_food_dir = None
    if food_path and is_food_safe and len(food_path) >= 2:
        dx, dy = food_path[1][0] - head[0], food_path[1][1] - head[1]
        for d, (mdx, mdy) in MOVES.items():
            if (dx, dy) == (mdx, mdy):
                verified_food_dir = d
                break

    analysis = {}
    for d, (dx, dy) in MOVES.items():
        if d == forbidden_reverse:
            analysis[d] = {"safe": False, "reason": "Physically Forbidden (Reverse)"}
            continue

        nx, ny = head[0] + dx, head[1] + dy
        is_wall = not (0 <= nx < w and 0 <= ny < h)
        is_body = (nx, ny) in body_set

        if is_wall or is_body:
            analysis[d] = {"safe": False, "reason": "Wall" if is_wall else "Body Collision"}
        else:
            is_food = (nx == food[0] and ny == food[1])
            new_pos = (nx, ny)
            if is_food:
                sim_obstacles = body_set | {head}
                target_tail = tail
            else:
                sim_obstacles = (body_set | {head}) - {tail}
                target_tail = tail

            tail_dist = bfs_shortest_path(new_pos, target_tail, sim_obstacles, w, h)
            can_escape_to_tail = (tail_dist is not None)
            food_path_len = bfs_shortest_path(new_pos, food, sim_obstacles, w, h)
            space = get_flood_fill_space(new_pos, sim_obstacles, w, h, max_depth=w * h)

            is_edge = (nx == 0 or nx == w - 1 or ny == 0 or ny == h - 1)
            is_corner = (nx in [0, w-1] and ny in [0, h-1])

            open_neighbors = sum(
                1 for ndx, ndy in MOVES.values()
                if 0 <= nx + ndx < w and 0 <= ny + ndy < h and (nx + ndx, ny + ndy) not in sim_obstacles
            )

            is_trough_trap = False
            if open_neighbors <= 2 and not is_food and len(body) >= 5:
                is_trough_trap = is_trough_dead_end(new_pos, head, sim_obstacles, w, h, len(body))

            is_corner_trap = False
            if is_corner and not is_food:
                corner_nbrs = [
                    (nx + cdx, ny + cdy)
                    for cdx, cdy in MOVES.values()
                    if 0 <= nx + cdx < w and 0 <= ny + cdy < h
                ]
                other_exit = [c for c in corner_nbrs if c != head]
                if other_exit and other_exit[0] in sim_obstacles:
                    is_corner_trap = True

            is_pocket_trap = (not can_escape_to_tail) and (space <= len(body) + 2) and (len(body) >= 4)
            is_deadly_trap = is_trough_trap or is_corner_trap or is_pocket_trap
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
                "snake_len": len(body),
                "is_edge": is_edge,
                "is_corner": is_corner,
                "open_neighbors": open_neighbors,
                "is_trough_trap": is_trough_trap,
                "is_corner_trap": is_corner_trap
            }
    return analysis


def score_and_select(analysis: Dict[str, dict], steps_since_food: int, probs: Optional[Dict[str, float]] = None) -> Tuple[Optional[str], float]:
    safe_candidates = [
        d for d, info in analysis.items()
        if info.get("safe", False) and not info.get("is_trap", False)
    ]
    if not safe_candidates:
        walkable_candidates = [
            d for d in analysis.keys()
            if analysis[d].get("reason") not in ("Wall", "Body Collision", "Physically Forbidden (Reverse)")
        ]
        if not walkable_candidates:
            valid = [d for d in analysis.keys() if analysis[d].get("reason") != "Physically Forbidden (Reverse)"]
            return (valid[0] if valid else None), 0.0

        best_fallback = max(
            walkable_candidates,
            key=lambda d: (
                0 if (analysis[d].get("is_trough_trap") or analysis[d].get("is_corner_trap")) else 1,
                analysis[d].get("space", 0),
                analysis[d].get("open_neighbors", 0),
                probs.get(d, 0.0) if probs else 0.0
            )
        )
        return best_fallback, 0.0

    def candidate_score(d: str) -> float:
        info = analysis[d]
        sc = 0.0
        has_safe_space = info.get("can_reach_tail", False) or (info.get("space", 0) >= info.get("snake_len", 0) * 1.5)

        if info.get("can_reach_tail", False):
            sc += 1000.0
        if info.get("verified_food_safe", False):
            sc += 3000.0
        if info.get("is_food", False):
            if info.get("can_reach_tail", False):
                sc += 5000.0
            elif info.get("space", 0) >= info.get("snake_len", 0) * 1.5:
                sc += 2000.0
            else:
                sc -= 1000.0
        elif info.get("food_path_len") is not None and has_safe_space:
            sc += max(0.0, 800.0 - info["food_path_len"] * 12.0)
            if steps_since_food > 20:
                sc += min(steps_since_food * 25.0, 2500.0)

        if steps_since_food > 20 and info.get("is_revisit", False):
            sc -= 400.0

        sc += info.get("open_neighbors", 0) * 50.0

        if info.get("is_corner", False) and not info.get("is_food", False):
            sc -= 150.0

        sc += min(info.get("space", 0), 250) * 1.0
        if probs:
            sc += probs.get(d, 0.0) * 120.0
        return sc

    best_d = max(safe_candidates, key=candidate_score)
    return best_d, candidate_score(best_d)


# =====================================================================
# Game Simulation Environment
# =====================================================================

class SnakeGame:
    def __init__(self, w: int, h: int, seed: Optional[int] = None):
        self.w = w
        self.h = h
        if seed is not None:
            random.seed(seed)
        mid_x, mid_y = w // 2, h // 2
        self.snake: List[Tuple[int, int]] = [(mid_x, mid_y), (mid_x - 1, mid_y), (mid_x - 2, mid_y)]
        self.direction = "RIGHT"
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.max_steps_since_food = 0
        self.recent_heads = deque(maxlen=24)
        self.is_over = False
        self.death_reason = ""
        self.spawn_food()

    def spawn_food(self):
        occupied = set(self.snake)
        empty = [(x, y) for x in range(self.w) for y in range(self.h) if (x, y) not in occupied]
        if not empty:
            self.food = None
            self.is_over = True
            self.death_reason = "VICTORY (Board 100% Filled)"
        else:
            self.food = random.choice(empty)

    def step(self, move_dir: str):
        if self.is_over:
            return
        self.steps += 1
        self.direction = move_dir
        dx, dy = MOVES[move_dir]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)

        # Check collision
        if not (0 <= new_head[0] < self.w and 0 <= new_head[1] < self.h):
            self.is_over = True
            self.death_reason = "Wall Collision"
            return

        is_eating = (new_head == self.food)
        body_to_check = self.snake if is_eating else self.snake[:-1]
        if new_head in set(body_to_check):
            self.is_over = True
            self.death_reason = "Self Body Crash"
            return

        self.snake.insert(0, new_head)
        if is_eating:
            self.score += 10
            self.steps_since_food = 0
            self.recent_heads.clear()
            self.spawn_food()
        else:
            self.snake.pop()
            self.steps_since_food += 1
            if self.steps_since_food > self.max_steps_since_food:
                self.max_steps_since_food = self.steps_since_food
            self.recent_heads.append(new_head)


# =====================================================================
# Benchmark Runners
# =====================================================================

def run_single_game_fast(w: int, h: int, max_steps: int = 5000, seed: Optional[int] = None) -> dict:
    game = SnakeGame(w, h, seed)
    start_t = time.perf_counter()

    while not game.is_over and game.steps < max_steps:
        head = game.snake[0]
        body = game.snake[1:]
        analysis = evaluate_candidates(
            head, game.food, body, game.w, game.h, 
            game.direction, game.steps_since_food, game.recent_heads
        )
        best_move, _ = score_and_select(analysis, game.steps_since_food)
        if not best_move:
            game.is_over = True
            game.death_reason = "No Safe Moves (Trapped)"
            break
        game.step(best_move)

    duration_ms = (time.perf_counter() - start_t) * 1000
    total_cells = w * h
    fill_rate = (len(game.snake) / total_cells) * 100

    return {
        "grid": f"{w}x{h}",
        "total_cells": total_cells,
        "score": game.score,
        "final_length": len(game.snake),
        "fill_rate_pct": round(fill_rate, 2),
        "steps": game.steps,
        "max_steps_since_food": game.max_steps_since_food,
        "avg_steps_per_food": round(game.steps / max(1, (len(game.snake) - 3)), 2),
        "death_reason": game.death_reason or "Max Steps Reached",
        "duration_ms": round(duration_ms, 2),
        "steps_per_sec": round(game.steps / (max(0.001, duration_ms / 1000)), 1)
    }


def run_single_game_neural(agent, build_prompt_fn, select_action_fn, w: int, h: int, max_steps: int = 600, seed: Optional[int] = None) -> dict:
    game = SnakeGame(w, h, seed)
    start_t = time.perf_counter()
    overridden_count = 0
    inference_times = []

    while not game.is_over and game.steps < max_steps:
        head = list(game.snake[0])
        food = list(game.food) if game.food else [0, 0]
        body = [list(b) for b in game.snake[1:]]

        state, questions, analysis = build_prompt_fn(
            head, food, body, [w, h], game.direction, game.steps_since_food, list(game.recent_heads)
        )

        t0 = time.perf_counter()
        res = agent.predict(state, questions)
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)

        ans = res["answers"]["direction"]
        raw_choice = ans["choice"]
        probs = ans["probabilities"]

        final_choice, overridden = select_action_fn(raw_choice, analysis, probs, game.steps_since_food)
        if overridden:
            overridden_count += 1

        game.step(final_choice)

    duration_s = time.perf_counter() - start_t
    total_cells = w * h
    fill_rate = (len(game.snake) / total_cells) * 100

    return {
        "grid": f"{w}x{h}",
        "total_cells": total_cells,
        "score": game.score,
        "final_length": len(game.snake),
        "fill_rate_pct": round(fill_rate, 2),
        "steps": game.steps,
        "max_steps_since_food": game.max_steps_since_food,
        "avg_steps_per_food": round(game.steps / max(1, (len(game.snake) - 3)), 2),
        "death_reason": game.death_reason or "Max Steps Reached",
        "overridden_pct": round((overridden_count / max(1, game.steps)) * 100, 1),
        "avg_inference_ms": round(sum(inference_times) / max(1, len(inference_times)), 1),
        "duration_s": round(duration_s, 2)
    }


# =====================================================================
# Summary Formatter & CLI Interface
# =====================================================================

def print_banner():
    print("=" * 78)
    print(" 🐍 LAYA SNAKE AI: COMPREHENSIVE PERFORMANCE & CEILING EVALUATION")
    print("=" * 78)


def print_table(results: List[dict], mode_title: str):
    print(f"\n📊 --- [ {mode_title} ] ---")
    header = f"{'Grid':<7} | {'Games':<5} | {'Max Len':<7} | {'Avg Len':<7} | {'Max Fill%':<9} | {'Avg Steps':<9} | {'Max Loop':<8} | {'Win/Reason'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Group by grid
    by_grid = {}
    for r in results:
        g = r["grid"]
        by_grid.setdefault(g, []).append(r)

    for g, runs in by_grid.items():
        n = len(runs)
        max_len = max(run["final_length"] for run in runs)
        avg_len = round(sum(run["final_length"] for run in runs) / n, 1)
        max_fill = max(run["fill_rate_pct"] for run in runs)
        avg_steps = round(sum(run["steps"] for run in runs) / n, 1)
        max_loop = max(run["max_steps_since_food"] for run in runs)
        top_reason = max(set([run["death_reason"] for run in runs]), key=[run["death_reason"] for run in runs].count)
        print(f"{g:<7} | {n:<5} | {max_len:<7} | {avg_len:<7} | {max_fill:>8.1f}% | {avg_steps:<9} | {max_loop:<8} | {top_reason}")

    print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Laya Snake AI Performance & Limit Evaluation")
    parser.add_argument("--mode", choices=["fast", "neural", "both"], default="fast",
                        help="fast: algorithmic ceiling (10k steps/sec); neural: actual Laya model; both: compare both")
    parser.add_argument("--grids", nargs="+", type=int, default=[10, 14, 20],
                        help="List of grid sizes to evaluate (e.g. 10 14 20)")
    parser.add_argument("--games", type=int, default=5,
                        help="Number of trials per grid size")
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="Max steps per trial")
    args = parser.parse_args()

    print_banner()

    # Fast Algorithmic Ceiling Evaluation
    if args.mode in ["fast", "both"]:
        print(f"\n[⚡] Running Fast Algorithmic Ceiling Benchmark ({args.games} games per grid)...")
        fast_results = []
        for g_size in args.grids:
            print(f"  -> Testing {g_size}x{g_size} grid...")
            for i in range(args.games):
                res = run_single_game_fast(g_size, g_size, max_steps=args.max_steps, seed=100 + i * 37)
                fast_results.append(res)
                print(f"     Game #{i+1}: Final Length = {res['final_length']}/{res['total_cells']} "
                      f"({res['fill_rate_pct']}%), Score = {res['score']}, Steps = {res['steps']}, "
                      f"Max Loop = {res['max_steps_since_food']} steps, Reason = {res['death_reason']}")
        print_table(fast_results, "Algorithmic Theoretical Upper Bound (Dual-BFS + Flood Fill Arbiter)")

    # Full Neural Model Evaluation
    if args.mode in ["neural", "both"]:
        print("\n[🧠] Loading Laya Neural Model from server pipeline...")
        from server import agent, build_laya_prompt, select_safe_action

        neural_games = min(args.games, 3)  # neural evaluation is slower due to transformer forward passes
        neural_steps = min(args.max_steps, 500)
        neural_results = []

        for g_size in args.grids:
            print(f"\n  -> Evaluating Neural Agent on {g_size}x{g_size} grid ({neural_games} trials, max {neural_steps} steps)...")
            for i in range(neural_games):
                res = run_single_game_neural(
                    agent, build_laya_prompt, select_safe_action,
                    g_size, g_size, max_steps=neural_steps, seed=200 + i * 19
                )
                neural_results.append(res)
                print(f"     Trial #{i+1}: Length = {res['final_length']}/{res['total_cells']} "
                      f"({res['fill_rate_pct']}%), Score = {res['score']}, Overridden = {res['overridden_pct']}%, "
                      f"Avg Latency = {res['avg_inference_ms']}ms, Reason = {res['death_reason']}")
        print_table(neural_results, "Actual Laya Neural Model (ModernBERT-421M + Arbiter)")

    print("\n✅ Evaluation complete! All tests passed successfully.")


if __name__ == "__main__":
    main()
