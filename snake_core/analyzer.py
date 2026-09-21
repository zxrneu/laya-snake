"""Spatial candidate analyzer and trap detection for Snake AI."""

from typing import Any, Collection, Dict, List, Optional, Set, Tuple

from snake_core.constants import MOVES, OPPOSITE, Point
from snake_core.geometry import (
    bfs_shortest_path,
    get_flood_fill_space,
    is_trough_dead_end,
    simulate_path_safety,
)


class SafetyAnalyzer:
    """Analyzes candidate moves around the snake's head for physical and topological safety."""

    @staticmethod
    def analyze(
        head: Point,
        food: Point,
        body: List[Point],
        grid_size: Tuple[int, int],
        cur_dir: str,
        recent_heads: Optional[Collection[Point]] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Optional[str], Optional[List[Point]]]:
        """Evaluates all 4 directions for physical barriers, traps, and paths.

        Returns:
            (analysis_dict, verified_food_dir, food_path)
        """
        w, h = grid_size
        body_set: Set[Point] = set(body)
        tail = body[-1] if body else head
        recent_set: Set[Point] = set(recent_heads) if recent_heads else set()
        forbidden_reverse = OPPOSITE.get(cur_dir) if len(body) >= 1 else None

        food_path, is_food_safe = simulate_path_safety(head, food, body, w, h)
        verified_food_dir: Optional[str] = None
        if food_path and is_food_safe and len(food_path) >= 2:
            dx, dy = food_path[1][0] - head[0], food_path[1][1] - head[1]
            for d, (mdx, mdy) in MOVES.items():
                if (dx, dy) == (mdx, mdy):
                    verified_food_dir = d
                    break

        analysis: Dict[str, Dict[str, Any]] = {}

        for d, (dx, dy) in MOVES.items():
            if d == forbidden_reverse:
                analysis[d] = {
                    "safe": False,
                    "is_trap": True,
                    "reason": "Physically Forbidden (Reverse)",
                }
                continue

            nx, ny = head[0] + dx, head[1] + dy
            is_wall = not (0 <= nx < w and 0 <= ny < h)
            is_body = (nx, ny) in body_set

            if is_wall or is_body:
                reason = "Wall" if is_wall else "Body Collision"
                analysis[d] = {
                    "safe": False,
                    "is_trap": True,
                    "reason": reason,
                }
                continue

            is_food = (nx == food[0] and ny == food[1])
            new_pos = (nx, ny)

            # Virtual next step obstacle simulation for tail reachability
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
            is_corner = (nx in [0, w - 1] and ny in [0, h - 1])

            open_neighbors = sum(
                1
                for ndx, ndy in MOVES.values()
                if 0 <= nx + ndx < w
                and 0 <= ny + ndy < h
                and (nx + ndx, ny + ndy) not in sim_obstacles
            )

            # 1. 1-wide dead-end trough check (boundary funnel)
            is_trough_trap = False
            if open_neighbors <= 2 and not is_food and len(body) >= 5:
                is_trough_trap = is_trough_dead_end(new_pos, head, sim_obstacles, w, h, len(body))

            # 2. Corner coffin trap: corner cell with its other exit blocked
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

            # 3. Pocket Trap: disconnected from tail and space cannot fit the snake
            is_pocket_trap = (not can_escape_to_tail) and (space <= len(body) + 2) and (len(body) >= 4)
            is_deadly_trap = is_trough_trap or is_corner_trap or is_pocket_trap
            is_revisit = (new_pos in recent_set)

            trap_reason: Optional[str] = None
            if is_deadly_trap:
                if is_corner_trap:
                    trap_reason = "Dead-end corner trap (exit blocked by body)"
                elif is_trough_trap:
                    trap_reason = "Narrow boundary trough funneling into wall/dead-end"
                else:
                    trap_reason = f"Dead-end pocket ({space} cells <= needed {len(body) + 2})"

            analysis[d] = {
                "safe": not is_deadly_trap,
                "is_trap": is_deadly_trap,
                "reason": trap_reason,
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
                "is_corner_trap": is_corner_trap,
                "is_pocket_trap": is_pocket_trap,
            }

        return analysis, verified_food_dir, food_path
