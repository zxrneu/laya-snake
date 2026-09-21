"""Spatial geometry and pathfinding algorithms for Snake AI."""

from collections import deque
from typing import List, Optional, Set, Tuple

from snake_core.constants import MOVES, Point


def bfs_shortest_path(
    start: Point, target: Point, obstacles: Set[Point], w: int, h: int
) -> Optional[int]:
    """Computes the shortest path length from start to target avoiding obstacles.
    Returns None if no path exists.
    """
    if start == target:
        return 0
    if start in obstacles or not (0 <= start[0] < w and 0 <= start[1] < h):
        return None

    visited = {start}
    q = deque([(start[0], start[1], 0)])

    while q:
        cx, cy, dist = q.popleft()
        for dx, dy in MOVES.values():
            nx, ny = cx + dx, cy + dy
            if (nx, ny) == target:
                return dist + 1
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in obstacles and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny, dist + 1))
    return None


def bfs_path(
    start: Point, target: Point, obstacles: Set[Point], w: int, h: int
) -> Optional[List[Point]]:
    """Finds the actual shortest coordinate path from start to target."""
    if start == target:
        return [start]
    if start in obstacles or not (0 <= start[0] < w and 0 <= start[1] < h):
        return None

    visited = {start}
    q = deque([[start]])

    while q:
        path = q.popleft()
        cx, cy = path[-1]
        for dx, dy in MOVES.values():
            nx, ny = cx + dx, cy + dy
            nxt = (nx, ny)
            if nxt == target:
                return path + [nxt]
            if 0 <= nx < w and 0 <= ny < h and nxt not in obstacles and nxt not in visited:
                visited.add(nxt)
                q.append(path + [nxt])
    return None


def get_flood_fill_space(
    start_cell: Point, obstacles: Set[Point], w: int, h: int, max_depth: int = 400
) -> int:
    """Calculates accessible empty territory from start_cell using BFS flood fill."""
    if start_cell in obstacles or not (0 <= start_cell[0] < w and 0 <= start_cell[1] < h):
        return 0
    visited = {start_cell}
    q = deque([start_cell])
    count = 0
    while q and count < max_depth:
        cx, cy = q.popleft()
        count += 1
        for dx, dy in MOVES.values():
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                nxt = (nx, ny)
                if nxt not in obstacles and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
    return count


def simulate_path_safety(
    head: Point, food: Point, body: List[Point], w: int, h: int
) -> Tuple[Optional[List[Point]], bool]:
    """Simulates eating food along the shortest path and checks if virtual tail is reachable.
    Returns (path_to_food, is_safe_with_tail_exit).
    """
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


def is_trough_dead_end(
    new_pos: Point,
    head: Point,
    sim_obstacles: Set[Point],
    w: int,
    h: int,
    snake_len: int,
    max_trace_depth: int = 30,
) -> bool:
    """Traces a 1-wide trench along a wall or between bodies to check if it dead-ends
    before the snake can fit or escape.
    """
    curr = new_pos
    prev = head
    depth = 1
    cutoff = min(snake_len, max_trace_depth)

    while depth < cutoff:
        exits = [
            (curr[0] + dx, curr[1] + dy)
            for dx, dy in MOVES.values()
            if 0 <= curr[0] + dx < w
            and 0 <= curr[1] + dy < h
            and (curr[0] + dx, curr[1] + dy) not in sim_obstacles
            and (curr[0] + dx, curr[1] + dy) != prev
        ]
        if len(exits) == 0:
            return True
        if len(exits) == 1:
            prev = curr
            curr = exits[0]
            depth += 1
        else:
            return False
    return False
