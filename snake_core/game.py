"""Snake game simulation environment for evaluation and gameplay."""

from collections import deque
import random
from typing import Deque, List, Optional, Set, Tuple

from snake_core.constants import MOVES, Point


class SnakeGame:
    """Standard Snake game environment supporting seeding, steps, and collision detection."""

    def __init__(self, w: int, h: int, seed: Optional[int] = None):
        self.w = w
        self.h = h
        if seed is not None:
            random.seed(seed)
        mid_x, mid_y = w // 2, h // 2
        self.snake: List[Point] = [(mid_x, mid_y), (mid_x - 1, mid_y), (mid_x - 2, mid_y)]
        self.direction = "RIGHT"
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.max_steps_since_food = 0
        self.recent_heads: Deque[Point] = deque(maxlen=24)
        self.is_over = False
        self.death_reason = ""
        self.food: Optional[Point] = None
        self.spawn_food()

    def spawn_food(self) -> None:
        """Spawns food in a random unoccupied cell."""
        occupied: Set[Point] = set(self.snake)
        empty = [(x, y) for x in range(self.w) for y in range(self.h) if (x, y) not in occupied]
        if not empty:
            self.food = None
            self.is_over = True
            self.death_reason = "VICTORY (Board 100% Filled)"
        else:
            self.food = random.choice(empty)

    def step(self, move_dir: str) -> dict:
        """Advances the game by one tick in the specified direction.

        Returns:
            dict containing alive, ate, score, death_reason.
        """
        if self.is_over:
            return {"alive": False, "ate": False, "score": self.score, "reason": self.death_reason}

        self.steps += 1
        self.direction = move_dir
        dx, dy = MOVES[move_dir]
        head = self.snake[0]
        new_head: Point = (head[0] + dx, head[1] + dy)

        # 1. Wall collision check
        if not (0 <= new_head[0] < self.w and 0 <= new_head[1] < self.h):
            self.is_over = True
            self.death_reason = "Wall Collision"
            return {"alive": False, "ate": False, "score": self.score, "reason": self.death_reason}

        is_eating = (new_head == self.food)
        body_to_check = self.snake if is_eating else self.snake[:-1]

        # 2. Self body crash check
        if new_head in set(body_to_check):
            self.is_over = True
            self.death_reason = "Self Body Crash"
            return {"alive": False, "ate": False, "score": self.score, "reason": self.death_reason}

        # 3. Advance snake
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

        return {"alive": True, "ate": is_eating, "score": self.score, "reason": ""}
