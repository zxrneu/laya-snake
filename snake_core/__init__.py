"""Snake Core package: Clean, modular, and SOLID Snake AI engine."""

from snake_core.analyzer import SafetyAnalyzer
from snake_core.arbiter import DecisionArbiter
from snake_core.constants import MOVES, OPPOSITE, Point
from snake_core.game import SnakeGame
from snake_core.geometry import (
    bfs_path,
    bfs_shortest_path,
    get_flood_fill_space,
    is_trough_dead_end,
    simulate_path_safety,
)
from snake_core.prompt_builder import PromptBuilder

__all__ = [
    "Point",
    "MOVES",
    "OPPOSITE",
    "bfs_path",
    "bfs_shortest_path",
    "get_flood_fill_space",
    "simulate_path_safety",
    "is_trough_dead_end",
    "SafetyAnalyzer",
    "DecisionArbiter",
    "PromptBuilder",
    "SnakeGame",
]
