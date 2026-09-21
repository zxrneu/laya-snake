"""Constants for Snake AI Core."""

from typing import Dict, Tuple

Point = Tuple[int, int]

MOVES: Dict[str, Point] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

OPPOSITE: Dict[str, str] = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}
