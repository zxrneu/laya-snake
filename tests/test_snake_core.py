"""Unit tests for the modular snake_core package."""

import unittest

from snake_core import (
    DecisionArbiter,
    MOVES,
    OPPOSITE,
    PromptBuilder,
    SafetyAnalyzer,
    SnakeGame,
    bfs_path,
    bfs_shortest_path,
    get_flood_fill_space,
    is_trough_dead_end,
    simulate_path_safety,
)


class TestSnakeGeometry(unittest.TestCase):
    def test_bfs_shortest_path(self):
        w, h = 10, 10
        obstacles = {(1, 0), (1, 1), (1, 2)}
        # Path from (0, 0) to (2, 0) around the obstacle
        dist = bfs_shortest_path((0, 0), (2, 0), obstacles, w, h)
        self.assertIsNotNone(dist)
        self.assertEqual(dist, 8)  # down 3, right 2, up 3

    def test_flood_fill_space(self):
        w, h = 5, 5
        # Box off a 2x2 corner at (0, 0)
        obstacles = {(2, 0), (2, 1), (0, 2), (1, 2), (2, 2)}
        space = get_flood_fill_space((0, 0), obstacles, w, h)
        self.assertEqual(space, 4)  # (0,0), (1,0), (0,1), (1,1)

    def test_trough_dead_end(self):
        w, h = 10, 10
        # Wall on top (y=0). Body on row 1 from x=0 to x=6.
        # Trench is row 0.
        head = (2, 0)
        sim_obstacles = {(x, 1) for x in range(7)} | {(6, 0)}  # end is capped at (6, 0)
        is_trap = is_trough_dead_end((3, 0), head, sim_obstacles, w, h, snake_len=20)
        self.assertTrue(is_trap)


class TestSafetyAnalyzer(unittest.TestCase):
    def test_corner_trap_detection(self):
        w, h = 10, 10
        # Corner is (0, 0). Snake enters from (1, 0).
        # Other exit of (0, 0) is (0, 1). If (0, 1) is blocked, it must be a corner trap.
        head = (1, 0)
        food = (5, 5)
        body = [(1, 1), (0, 1), (0, 2), (0, 3)]  # (0, 1) is blocked!
        analysis, _, _ = SafetyAnalyzer.analyze(head, food, body, (w, h), cur_dir="LEFT")

        left_info = analysis["LEFT"]  # moving into (0, 0)
        self.assertTrue(left_info["is_corner"])
        self.assertTrue(left_info["is_corner_trap"])
        self.assertTrue(left_info["is_trap"])
        self.assertFalse(left_info["safe"])

    def test_forbidden_reverse(self):
        w, h = 10, 10
        head = (5, 5)
        food = (5, 6)
        body = [(5, 4), (5, 3)]
        analysis, _, _ = SafetyAnalyzer.analyze(head, food, body, (w, h), cur_dir="DOWN")
        self.assertFalse(analysis["UP"]["safe"])
        self.assertEqual(analysis["UP"]["reason"], "Physically Forbidden (Reverse)")


class TestDecisionArbiter(unittest.TestCase):
    def test_safe_selection_overrides_suicide(self):
        analysis = {
            "UP": {"safe": False, "is_trap": True, "reason": "Wall"},
            "DOWN": {"safe": True, "is_trap": False, "space": 50, "can_reach_tail": True, "open_neighbors": 3},
            "LEFT": {"safe": False, "is_trap": True, "reason": "Body Collision"},
            "RIGHT": {"safe": False, "is_trap": True, "reason": "Physically Forbidden (Reverse)"},
        }
        # If raw model stupidly outputs UP (Wall), arbiter must override to DOWN
        action, overridden = DecisionArbiter.select_action("UP", analysis)
        self.assertEqual(action, "DOWN")
        self.assertTrue(overridden)

    def test_fallback_never_picks_wall(self):
        analysis = {
            "UP": {"safe": False, "is_trap": True, "reason": "Wall", "space": 0},
            "DOWN": {"safe": False, "is_trap": True, "reason": "Dead-end pocket", "space": 10, "open_neighbors": 1},
            "LEFT": {"safe": False, "is_trap": True, "reason": "Body Collision", "space": 0},
            "RIGHT": {"safe": False, "is_trap": True, "reason": "Physically Forbidden (Reverse)"},
        }
        action, overridden = DecisionArbiter.select_action("UP", analysis)
        self.assertEqual(action, "DOWN")  # Pick the walkable pocket, NEVER the wall!
        self.assertTrue(overridden)


class TestSnakeGame(unittest.TestCase):
    def test_game_loop_and_food_eating(self):
        game = SnakeGame(10, 10, seed=42)
        initial_len = len(game.snake)
        # Force food in front of head
        head = game.snake[0]
        game.food = (head[0] + 1, head[1])
        res = game.step("RIGHT")
        self.assertTrue(res["alive"])
        self.assertTrue(res["ate"])
        self.assertEqual(len(game.snake), initial_len + 1)
        self.assertEqual(game.score, 10)


if __name__ == "__main__":
    unittest.main()
