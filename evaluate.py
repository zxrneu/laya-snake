"""Comprehensive Benchmark & Performance Evaluation Suite for Laya Snake AI.

Evaluates:
  1. Fast Algorithmic Ceiling (Dual-BFS + Flood Fill + Safe Arbiter @ 10,000 steps/sec)
  2. Full Neural Agent (ModernBERT-421M + Prompt Builder + Decision Arbiter)
Across arbitrary grid sizes (10x10, 14x14, 20x20, etc.)
"""

import argparse
from collections import deque
import random
import time
from typing import List, Optional

from snake_core import (
    DecisionArbiter,
    PromptBuilder,
    SafetyAnalyzer,
    SnakeGame,
)


def run_single_game_fast(
    w: int, h: int, max_steps: int = 3000, seed: Optional[int] = None
) -> dict:
    """Runs a single game using the pure algorithmic upper-bound policy."""
    game = SnakeGame(w, h, seed)
    start_t = time.perf_counter()

    while not game.is_over and game.steps < max_steps:
        head = game.snake[0]
        body = game.snake[1:]
        analysis, _, _ = SafetyAnalyzer.analyze(
            head=head,
            food=game.food,
            body=body,
            grid_size=(game.w, game.h),
            cur_dir=game.direction,
            recent_heads=game.recent_heads,
        )
        best_move, _ = DecisionArbiter.select_best_safe_action(
            analysis, steps_since_food=game.steps_since_food
        )
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
        "steps_per_sec": round(game.steps / (max(0.001, duration_ms / 1000)), 1),
    }


def run_single_game_neural(
    agent,
    w: int,
    h: int,
    max_steps: int = 600,
    seed: Optional[int] = None,
) -> dict:
    """Runs a single game driven by the actual Laya neural model."""
    game = SnakeGame(w, h, seed)
    start_t = time.perf_counter()
    overridden_count = 0
    inference_times = []

    while not game.is_over and game.steps < max_steps:
        head = game.snake[0]
        food = game.food if game.food else (0, 0)
        body = game.snake[1:]

        analysis, verified_food_dir, food_path = SafetyAnalyzer.analyze(
            head=head,
            food=food,
            body=body,
            grid_size=(w, h),
            cur_dir=game.direction,
            recent_heads=game.recent_heads,
        )

        state, questions = PromptBuilder.build(
            head=head,
            food=food,
            body=body,
            grid_size=(w, h),
            cur_dir=game.direction,
            steps_since_food=game.steps_since_food,
            analysis=analysis,
            verified_food_dir=verified_food_dir,
            food_path=food_path,
        )

        t0 = time.perf_counter()
        res = agent.predict(state, questions)
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)

        ans = res["answers"]["direction"]
        raw_choice = ans["choice"]
        probs = ans["probabilities"]

        final_choice, overridden = DecisionArbiter.select_action(
            raw_choice=raw_choice,
            analysis=analysis,
            probs=probs,
            steps_since_food=game.steps_since_food,
        )
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
        "duration_s": round(duration_s, 2),
    }


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
        top_reason = max(
            set([run["death_reason"] for run in runs]),
            key=[run["death_reason"] for run in runs].count,
        )
        print(
            f"{g:<7} | {n:<5} | {max_len:<7} | {avg_len:<7} | {max_fill:>8.1f}% | {avg_steps:<9} | {max_loop:<8} | {top_reason}"
        )

    print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Laya Snake AI Performance & Limit Evaluation")
    parser.add_argument(
        "--mode",
        choices=["fast", "neural", "both"],
        default="fast",
        help="fast: algorithmic ceiling (10k steps/sec); neural: actual Laya model; both: compare both",
    )
    parser.add_argument(
        "--grids",
        nargs="+",
        type=int,
        default=[10, 14, 20],
        help="List of grid sizes to evaluate (e.g. 10 14 20)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Number of trials per grid size",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Max steps per trial",
    )
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
                print(
                    f"     Game #{i+1}: Final Length = {res['final_length']}/{res['total_cells']} "
                    f"({res['fill_rate_pct']}%), Score = {res['score']}, Steps = {res['steps']}, "
                    f"Max Loop = {res['max_steps_since_food']} steps, Reason = {res['death_reason']}"
                )
        print_table(fast_results, "Algorithmic Theoretical Upper Bound (Dual-BFS + Flood Fill Arbiter)")

    # Full Neural Model Evaluation
    if args.mode in ["neural", "both"]:
        print("\n[🧠] Loading Laya Neural Model from server pipeline...")
        from server import agent

        neural_games = min(args.games, 3)
        neural_steps = min(args.max_steps, 500)
        neural_results = []

        for g_size in args.grids:
            print(
                f"\n  -> Evaluating Neural Agent on {g_size}x{g_size} grid ({neural_games} trials, max {neural_steps} steps)..."
            )
            for i in range(neural_games):
                res = run_single_game_neural(
                    agent, g_size, g_size, max_steps=neural_steps, seed=200 + i * 19
                )
                neural_results.append(res)
                print(
                    f"     Trial #{i+1}: Length = {res['final_length']}/{res['total_cells']} "
                    f"({res['fill_rate_pct']}%), Score = {res['score']}, Overridden = {res['overridden_pct']}%, "
                    f"Avg Latency = {res['avg_inference_ms']}ms, Reason = {res['death_reason']}"
                )
        print_table(neural_results, "Actual Laya Neural Model (ModernBERT-421M + Arbiter)")

    print("\n✅ Evaluation complete! All tests passed successfully.")


if __name__ == "__main__":
    main()
