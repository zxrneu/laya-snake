"""Natural language and structured prompt synthesis for Laya Snake AI."""

from typing import Any, Dict, List, Optional, Tuple

from snake_core.constants import Point


class PromptBuilder:
    """Builds state representations and question criteria for the Laya neural model."""

    @staticmethod
    def build(
        head: Point,
        food: Point,
        body: List[Point],
        grid_size: Tuple[int, int],
        cur_dir: str,
        steps_since_food: int,
        analysis: Dict[str, Dict[str, Any]],
        verified_food_dir: Optional[str] = None,
        food_path: Optional[List[Point]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Constructs (state, questions) for Laya inference."""
        w, h = grid_size
        criteria: Dict[str, str] = {}
        safe_moves: List[str] = []
        danger_moves: List[str] = []

        for d, info in analysis.items():
            if not info.get("safe", False) or info.get("is_trap", False):
                reason = info.get("reason", "Fatal move")
                danger_moves.append(d)
                criteria[d] = f"DANGER: {reason}. Snake dies or becomes fatally trapped."
            else:
                safe_moves.append(d)
                is_food = info.get("is_food", False)
                space = info.get("space", 0)
                food_path_len = info.get("food_path_len")
                can_escape_to_tail = info.get("can_reach_tail", False)
                has_safe_space = can_escape_to_tail or (space >= info.get("snake_len", 0) * 1.5)

                if is_food:
                    if has_safe_space:
                        criteria[d] = f"CRITICAL BEST: Direct hit! Eats food with guaranteed safe space ({space} cells)."
                    else:
                        criteria[d] = f"TRAP WARNING: Eats food but risks being trapped in coils ({space} cells)."
                elif d == verified_food_dir:
                    steps_needed = len(food_path) - 1 if food_path else 1
                    criteria[d] = (
                        f"CRITICAL BEST: 100% verified safe shortest path to FOOD ({steps_needed} steps). "
                        "Guaranteed exit to tail after eating. MUST take to eat fruit and break loops!"
                    )
                elif food_path_len is not None and has_safe_space:
                    criteria[d] = f"BEST: Safe approach to food (path: {food_path_len} steps, {space} free cells). Breaks out of tail loop."
                elif can_escape_to_tail or space >= info.get("snake_len", 0) * 2:
                    if verified_food_dir is not None:
                        criteria[d] = f"AVOID LOOP: Circles tail away from food ({space} free cells). Food is safely accessible via {verified_food_dir}; do not loop!"
                    else:
                        criteria[d] = f"DEFENSIVE: Safe tail-following survival ({space} free cells) until food path clears."
                else:
                    criteria[d] = f"CAUTION: Narrow territory ({space} cells) with no direct tail route."

        # If all moves were classified as deadly traps, rescue the walkable one with max space
        walkable_danger = [
            d
            for d in danger_moves
            if analysis[d].get("reason") not in ("Wall", "Body Collision", "Physically Forbidden (Reverse)")
        ]
        if not safe_moves and walkable_danger:
            best_trap_move = max(
                walkable_danger,
                key=lambda d: (analysis[d].get("space", 0), analysis[d].get("open_neighbors", 0)),
            )
            criteria[best_trap_move] = (
                f"EMERGENCY: Maximum reachable free space ({analysis[best_trap_move].get('space', 0)} cells). "
                "Best survival chance."
            )
            safe_moves.append(best_trap_move)

        situation: List[str] = []
        if safe_moves:
            if verified_food_dir:
                situation.append(
                    f"CRITICAL: Direct safe path to fruit is OPEN in direction: {verified_food_dir}! Advance towards food."
                )
            else:
                best_opts = [d for d in safe_moves if "BEST" in criteria.get(d, "")]
                if best_opts:
                    situation.append(f"Recommended safe direction(s) towards food: {', '.join(best_opts)}.")
                else:
                    situation.append(f"Safe defensive moves: {', '.join(safe_moves)}.")
        if danger_moves:
            situation.append(f"Fatal directions to avoid: {', '.join(danger_moves)}.")

        state = {
            "snake_game": {
                "grid_size": f"{w}x{h}",
                "snake_head": list(head),
                "food_location": list(food) if food else [0, 0],
                "snake_length": len(body),
                "current_direction": cur_dir,
                "steps_since_food": steps_since_food,
                "safe_directions": safe_moves,
                "danger_directions": danger_moves,
                "situation": " ".join(situation),
            }
        }

        questions = {
            "direction": {
                "type": "choice",
                "instructions": (
                    "Which direction should the snake move to safely eat food and advance without "
                    "getting trapped or looping in its own body?"
                ),
                "criteria": criteria,
            }
        }

        return state, questions
