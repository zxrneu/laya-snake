"""Decision Arbiter and candidate scoring policy for Snake AI."""

from typing import Any, Dict, Optional, Tuple


class DecisionArbiter:
    """Evaluates candidate scores, breaks infinite loops, and arbitrates safe actions."""

    @staticmethod
    def score_candidate(
        d: str,
        info: Dict[str, Any],
        steps_since_food: int = 0,
        probs: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculates utility score for a given safe candidate direction."""
        sc = 0.0
        has_safe_space = info.get("can_reach_tail", False) or (
            info.get("space", 0) >= info.get("snake_len", 0) * 1.5
        )

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

        # Anti-Looping Urgency: penalize revisiting recent cells if stalled without food
        if steps_since_food > 20 and info.get("is_revisit", False):
            sc -= 400.0

        # Mobility bonus: reward open maneuverability (more free neighbors = more escape routes)
        sc += info.get("open_neighbors", 0) * 50.0

        # Gentle avoidance of dead corners when not eating food
        if info.get("is_corner", False) and not info.get("is_food", False):
            sc -= 150.0

        sc += min(info.get("space", 0), 250) * 1.0

        if probs:
            sc += probs.get(d, 0.0) * 120.0

        return sc

    @classmethod
    def select_action(
        cls,
        raw_choice: str,
        analysis: Dict[str, Dict[str, Any]],
        probs: Optional[Dict[str, float]] = None,
        steps_since_food: int = 0,
    ) -> Tuple[str, bool]:
        """Arbitrates between model prediction and algorithmic safety.

        Returns:
            (final_action, was_overridden)
        """
        safe_candidates = [
            d
            for d, info in analysis.items()
            if info.get("safe", False) and not info.get("is_trap", False)
        ]

        # Fallback when no 100% clean safe candidate exists
        if not safe_candidates:
            walkable_candidates = [
                d
                for d, info in analysis.items()
                if info.get("reason") not in ("Wall", "Body Collision", "Physically Forbidden (Reverse)")
            ]
            if not walkable_candidates:
                valid = [
                    d
                    for d, info in analysis.items()
                    if info.get("reason") != "Physically Forbidden (Reverse)"
                ]
                fallback = valid[0] if valid else raw_choice
                return fallback, True

            best_fallback = max(
                walkable_candidates,
                key=lambda d: (
                    0 if (analysis[d].get("is_trough_trap") or analysis[d].get("is_corner_trap")) else 1,
                    analysis[d].get("space", 0),
                    analysis[d].get("open_neighbors", 0),
                    probs.get(d, 0.0) if probs else 0.0,
                ),
            )
            return best_fallback, True

        best_safe = max(
            safe_candidates,
            key=lambda d: cls.score_candidate(d, analysis[d], steps_since_food, probs),
        )

        # Check if raw model choice is safe and competitive
        choice_info = analysis.get(raw_choice, {})
        if choice_info.get("safe", False) and not choice_info.get("is_trap", False):
            choice_score = cls.score_candidate(raw_choice, choice_info, steps_since_food, probs)
            best_score = cls.score_candidate(best_safe, analysis[best_safe], steps_since_food, probs)
            if raw_choice == best_safe or (best_score - choice_score < 200.0):
                return raw_choice, False

        return best_safe, True

    @classmethod
    def select_best_safe_action(
        cls,
        analysis: Dict[str, Dict[str, Any]],
        steps_since_food: int = 0,
        probs: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[str], float]:
        """Pure algorithmic ceiling selection (no neural agent required)."""
        safe_candidates = [
            d
            for d, info in analysis.items()
            if info.get("safe", False) and not info.get("is_trap", False)
        ]

        if not safe_candidates:
            walkable_candidates = [
                d
                for d, info in analysis.items()
                if info.get("reason") not in ("Wall", "Body Collision", "Physically Forbidden (Reverse)")
            ]
            if not walkable_candidates:
                valid = [
                    d
                    for d, info in analysis.items()
                    if info.get("reason") != "Physically Forbidden (Reverse)"
                ]
                return (valid[0] if valid else None), 0.0

            best_fallback = max(
                walkable_candidates,
                key=lambda d: (
                    0 if (analysis[d].get("is_trough_trap") or analysis[d].get("is_corner_trap")) else 1,
                    analysis[d].get("space", 0),
                    analysis[d].get("open_neighbors", 0),
                    probs.get(d, 0.0) if probs else 0.0,
                ),
            )
            return best_fallback, 0.0

        best_d = max(
            safe_candidates,
            key=lambda d: cls.score_candidate(d, analysis[d], steps_since_food, probs),
        )
        return best_d, cls.score_candidate(best_d, analysis[best_d], steps_since_food, probs)
