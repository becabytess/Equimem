"""AI Tutor orchestrator combining geometric mass-field tracking with Decision grading."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from .engine import MassFieldEngine
from .grader import DecisionGrader
from .models import Card, GradeResult


class AITutor:
    """Intelligent Socratic tutor powered by continuous geometric knowledge fields.

    - Uses MassFieldEngine to track the student's mastery surface over concept embeddings.
    - Uses DecisionGrader (gemini-decision-1) to evaluate free-form natural language responses.
    - Traces conceptual bridges between mastered anchors and unmastered gaps.
    """

    def __init__(
        self,
        engine: MassFieldEngine,
        grader: Optional[DecisionGrader] = None,
    ):
        self.engine = engine
        self.grader = grader or DecisionGrader()

    def select_card(self, strategy: str = "boundary", temperature: Optional[float] = None) -> Card:
        """Select the next optimal card to study.

        'boundary' picks concepts at the active learning frontier (between mastered and weak).
        'valley' picks the deepest unmastered knowledge gap.
        """
        cid = self.engine.select_next(strategy=strategy, temperature=temperature)
        return self.engine.cards[cid]

    def submit_answer(
        self,
        card_id: int,
        student_response: str,
    ) -> Tuple[GradeResult, Dict[str, float]]:
        """Evaluate a free-form student response and update the geometric knowledge surface."""
        card = self.engine.cards[card_id]
        grade_res = self.grader.grade(card, student_response)

        # Update the mass field with the continuous score y in [0.0, 1.0]
        review_update = self.engine.review(card_id, grade_res.score)

        return grade_res, review_update

    def find_learning_bridge(self, topic_query: str) -> Dict[str, Any]:
        """Diagnose a conceptual gap and find the nearest mastered anchor concept."""
        downhill = self.engine.trajectory_search(topic_query, k=3, direction="downhill")
        uphill = self.engine.trajectory_search(topic_query, k=3, direction="uphill")

        gap_card = downhill.hits[0] if downhill.hits else None
        anchor_card = uphill.hits[0] if uphill.hits else None

        return {
            "query": topic_query,
            "target_gap": gap_card,
            "mastered_anchor": anchor_card,
            "gap_smoothed_mastery": gap_card.smoothed_mass if gap_card else 0.0,
            "anchor_smoothed_mastery": anchor_card.smoothed_mass if anchor_card else 0.0,
        }
