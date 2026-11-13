"""Decision-model grader for free-form student responses using Gemini Decision API."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .models import Card, GradeResult


class DecisionGrader:
    """Non-generative semantic grader using gemini-decision-1.

    Evaluates free-form student answers against ground-truth question-answer pairs
    in a single prefill pass with zero output tokens, producing continuous calibrated
    scores in [0.0, 1.0] for the mass-field engine.
    """

    CRITERIA = [
        "Level 0: Completely incorrect, unrelated, missing essential core facts, or contradicts reference answer.",
        "Level 1: Partially correct; captures basic intuition or one element, but omits critical components.",
        "Level 2: Substantially correct; accurately conveys the core mechanism or factual answer.",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-decision-1",
        timeout: float = 15.0,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            for candidate in [".env", "../.env", "../../.env"]:
                if os.path.exists(candidate):
                    with open(candidate, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                if k.strip() in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                                    self.api_key = v.strip()
                                    os.environ[k.strip()] = self.api_key
                                    break
                if self.api_key:
                    break

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment or .env file")

        from google import genai
        self.client = genai.Client(http_options={"api_version": "v1beta"}, api_key=self.api_key)
        self.model = model
        self.timeout = float(timeout)
        self.calls: int = 0
        self.fails: int = 0

    def grade(
        self,
        card: Card,
        student_answer: str,
    ) -> GradeResult:
        """Grade a single free-form student answer against a card's ground-truth Q&A."""
        results = self.grade_batch([(card, student_answer)])
        return results[0]

    def grade_batch(
        self,
        evaluations: Sequence[Tuple[Card, str]],
    ) -> List[GradeResult]:
        """Grade multiple student responses in parallel in a single API call."""
        if not evaluations:
            return []

        # Construct shared context prefix
        sections = []
        questions = {}
        for idx, (card, s_ans) in enumerate(evaluations):
            qid = f"eval_{idx}"
            sections.append(
                f"[Item {idx}]\n"
                f"Target Question: {card.question}\n"
                f"Reference Answer: {card.answer}\n"
                f"Student Response: {s_ans.strip()}"
            )
            questions[qid] = {
                "type": "score",
                "instructions": (
                    f"Grade Item {idx}: Determine whether the student response accurately and "
                    f"substantially matches the reference answer for the given target question."
                ),
                "criteria": self.CRITERIA,
            }

        shared_input = "EVALUATION CONTEXT:\n\n" + "\n\n".join(sections)

        try:
            self.calls += 1
            response = self.client.decisions.create(
                model=self.model,
                input=shared_input,
                questions=questions,
                timeout=self.timeout,
            )
        except Exception as e:
            self.fails += 1
            # Fallback heuristic: 0.0 with zero confidence
            return [
                GradeResult(
                    score=0.0,
                    raw_score=0.0,
                    confidence=0.0,
                    level_probabilities={"0": 1.0, "1": 0.0, "2": 0.0},
                    feedback=f"API grading failed ({type(e).__name__}: {e}); defaulted to 0.0.",
                )
                for _ in evaluations
            ]

        results = []
        answers = getattr(response, "answers", {}) or {}

        for idx, (card, s_ans) in enumerate(evaluations):
            qid = f"eval_{idx}"
            ans = answers.get(qid)
            if ans is None:
                results.append(
                    GradeResult(
                        score=0.0,
                        raw_score=0.0,
                        confidence=0.0,
                        level_probabilities={"0": 1.0, "1": 0.0, "2": 0.0},
                        feedback="No answer returned for item.",
                    )
                )
                continue

            raw_score = float(getattr(ans, "score", 0.0))
            conf = float(getattr(ans, "confidence", 0.0))
            probs = getattr(ans, "probabilities", {}) or {}

            # Normalized score in [0.0, 1.0]
            norm_score = float(max(0.0, min(1.0, raw_score / 2.0)))

            p0 = float(probs.get("0", 0.0))
            p1 = float(probs.get("1", 0.0))
            p2 = float(probs.get("2", 0.0))

            if norm_score >= 0.75 or p2 >= 0.60:
                summary = "Mastered: Student answer accurately captures the core factual knowledge."
            elif norm_score >= 0.35 or p1 >= 0.50:
                summary = "Partially Correct: Student captured partial elements but missed key concepts."
            else:
                summary = "Incorrect: Student answer fails to convey the required target knowledge."

            results.append(
                GradeResult(
                    score=norm_score,
                    raw_score=raw_score,
                    confidence=conf,
                    level_probabilities={"0": p0, "1": p1, "2": p2},
                    feedback=summary,
                )
            )

        return results
