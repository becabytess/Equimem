"""Structured concept card formatting for dense hyperspherical embeddings."""
from __future__ import annotations

from typing import Optional


def format_card(question: str, answer: str, topic: Optional[str] = None) -> str:
    """Format a concept flashcard into a clean, structured document for dense embedding.

    Combines the inquiry intent (question), the factual resolution (answer),
    and domain context (topic) so that transformer self-attention captures the
    full conceptual meaning rather than blind concatenation.
    """
    q_clean = " ".join(question.strip().split())
    a_clean = " ".join(answer.strip().split())
    if topic and topic.strip():
        t_clean = " ".join(topic.strip().split())
        return f"[CONCEPT CARD | Domain: {t_clean}]\nTarget Inquiry: {q_clean}\nCore Knowledge: {a_clean}"
    return f"[CONCEPT CARD]\nTarget Inquiry: {q_clean}\nCore Knowledge: {a_clean}"
