"""Model Context Protocol (MCP) Server for Equimem.

Enables any MCP-compatible AI assistant (Claude Desktop, Cursor, Antigravity, etc.)
to natively interact with the continuous conserved-memory knowledge manifold.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Ensure local package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.server.mcpserver import MCPServer
from equimem.models import FieldConfig
from equimem.engine import MassFieldEngine
from equimem.embedder import GeminiEmbedder, NGramEmbedder
from equimem.grader import DecisionGrader
from equimem.mutator import CardMutator
from lab.deck_50 import CARDS_DATA

# Load API credentials
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

server = MCPServer("equimem")

# Global engine instance
_engine: Optional[MassFieldEngine] = None
_grader: Optional[DecisionGrader] = None
_mutator: Optional[CardMutator] = None
_cooldown_history: List[int] = []


def _get_mutator() -> Optional[CardMutator]:
    global _mutator
    if _mutator is None:
        try:
            _mutator = CardMutator()
        except Exception:
            _mutator = None
    return _mutator


def _get_engine() -> tuple[MassFieldEngine, Optional[DecisionGrader]]:
    global _engine, _grader
    if _engine is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                cache_file = os.path.join(os.path.dirname(__file__), "..", "lab", "cache_embeddings_60.npz")
                if os.path.exists(cache_file):
                    from lab.simulate_50_cards import CachedGeminiEmbedder
                    embedder = CachedGeminiEmbedder(cache_file, model="gemini-embedding-001", dim=3072)
                else:
                    embedder = GeminiEmbedder(model="gemini-embedding-001", dim=3072)
            except Exception as e:
                embedder = NGramEmbedder(dim=512)
            _grader = DecisionGrader(model="gemini-decision-1")
        else:
            embedder = NGramEmbedder(dim=512)
            _grader = None

        cfg = FieldConfig(tau=0.15, alpha_gain=0.70, alpha_loss=0.85, select_temp=0.10)
        _engine = MassFieldEngine(embedder=embedder, config=cfg)

        card_tuples = [
            (c["question"], c["answer"], c.get("topic", "General"), {"card_id": c["id"]})
            for c in CARDS_DATA
        ]
        _engine.add_cards(card_tuples)

    return _engine, _grader


@server.tool()
def select_next_card(
    strategy: str = "boundary",
    temperature: float = 0.10,
    cooldown: int = 8,
) -> Dict[str, Any]:
    """Select the next flashcard to test the user's knowledge.

    Args:
        strategy: Selection criterion: 'boundary' (active learning frontier) or 'valley' (deepest gaps).
        temperature: Softmax temperature for exploration (0.0 = deterministic argmax).
        cooldown: Number of recently reviewed cards to temporarily exclude.

    Returns:
        The selected card's ID, question, topic, review count, and current mastery mass.
    """
    engine, _ = _get_engine()
    exclude_set = set()
    if cooldown > 0 and _cooldown_history:
        exclude_set.update(_cooldown_history[-cooldown:])

    card_id = engine.select_next(strategy=strategy, temperature=temperature, exclude=exclude_set)
    _cooldown_history.append(card_id)

    card = engine.cards[card_id]
    snap = engine.surface_state()

    return {
        "card_id": card.id,
        "question": card.question,
        "topic": card.topic,
        "reviews": card.reviews,
        "mass": round(float(snap.m[card_id]), 4),
        "smoothed_mass": round(float(snap.m_bar[card_id]), 4),
    }


@server.tool()
def review_card(card_id: int, student_answer: str) -> Dict[str, Any]:
    """Evaluate a student's answer to a flashcard and update the conserved mass field.

    Uses Gemini Decision 1 (zero-output tokens) to grade the recall quality, updates
    the concept's mass, and redistributes mass across the deck.

    Args:
        card_id: The integer ID of the card being reviewed.
        student_answer: The free-form natural language response from the user.

    Returns:
        Evaluation score, level probabilities, mass update details, and reference answer.
    """
    engine, grader = _get_engine()
    if not (0 <= card_id < engine.n):
        return {"error": f"Card ID {card_id} out of bounds (deck size: {engine.n})"}

    card = engine.cards[card_id]
    m_before = float(engine.m[card_id])

    if grader:
        grade_res = grader.grade(card=card, student_answer=student_answer)
        score = grade_res.score
        probs = grade_res.level_probabilities
        feedback = grade_res.feedback
    else:
        score = 0.5
        probs = {"0": 0.25, "1": 0.50, "2": 0.25}
        feedback = "Evaluated via fallback."

    engine.review(card_id, score=score)

    snap = engine.surface_state()
    m_after = float(snap.m[card_id])
    mb_after = float(snap.m_bar[card_id])

    return {
        "card_id": card_id,
        "score": round(score, 4),
        "feedback": feedback,
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "mass_before": round(m_before, 4),
        "mass_after": round(m_after, 4),
        "delta_mass": round(m_after - m_before, 4),
        "smoothed_mass": round(mb_after, 4),
        "reference_answer": card.answer,
        "total_deck_mass": round(float(snap.m.sum()), 4),
    }


@server.tool()
def add_card(question: str, answer: str, topic: str = "General") -> Dict[str, Any]:
    """Add a new concept flashcard to the continuous conserved-memory deck.

    Args:
        question: The target inquiry prompt.
        answer: The reference factual explanation.
        topic: Conceptual domain (e.g. 'Quantum Mechanics', 'Linear Algebra').

    Returns:
        The newly assigned card ID and new total deck size.
    """
    engine, _ = _get_engine()
    new_ids = engine.add_cards([(question, answer, topic, {})])
    return {
        "card_id": new_ids[0],
        "message": "Card successfully added to the conserved field.",
        "deck_size": engine.n,
        "total_mass": round(float(engine.m.sum()), 4),
    }


@server.tool()
def trajectory_search(
    query: str,
    direction: str = "downhill",
    k: int = 5,
) -> Dict[str, Any]:
    """Search for concepts by following spherical mastery gradients from a query.

    Args:
        query: Conceptual topic inquiry (e.g. 'eigenvalues and decomposition').
        direction: 'downhill' to discover unmastered knowledge gaps, or 'uphill' to find strong mastery anchors.
        k: Maximum number of trajectory hits to return.

    Returns:
        Ranked list of concept cards along the spherical trajectory.
    """
    engine, _ = _get_engine()
    res = engine.trajectory_search(query=query, k=k, direction=direction)
    return {
        "query": res.query,
        "direction": res.direction,
        "hits": [
            {
                "card_id": h.card_id,
                "question": h.question,
                "score": round(h.score, 4),
                "mass": round(h.mass, 4),
                "smoothed_mass": round(h.smoothed_mass, 4),
            }
            for h in res.hits
        ],
    }


@server.tool()
def get_knowledge_state() -> Dict[str, Any]:
    """Retrieve full diagnostic metrics of the current knowledge state.

    Returns:
        Deck size, mass invariant, global surface variance, entropy, and per-topic mastery metrics.
    """
    engine, _ = _get_engine()
    snap = engine.surface_state()

    topics = sorted(list({c.topic for c in engine.cards}))
    topic_breakdown = []
    for t in topics:
        idxs = [i for i, c in enumerate(engine.cards) if c.topic == t]
        topic_breakdown.append({
            "topic": t,
            "count": len(idxs),
            "avg_mass": round(float(snap.m[idxs].mean()), 4),
            "avg_smoothed_mass": round(float(snap.m_bar[idxs].mean()), 4),
            "min_smoothed_mass": round(float(snap.m_bar[idxs].min()), 4),
            "max_smoothed_mass": round(float(snap.m_bar[idxs].max()), 4),
        })

    return {
        "deck_size": engine.n,
        "total_mass": round(float(snap.m.sum()), 4),
        "surface_variance": round(snap.variance, 6),
        "entropy": round(snap.entropy, 4),
        "mean_gradient_norm": round(snap.mean_grad_norm, 4),
        "topics": topic_breakdown,
    }


@server.tool()
def diagnose_card(card_id: int) -> Dict[str, Any]:
    """Diagnose a card's geometric state (isolated spike, consolidated plateau, pothole, unmastered).

    Args:
        card_id: Integer card ID to diagnose.

    Returns:
        Diagnostic classification, recommended action, card mass, and neighborhood mastery.
    """
    engine, _ = _get_engine()
    if not (0 <= card_id < engine.n):
        return {"error": f"Card ID {card_id} out of bounds"}
    return engine.diagnose_card(card_id)


@server.tool()
def evolve_card(card_id: int) -> Dict[str, Any]:
    """Manually trigger the evolution of a card based on its neighborhood geometry.

    Upgrades isolated spikes in-place into deeper probes, reframes potholes, or expands plateaus.

    Args:
        card_id: Integer card ID to evolve.

    Returns:
        Result of the evolution operation (new question/answer, action taken, mass relaxation).
    """
    engine, _ = _get_engine()
    if not (0 <= card_id < engine.n):
        return {"error": f"Card ID {card_id} out of bounds"}
    mutator = _get_mutator()
    if not mutator:
        return {"error": "CardMutator is not configured (missing API key)"}
    return mutator.evolve_card(engine, card_id)


@server.tool()
def expand_cluster(card_id: int, count: int = 2) -> Dict[str, Any]:
    """Manually trigger outward frontier expansion from a mastered plateau cluster.

    Args:
        card_id: A mastered card ID whose cluster should serve as the expansion anchor.
        count: Number of brand-new frontier concept cards to synthesize.

    Returns:
        Newly created concept cards and new deck size.
    """
    engine, _ = _get_engine()
    if not (0 <= card_id < engine.n):
        return {"error": f"Card ID {card_id} out of bounds"}
    mutator = _get_mutator()
    if not mutator:
        return {"error": "CardMutator is not configured (missing API key)"}
    return mutator.expand_plateau_cluster(engine, card_id, count=count)


def main():
    """Run the Equimem MCP server via standard IO or SSE."""
    transport = "stdio"
    if "--sse" in sys.argv:
        transport = "sse"
    server.run(transport=transport)


if __name__ == "__main__":
    main()
