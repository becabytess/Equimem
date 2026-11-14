"""Unit tests for the Equimem MCP Server tools."""
from __future__ import annotations

import pytest
from equimem.mcp_server import (
    _get_engine,
    select_next_card,
    review_card,
    add_card,
    trajectory_search,
    get_knowledge_state,
)
from equimem.models import FieldConfig
from equimem.engine import MassFieldEngine
from equimem.embedder import NGramEmbedder
import equimem.mcp_server as mcp_mod


@pytest.fixture(autouse=True)
def setup_mcp_engine():
    """Initialize a fast, local engine for testing MCP tools."""
    embedder = NGramEmbedder(dim=128)
    cfg = FieldConfig(tau=0.15, alpha_gain=0.70, alpha_loss=0.85, select_temp=0.10)
    engine = MassFieldEngine(embedder=embedder, config=cfg)
    engine.add_cards([
        ("What is entropy?", "Expected information content.", "Info Theory"),
        ("What is KL divergence?", "Relative entropy between P and Q.", "Info Theory"),
        ("What is SVD?", "Matrix decomposition into orthogonal matrices.", "Linear Algebra"),
    ])
    mcp_mod._engine = engine
    mcp_mod._grader = None
    mcp_mod._cooldown_history = []


def test_mcp_get_knowledge_state():
    res = get_knowledge_state()
    assert res["deck_size"] == 3
    assert abs(res["total_mass"] - 3.0) < 1e-4
    assert len(res["topics"]) == 2


def test_mcp_select_next_card():
    res = select_next_card(strategy="boundary", temperature=0.10, cooldown=1)
    assert "card_id" in res
    assert 0 <= res["card_id"] < 3
    assert "question" in res
    assert "mass" in res


def test_mcp_review_card():
    res = review_card(card_id=0, student_answer="A measure of uncertainty or information.")
    assert res["card_id"] == 0
    assert "score" in res
    assert "mass_after" in res
    assert abs(res["total_deck_mass"] - 3.0) < 1e-4


def test_mcp_add_card():
    res = add_card(question="What is Bayes rule?", answer="Posterior formula.", topic="Info Theory")
    assert res["deck_size"] == 4
    assert res["card_id"] == 3


def test_mcp_trajectory_search():
    res = trajectory_search(query="information entropy", direction="downhill", k=2)
    assert res["query"] == "information entropy"
    assert len(res["hits"]) <= 2
