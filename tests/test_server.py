"""Unit tests for the Equimem FastAPI REST server."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from equimem.server import app, state
from equimem.engine import MassFieldEngine
from equimem.models import FieldConfig
from equimem.embedder import NGramEmbedder


@pytest.fixture(autouse=True)
def init_test_engine():
    """Ensure engine is initialized with deterministic fallback embedder for tests."""
    embedder = NGramEmbedder(dim=128)
    cfg = FieldConfig(tau=0.15, alpha_gain=0.70, alpha_loss=0.85, select_temp=0.10)
    state.engine = MassFieldEngine(embedder=embedder, config=cfg)
    state.grader = None
    state.embedder_type = "ngram-test"
    state.cooldown_history = []

    # Add 4 test cards
    state.engine.add_cards([
        ("What is entropy?", "Expected information content.", "Info Theory"),
        ("What is KL divergence?", "Relative entropy between P and Q.", "Info Theory"),
        ("What is SVD?", "Matrix decomposition A = U Sigma V^T.", "Linear Algebra"),
        ("What is an eigenvalue?", "Scalar lambda where Av = lambda v.", "Linear Algebra"),
    ])


def test_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["deck_size"] == 4
        assert abs(data["total_mass"] - 4.0) < 1e-6


def test_get_cards_and_state():
    with TestClient(app) as client:
        res_cards = client.get("/api/cards")
        assert res_cards.status_code == 200
        cards = res_cards.json()
        assert len(cards) == 4
        assert cards[0]["question"] == "What is entropy?"

        res_state = client.get("/api/state")
        assert res_state.status_code == 200
        st = res_state.json()
        assert st["deck_size"] == 4
        assert len(st["topics"]) == 2


def test_add_card():
    with TestClient(app) as client:
        payload = [
            {
                "question": "What is backpropagation?",
                "answer": "Chain rule gradient computation.",
                "topic": "Deep Learning",
            }
        ]
        res = client.post("/api/cards", json=payload)
        assert res.status_code == 201
        data = res.json()
        assert len(data["card_ids"]) == 1

        # Check that deck size updated
        res_health = client.get("/api/health")
        assert res_health.json()["deck_size"] == 5


def test_select_next_and_review():
    with TestClient(app) as client:
        # 1. Select next card
        req_next = {"strategy": "boundary", "temperature": 0.10, "cooldown": 2}
        res_next = client.post("/api/session/next", json=req_next)
        assert res_next.status_code == 200
        card_data = res_next.json()
        card_id = card_data["id"]
        assert 0 <= card_id < 4
        assert "answer" not in card_data  # Should not reveal answer upfront

        # 2. Review card
        req_review = {
            "card_id": card_id,
            "student_answer": "Average information measure.",
        }
        res_rev = client.post("/api/session/review", json=req_review)
        assert res_rev.status_code == 200
        rev_data = res_rev.json()
        assert rev_data["card_id"] == card_id
        assert abs(rev_data["total_deck_mass"] - 4.0) < 1e-6
        assert "reference_answer" in rev_data


def test_trajectory_search():
    with TestClient(app) as client:
        req = {"query": "matrix factorization and eigenvalues", "k": 2, "direction": "downhill"}
        res = client.post("/api/search/trajectory", json=req)
        assert res.status_code == 200
        data = res.json()
        assert data["query"] == "matrix factorization and eigenvalues"
        assert len(data["hits"]) <= 2
