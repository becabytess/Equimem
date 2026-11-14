"""Unit tests for the Leave-One-Out Regional Diagnostics and In-Place Card Mutation."""
from __future__ import annotations

import pytest
import numpy as np

from equimem.engine import MassFieldEngine
from equimem.models import FieldConfig
from equimem.embedder import NGramEmbedder
from equimem.mutator import CardMutator


@pytest.fixture
def test_engine():
    """Create a 4-card engine with an isolated spike on Card 0."""
    embedder = NGramEmbedder(dim=128)
    cfg = FieldConfig(tau=0.15, alpha_gain=0.70, alpha_loss=0.85)
    engine = MassFieldEngine(embedder=embedder, config=cfg)
    engine.add_cards([
        ("What is an eigenvalue?", "Scalar lambda such that Av = lambda v.", "Linear Algebra"),
        ("What is a matrix condition number?", "Ratio of max to min singular value.", "Linear Algebra"),
        ("What is the spectral norm of a matrix?", "Largest singular value of A.", "Linear Algebra"),
        ("What is Singular Value Decomposition?", "Factorization A = U Sigma V^T.", "Linear Algebra"),
    ])
    # Manually configure an isolated spike: Card 0 has high mass, neighbors have low mass
    engine._m[0] = 2.2
    engine._m[1] = 0.6
    engine._m[2] = 0.6
    engine._m[3] = 0.6
    engine._conserve()
    return engine


def test_leave_one_out_diagnostics(test_engine):
    diag = test_engine.region_diagnostics()
    assert "m" in diag
    assert "m_bar_neighbors" in diag
    assert "spike_residual" in diag
    assert len(diag["m"]) == 4

    # Card 0 should have a large positive spike residual
    res_0 = test_engine.diagnose_card(0)
    assert res_0["state"] == "isolated_spike"
    assert res_0["recommended_action"] == "replace"


def test_replace_card_in_place(test_engine):
    old_q = test_engine.cards[0].question
    mass_before = float(test_engine.m[0])

    res = test_engine.replace_card(
        card_id=0,
        new_question="How do eigenvalues govern vanishing gradients in deep networks?",
        new_answer="Spectral radius < 1 causes repeated Jacobian products to attenuate gradients.",
    )

    assert res["card_id"] == 0
    assert res["generation"] == 1
    assert res["old_question"] == old_q
    assert test_engine.cards[0].question == "How do eigenvalues govern vanishing gradients in deep networks?"

    # Card mass should have relaxed down from 2.0+ towards neighbor level
    assert res["mass_after"] < mass_before

    # Total mass across the deck must remain strictly 4.0
    assert abs(float(test_engine.m.sum()) - 4.0) < 1e-6


def test_consolidated_plateau_diagnosis():
    embedder = NGramEmbedder(dim=128)
    cfg = FieldConfig(tau=0.15)
    engine = MassFieldEngine(embedder=embedder, config=cfg)
    # 6-card deck: Cluster A (cards 0-2) and Cluster B (cards 3-5)
    engine.add_cards([
        ("Linear Algebra: Card 1", "Ans 1", "LA"),
        ("Linear Algebra: Card 2", "Ans 2", "LA"),
        ("Linear Algebra: Card 3", "Ans 3", "LA"),
        ("Quantum: Card 4", "Ans 4", "QM"),
        ("Quantum: Card 5", "Ans 5", "QM"),
        ("Quantum: Card 6", "Ans 6", "QM"),
    ])
    # Cluster A has uniformly high mass, Cluster B has lower mass, total conserved to 6.0
    engine._m[0:3] = 1.4
    engine._m[3:6] = 0.6
    engine._conserve()

    res = engine.diagnose_card(0)
    assert res["state"] == "consolidated_plateau"
    assert res["recommended_action"] == "expand"
