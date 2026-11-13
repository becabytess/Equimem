"""Unit tests for the core MassFieldEngine mechanics."""
import pytest
import numpy as np
from geo_anki import MassFieldEngine, FieldConfig, NGramEmbedder


def test_mass_conservation_and_ripple():
    eng = MassFieldEngine(
        embedder=NGramEmbedder(dim=128),
        config=FieldConfig(tau=0.15, alpha_gain=0.7, alpha_loss=0.8, seed=42),
    )

    cards = [
        ("What is an eigenvalue?", "Scalar lambda where Av = lambda v.", "math"),
        ("What is an eigenvector?", "Non-zero vector v whose direction is unchanged by A.", "math"),
        ("What is SVD?", "Singular Value Decomposition factoring matrix into U Sigma V^T.", "math"),
        ("What is a croissant?", "A flaky French buttery pastry.", "food"),
        ("What is sourdough?", "Fermented bread dough using wild lactobacillus yeast.", "food"),
    ]
    eng.add_cards(cards)

    assert eng.n == 5
    # Initial mass sum == N
    assert pytest.approx(float(eng.masses.sum()), 1e-5) == 5.0

    # Successful review of card 0 (eigenvalue)
    eng.review(0, score=1.0)

    # Mass MUST still be strictly conserved at N
    assert pytest.approx(float(eng.masses.sum()), 1e-5) == 5.0

    m = eng.masses
    # Card 0 must gain absolute mass
    assert m[0] > 1.0
    # Semantically close math card 1 (eigenvector) must gain relative ripple mass over unrelated cards
    assert m[1] > m[3]
    assert m[1] > m[4]


def test_density_normalized_surface_and_gradient():
    eng = MassFieldEngine(embedder=NGramEmbedder(dim=128))
    eng.add_card("Quantum entanglement", "Non-local correlation between particles.", "physics")
    eng.add_card("Bell theorem", "Proof that quantum mechanics violates local hidden variables.", "physics")
    eng.add_card("Aspect experiment", "1982 experimental test of Bell inequality.", "physics")

    snap = eng.surface_state()
    assert len(snap.m_bar) == 3
    assert snap.gradients.shape == (3, 128)
    # At uniform initial mass, surface is flat
    assert pytest.approx(snap.variance, 1e-4) == 0.0


def test_card_selection_valley_and_boundary():
    eng = MassFieldEngine(embedder=NGramEmbedder(dim=128))
    eng.add_card("Concept A", "Details A", "topic1")
    eng.add_card("Concept B", "Details B", "topic1")
    eng.add_card("Concept C", "Details C", "topic2")

    # Review A and B successfully multiple times, leave C untouched
    for _ in range(3):
        eng.review(0, 1.0)
        eng.review(1, 1.0)

    # Valley strategy must choose the neglected card C (lowest smoothed mastery)
    next_card = eng.select_next(strategy="valley", temperature=0.0)
    assert next_card == 2


def test_trajectory_search():
    eng = MassFieldEngine(embedder=NGramEmbedder(dim=128))
    eng.add_card("Derivative of x^2", "2x", "calculus")
    eng.add_card("Derivative of sin(x)", "cos(x)", "calculus")
    eng.add_card("Taylor series", "Polynomial approximation around a point", "calculus")
    eng.add_card("French bread baguette", "Long crusty loaf", "food")

    # Uphill search toward calculus
    res = eng.trajectory_search("differentiation calculus derivative", k=3, direction="uphill")
    assert len(res.hits) == 3
    hit_ids = [h.card_id for h in res.hits]
    assert 0 in hit_ids or 1 in hit_ids
    assert 3 not in hit_ids  # food card should not appear in top calculus hits
