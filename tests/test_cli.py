"""Unit tests for Mental State extraction, batch review, and Equimem CLI."""
from __future__ import annotations

import json
import pytest
import numpy as np

from equimem.engine import MassFieldEngine
from equimem.models import FieldConfig
from equimem.embedder import NGramEmbedder


@pytest.fixture
def sample_engine():
    """Create a 6-card engine with varied masses."""
    embedder = NGramEmbedder(dim=64)
    cfg = FieldConfig(tau=0.15, alpha_gain=0.70, alpha_loss=0.85)
    engine = MassFieldEngine(embedder=embedder, config=cfg)
    engine.add_cards([
        ("What is quantum superposition?", "Linear combination of basis states.", "Quantum"),
        ("What is quantum entanglement?", "Non-local correlation between particles.", "Quantum"),
        ("What is the Born rule?", "Probability is squared amplitude modulus.", "Quantum"),
        ("What is an eigenvalue?", "Scalar lambda where Av = lambda v.", "Linear Algebra"),
        ("What is SVD?", "Factorization A = U Sigma V^T.", "Linear Algebra"),
        ("What is matrix condition number?", "Ratio of max to min singular value.", "Linear Algebra"),
    ])
    # Set deliberate uneven distribution: Linear Algebra strong, Quantum weak
    engine._m[0:3] = 0.5
    engine._m[3:6] = 1.5
    engine._conserve()
    return engine


def test_get_mental_state(sample_engine):
    ms = sample_engine.get_mental_state(top_k=2)

    assert "valleys" in ms
    assert "anchors" in ms
    assert "frontier" in ms
    assert "spikes" in ms
    assert "deck_summary" in ms

    assert len(ms["valleys"]) == 2
    assert len(ms["anchors"]) == 2
    assert ms["deck_summary"]["total_cards"] == 6
    assert abs(ms["deck_summary"]["total_mass"] - 6.0) < 1e-5

    # Valleys should be the Quantum cards (low mass ~ 0.5)
    valley_topics = [c["topic"] for c in ms["valleys"]]
    assert all(t == "Quantum" for t in valley_topics)

    # Anchors should be the Linear Algebra cards (high mass ~ 1.5)
    anchor_topics = [c["topic"] for c in ms["anchors"]]
    assert all(t == "Linear Algebra" for t in anchor_topics)


def test_batch_review(sample_engine):
    masses_before = sample_engine.m.copy()

    # Student succeeds on Quantum cards 0, 1 (score=1.0) and misses Linear Algebra card 5 (score=0.0)
    res = sample_engine.batch_review({"0": 1.0, "1": 1.0, "5": 0.0})

    assert res["reviews_applied"] == 3
    assert abs(res["total_mass"] - 6.0) < 1e-5

    # Card 0 and 1 mass must increase
    assert sample_engine.m[0] > masses_before[0]
    assert sample_engine.m[1] > masses_before[1]

    # Card 5 mass must decrease
    assert sample_engine.m[5] < masses_before[5]


def test_cli_state_and_review(tmp_path):
    import subprocess
    import sys

    state_file = str(tmp_path / "test_state.npz")

    # Run cli state (should auto-init)
    cmd1 = [sys.executable, "-m", "equimem.cli", "--state", state_file, "state", "--top-k", "2"]
    p1 = subprocess.run(cmd1, capture_output=True, text=True)
    assert p1.returncode == 0
    data1 = json.loads(p1.stdout)
    assert "valleys" in data1
    assert data1["deck_summary"]["total_cards"] == 60

    # Run cli review
    cmd2 = [sys.executable, "-m", "equimem.cli", "--state", state_file, "review", "--scores", '{"0": 1.0, "1": 0.0}']
    p2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert p2.returncode == 0
    data2 = json.loads(p2.stdout)
    assert data2["reviews_applied"] == 2
    assert "0" in data2["deltas"]
    assert data2["deltas"]["0"]["delta"] > 0
    assert data2["deltas"]["1"]["delta"] < 0
