"""Unit tests for the AITutor class."""
from unittest.mock import MagicMock
import pytest
from equimem import AITutor, MassFieldEngine, FieldConfig, NGramEmbedder, Card, DecisionGrader


def test_tutor_loop_with_mocked_grader():
    eng = MassFieldEngine(embedder=NGramEmbedder(dim=128), config=FieldConfig(seed=1))
    eng.add_card("What is an eigenvalue?", "Scalar factor scaling an eigenvector.", "math")
    eng.add_card("What is an eigenvector?", "Vector whose direction does not change under transformation.", "math")

    # Mock DecisionGrader
    grader = DecisionGrader(api_key="mock_key")
    mock_client = MagicMock()
    grader.client = mock_client

    class MockAnswer:
        def __init__(self, score, conf):
            self.score = score
            self.confidence = conf
            self.probabilities = {"0": 0.05, "1": 0.15, "2": 0.80}

    mock_resp = MagicMock()
    mock_resp.answers = {"eval_0": MockAnswer(1.80, 0.90)}
    mock_client.decisions.create.return_value = mock_resp

    tutor = AITutor(engine=eng, grader=grader)

    # 1. Select card
    card = tutor.select_card(strategy="boundary")
    assert isinstance(card, Card)

    # 2. Submit student answer
    grade_res, review_update = tutor.submit_answer(card.id, "A scalar that scales the eigenvector.")
    assert grade_res.score == pytest.approx(0.90, 1e-2)
    assert review_update["score"] == pytest.approx(0.90, 1e-2)
    assert eng.masses[card.id] > 1.0


def test_tutor_learning_bridge():
    eng = MassFieldEngine(embedder=NGramEmbedder(dim=128))
    eng.add_card("Matrix multiplication", "Row by column dot products.", "math")
    eng.add_card("Eigenvalue decomposition", "Factoring matrix using eigenvectors.", "math")

    # Mark matrix multiplication as heavily mastered
    for _ in range(5):
        eng.review(0, 1.0)

    tutor = AITutor(engine=eng, grader=MagicMock())
    bridge = tutor.find_learning_bridge("eigenvalues and matrices")

    assert bridge["mastered_anchor"] is not None
    assert bridge["target_gap"] is not None
