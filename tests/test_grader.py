"""Unit tests for DecisionGrader."""
from unittest.mock import MagicMock
import pytest
from equimem import DecisionGrader, Card, GradeResult


def test_grader_mocked_success():
    grader = DecisionGrader(api_key="mock_key")
    mock_client = MagicMock()
    grader.client = mock_client

    class MockAnswer:
        def __init__(self, score, conf, probs):
            self.score = score
            self.confidence = conf
            self.probabilities = probs

    # Mock rubric returns: score=1.85 (correct), conf=0.92
    mock_resp = MagicMock()
    mock_resp.answers = {
        "eval_0": MockAnswer(1.85, 0.92, {"0": 0.02, "1": 0.08, "2": 0.90})
    }
    mock_client.decisions.create.return_value = mock_resp

    card = Card(id=0, question="What is Bell inequality?", answer="A theoretical bound violated by quantum mechanics.")
    res = grader.grade(card, "It is a mathematical bound on local realism that quantum mechanics violates.")

    assert isinstance(res, GradeResult)
    assert res.score == pytest.approx(1.85 / 2.0, 1e-3)
    assert res.confidence == 0.92
    assert "Mastered" in res.feedback
    assert grader.calls == 1
    assert grader.fails == 0


def test_grader_error_handling_fallback():
    grader = DecisionGrader(api_key="mock_key")
    mock_client = MagicMock()
    grader.client = mock_client
    mock_client.decisions.create.side_effect = RuntimeError("Network down")

    card = Card(id=1, question="Test question", answer="Test answer")
    res = grader.grade(card, "Some attempt")

    assert res.score == 0.0
    assert grader.fails == 1
