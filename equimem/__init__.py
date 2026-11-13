"""Conserved-Mass Geometric Anki & AI Tutor Engine."""
from __future__ import annotations

from .models import (
    Card,
    FieldConfig,
    FieldSnapshot,
    GradeResult,
    TrajectoryHit,
    TrajectoryResult,
)
from .template import format_card
from .embedder import Embedder, GeminiEmbedder, NGramEmbedder
from .engine import MassFieldEngine
from .grader import DecisionGrader
from .tutor import AITutor

__version__ = "0.1.0"

__all__ = [
    "MassFieldEngine",
    "FieldConfig",
    "Card",
    "FieldSnapshot",
    "GradeResult",
    "TrajectoryHit",
    "TrajectoryResult",
    "DecisionGrader",
    "GeminiEmbedder",
    "NGramEmbedder",
    "Embedder",
    "AITutor",
    "format_card",
    "__version__",
]
