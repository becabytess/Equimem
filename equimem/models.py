"""Data models and configuration for the Geometric Conserved-Mass Anki Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class FieldConfig:
    tau: float = 0.14              # Smooth exponential kernel width; controls semantic ripple radius
    alpha_gain: float = 0.65       # Mass injection scale on recall success
    alpha_loss: float = 0.85       # Multiplicative dip scale on recall failure
    select_temp: float = 0.05      # Softmax temperature for card selection (0 = greedy argmin)
    traj_branches: int = 3         # Top-k biased starting branches per query
    traj_lambda: float = 0.65      # Bias strength toward initial match: norm(q + lam * v_a)
    traj_steps: int = 6            # Gradient drift steps per trajectory branch
    traj_eta: float = 0.45         # Spherical step size along local gradient
    seed: int = 42

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError("tau must be > 0")
        if self.alpha_gain < 0.0 or self.alpha_loss < 0.0:
            raise ValueError("alpha_gain and alpha_loss must be >= 0")


@dataclass
class Card:
    id: int
    question: str
    answer: str
    topic: str = "general"
    reviews: int = 0
    correct_sum: float = 0.0
    last_step: int = -1
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class FieldSnapshot:
    m: np.ndarray                  # Raw conserved masses (sum == N)
    m_bar: np.ndarray              # Density-normalized smoothed mass at each card
    potential: np.ndarray          # Unnormalized potential Phi(v_i)
    gradients: np.ndarray          # Smooth log-surface gradient at each card, shape (N, D)
    grad_norm: np.ndarray          # Gradient magnitude ||grad log m_bar(v_i)||
    variance: float                # Var(m_bar): global surface roughness
    entropy: float                 # Normalized entropy of (m / N) in [0, 1]
    mean_grad_norm: float          # Mean gradient magnitude across the surface
    min_to_max_ratio: float        # min(m_bar) / max(m_bar)


@dataclass
class TrajectoryHit:
    card_id: int
    question: str
    answer: str
    score: float
    mass: float
    smoothed_mass: float


@dataclass
class TrajectoryResult:
    query: str
    direction: str
    hits: List[TrajectoryHit]
    branch_seeds: List[int]
    paths: List[List[int]]         # Nearest card id at each step of each branch


@dataclass
class GradeResult:
    score: float                   # Expected score in [0.0, 1.0]
    raw_score: float               # Raw score in [0.0, 2.0] from rubric
    confidence: float              # Model confidence in [0.0, 1.0]
    level_probabilities: Dict[str, float]  # {'0': p0, '1': p1, '2': p2}
    feedback: str                  # Summary diagnostic text
