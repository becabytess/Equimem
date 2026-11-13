"""Simulation harness for the 60-card Conserved-Mass Geometric Anki Engine.

Simulates multi-turn student sessions with varied natural language responses,
evaluated in real time by Gemini Decision 1, testing card selection dynamics,
mass conservation (sum m_i == N), semantic ripple, and active frontier exploration.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional
import numpy as np
from dotenv import load_dotenv

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from equimem.models import FieldConfig
from equimem.engine import MassFieldEngine
from equimem.embedder import GeminiEmbedder, Embedder
from equimem.grader import DecisionGrader
from equimem.template import format_card
from lab.deck_50 import CARDS_DATA
from lab.student_responses import STUDENT_RESPONSES

# Load API credentials from geo_anki/.env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class CachedGeminiEmbedder:
    """Wrapper that caches text embeddings to disk to prevent redundant API calls."""
    def __init__(self, cache_path: str, model: str = "gemini-embedding-001", dim: int = 3072):
        self.cache_path = cache_path
        self.dim = dim
        self.live = GeminiEmbedder(model=model, dim=dim, batch_size=32)
        self.cache: Dict[str, np.ndarray] = {}
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                keys = data["keys"]
                vecs = data["vectors"]
                for k, v in zip(keys, vecs):
                    self.cache[str(k)] = v
                print(f"[CACHE] Loaded {len(self.cache)} vectors from {cache_path}")
            except Exception as e:
                print(f"[CACHE] Failed to load cache: {e}. Starting fresh.")

    def _save(self):
        keys = list(self.cache.keys())
        if keys:
            vecs = np.array([self.cache[k] for k in keys], dtype=np.float64)
            np.savez_compressed(self.cache_path, keys=keys, vectors=vecs)

    def encode(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        missing = [t for t in texts if t not in self.cache]
        if missing:
            print(f"[EMBED] Encoding {len(missing)} texts via Gemini API...")
            new_vecs = self.live.encode(missing)
            for t, v in zip(missing, new_vecs):
                self.cache[t] = v
            self._save()
        return np.array([self.cache[t] for t in texts], dtype=np.float64)


def run_simulation(num_rounds: int = 40, cooldown: int = 10):
    cache_file = os.path.join(os.path.dirname(__file__), "cache_embeddings_60.npz")
    embedder = CachedGeminiEmbedder(cache_file, model="gemini-embedding-001", dim=3072)

    # Temperature 0.10 balances greedy valley-fixing with broad domain exploration
    cfg = FieldConfig(tau=0.15, alpha_gain=0.70, alpha_loss=0.85, select_temp=0.10)
    engine = MassFieldEngine(embedder=embedder, config=cfg)

    print(f"\n[LOADING DECK] Registering {len(CARDS_DATA)} cards into the Geometric Engine...")
    card_tuples = [
        (c["question"], c["answer"], c.get("topic", "General"), {"card_id": c["id"]})
        for c in CARDS_DATA
    ]
    engine.add_cards(card_tuples)

    print(f"\n[ENGINE INITIALIZED] Deck size: {engine.n} cards | Dimension: {engine.dim}")
    print(f"Total mass: {float(engine.m.sum()):.4f} (Conservation invariant = {engine.n}.0)")

    # Grader using gemini-decision-1
    grader = DecisionGrader(model="gemini-decision-1")

    # Define topics mapping
    topics = sorted(list({c["topic"] for c in CARDS_DATA}))

    def topic_stats(step_label: str):
        snap = engine.surface_state()
        print(f"\n--- {step_label} Mastery Distribution ---")
        print(f"{'Topic':<25} | {'Count':<5} | {'Avg Mass m':<12} | {'Avg Mastery m_bar':<18} | {'Min m_bar':<10} | {'Max m_bar':<10}")
        print("-" * 88)
        for t in topics:
            idxs = [i for i, c in enumerate(CARDS_DATA) if c["topic"] == t]
            m_t = snap.m[idxs]
            mb_t = snap.m_bar[idxs]
            print(f"{t:<25} | {len(idxs):<5} | {float(m_t.mean()):<12.4f} | {float(mb_t.mean()):<18.4f} | {float(mb_t.min()):<10.4f} | {float(mb_t.max()):<10.4f}")
        print(f"Global Invariant: Total Mass = {float(snap.m.sum()):.6f} (target {engine.n}) | Variance(m_bar) = {snap.variance:.5f}")

    topic_stats("Initial State (T=0, Flat Field)")

    recent_history: List[int] = []
    
    print("\n" + "=" * 90)
    print("PHASE 1: Uneven Student Knowledge Profile (Rounds 1 - 20)")
    print("  - Linear Algebra & Algorithms: Mastered (High accuracy)")
    print("  - Quantum Mechanics: Severe Confusion / Misconceptions (Failures)")
    print("  - Information Theory: Partial Knowledge (Average scores)")
    print("  - Deep Learning: Unvisited frontier (Flat baseline)")
    print("=" * 90)

    for step in range(1, num_rounds + 1):
        if step == 21:
            print("\n" + "=" * 90)
            print("PHASE 2: Active Remediation & Knowledge Transfer (Rounds 21 - 40)")
            print("  - Student studies Quantum Mechanics: Now answering with Mastered accuracy!")
            print("  - Student also begins mastering Information Theory.")
            print("  - Observe how engine detects rising valley, shifts away from QM, and targets")
            print("    remaining unmastered topics and unvisited Deep Learning cards.")
            print("=" * 90)

        # Select next card with valley / boundary priority, enforcing cooldown window
        exclude_set = recent_history[-cooldown:] if len(recent_history) >= cooldown else recent_history
        selected_idx = engine.select_next(strategy="boundary", temperature=0.10, exclude=exclude_set)
        recent_history.append(selected_idx)

        card_info = CARDS_DATA[selected_idx]
        topic = card_info["topic"]
        q_text = card_info["question"]

        # Student response determination based on profile
        if step < 21:
            if topic in ("Linear Algebra", "Algorithms"):
                mode = "mastered"
            elif topic == "Quantum Mechanics":
                mode = "mistake"
            elif topic == "Information Theory":
                mode = "partial"
            else:
                mode = "partial"
        else:
            # Phase 2: Quantum remediated + Info Theory mastered
            if topic in ("Linear Algebra", "Algorithms", "Quantum Mechanics", "Information Theory"):
                mode = "mastered"
            else:
                mode = "partial"

        student_answer = STUDENT_RESPONSES[selected_idx][mode]

        # Call DecisionGrader (Gemini Decision 1 API)
        grade_res = grader.grade(card=engine.cards[selected_idx], student_answer=student_answer)
        score = grade_res.score

        # Pre-update state for logging
        m_before = float(engine.m[selected_idx])
        mb_before = float(engine.m_bar[selected_idx])

        # Engine review update
        engine.review(selected_idx, score=score)

        m_after = float(engine.m[selected_idx])
        mb_after = float(engine.m_bar[selected_idx])
        delta_m = m_after - m_before

        print(
            f"[R{step:02d}] Card {selected_idx:02d} [{topic[:16]:<16}] | Mode: {mode:<8} | "
            f"Score: {score:.3f} | m: {m_before:.3f}->{m_after:.3f} ({delta_m:+.3f}) | m_bar: {mb_before:.3f}->{mb_after:.3f}"
        )
        print(f"      Q: {q_text[:65]}...")
        print(f"      Decision Probs: P0={grade_res.level_probabilities['0']:.2f}, P1={grade_res.level_probabilities['1']:.2f}, P2={grade_res.level_probabilities['2']:.2f}")

        # Invariant check
        current_sum = float(engine.m.sum())
        assert abs(current_sum - engine.n) < 1e-9, f"Mass conservation violated: {current_sum} != {engine.n}"

        if step == 20:
            topic_stats("End of Phase 1 (Step 20)")

    topic_stats("Final State (Step 40)")

    # Test Trajectory Search (Downhill to find remaining gaps vs Uphill to find mastered anchors)
    print("\n" + "=" * 90)
    print("TRAJECTORY DYNAMICS TEST: Spherical Gradient Drift in Latent Space")
    print("=" * 90)

    # 1. Downhill search from "quantum superposition" query: find biggest unmastered gaps
    traj_down = engine.trajectory_search("quantum states and superposition", k=3, direction="downhill")
    print(f"\n[DOWNHILL TRAJECTORY - Gap Finding] Query: 'quantum states and superposition'")
    for rank, hit in enumerate(traj_down.hits, 1):
        print(f"  #{rank}: Card {hit.card_id:02d} | Score: {hit.score:.4f} | Mass: {hit.mass:.4f} | m_bar: {hit.smoothed_mass:.4f} | Q: {hit.question}")

    # 2. Uphill search: find mastered anchor concepts
    traj_up = engine.trajectory_search("eigenvalues and matrix decomposition", k=3, direction="uphill")
    print(f"\n[UPHILL TRAJECTORY - Mastery Anchors] Query: 'eigenvalues and matrix decomposition'")
    for rank, hit in enumerate(traj_up.hits, 1):
        print(f"  #{rank}: Card {hit.card_id:02d} | Score: {hit.score:.4f} | Mass: {hit.mass:.4f} | m_bar: {hit.smoothed_mass:.4f} | Q: {hit.question}")

    print("\n[SUCCESS] Simulation completed. Mass conservation holds strictly across all rounds.")


if __name__ == "__main__":
    run_simulation(num_rounds=40, cooldown=10)
