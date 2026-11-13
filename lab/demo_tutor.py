"""Interactive live demonstration of the Geometric AI Tutor powered by Google models."""
import os
import sys

# Ensure local package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from equimem import (
    AITutor,
    DecisionGrader,
    FieldConfig,
    GeminiEmbedder,
    MassFieldEngine,
)

print("=" * 70)
print("GEOMETRIC AI TUTOR (100% GOOGLE STACK: GEMINI EMBEDDING + DECISION)")
print("=" * 70)

# Initialize engine with GeminiEmbedder
embedder = GeminiEmbedder()
config = FieldConfig(tau=0.15, alpha_gain=0.75, alpha_loss=0.85, seed=42)
engine = MassFieldEngine(embedder=embedder, config=config)

# Initialize grader with Gemini Decision API
grader = DecisionGrader()
tutor = AITutor(engine=engine, grader=grader)

# Seed realistic multi-domain card deck
deck = [
    (
        "Why does Alain Aspect's 1982 experiment rule out local hidden variable theories?",
        "Because it measured entangled photon pairs at spacelike separation and observed violations of Bell inequalities, demonstrating quantum nonlocality.",
        "quantum_physics",
    ),
    (
        "What is Bell's theorem and what does it state about local realism?",
        "It states that no physical theory of local hidden variables can ever reproduce all the statistical predictions of quantum mechanics.",
        "quantum_physics",
    ),
    (
        "What was the EPR paradox designed to argue about quantum mechanics?",
        "Einstein, Podolsky, and Rosen argued that entanglement implies incomplete quantum mechanics unless faster-than-light action exists.",
        "quantum_physics",
    ),
    (
        "What is Singular Value Decomposition (SVD)?",
        "A matrix factorization factoring any m x n matrix A into U Sigma V^T, where U and V are orthogonal and Sigma contains singular values.",
        "linear_algebra",
    ),
    (
        "What is the geometric meaning of an eigenvector and eigenvalue?",
        "An eigenvector is a non-zero vector whose spatial direction is unchanged by linear transformation A, scaled by factor lambda (eigenvalue).",
        "linear_algebra",
    ),
]

print(f"\nAdding {len(deck)} structured concept cards to the engine...")
cids = engine.add_cards(deck)
for cid in cids:
    card = engine.cards[cid]
    print(f"  Card #{card.id} [{card.topic}]: {card.question[:65]}...")

print(f"\nInitial State: Total mass = {engine.masses.sum():.2f} across {engine.n} cards (mean = 1.00)")

# --- Round 1: Socratic Frontier Selection & Evaluation ---
print("\n" + "-" * 70)
print("ROUND 1: Tutor selects card at the active learning frontier (boundary strategy)")
print("-" * 70)

selected_card = tutor.select_card(strategy="boundary")
print(f"Tutor Prompts Student with Card #{selected_card.id} [{selected_card.topic}]:")
print(f"  Q: \"{selected_card.question}\"")
print(f"  Reference Ground Truth: \"{selected_card.answer}\"")

# Simulated student gives a free-form answer in their own words
student_response_1 = (
    "SVD factors a matrix A into three matrices U, Sigma, and V transpose, "
    "where U and V are orthogonal matrices and Sigma is a diagonal matrix of singular values."
)
print(f"\nStudent submits free-form answer:")
print(f"  \"{student_response_1}\"")

print("\nGrading with gemini-decision-1 (single prefill pass, zero output tokens)...")
grade_1, update_1 = tutor.submit_answer(selected_card.id, student_response_1)

print(f"Decision Result:")
print(f"  Score: {grade_1.score:.2f} / 1.00 (Raw Rubric: {grade_1.raw_score:.2f} / 2.0)")
print(f"  Confidence: {grade_1.confidence:.2%}")
print(f"  Probabilities: {grade_1.level_probabilities}")
print(f"  Feedback: {grade_1.feedback}")

print(f"\nMass Field Surface Update:")
print(f"  Card #{selected_card.id} mass: {update_1['mass_before']:.3f} -> {update_1['mass_after']:.3f}")
print(f"  Total mass conserved: {engine.masses.sum():.6f}")

print("\nImpact on neighboring cards (semantic ripple vs unreviewed deflation):")
for c in engine.cards:
    m = engine.masses[c.id]
    m_bar = engine.surface_state().m_bar[c.id]
    print(f"  Card #{c.id} [{c.topic:15s}] mass={m:.3f} | smoothed mastery={m_bar:.3f}")

# --- Round 2: Diagnosing Learning Bridges ---
print("\n" + "-" * 70)
print("ROUND 2: Conceptual Pathfinding (Trajectory Search for Learning Bridges)")
print("-" * 70)

query_topic = "quantum nonlocality and Bell theorem"
print(f"Diagnosing learning landscape around: \"{query_topic}\"")
bridge = tutor.find_learning_bridge(query_topic)

print(f"  Mastered Anchor (Uphill): Card #{bridge['mastered_anchor'].card_id} (Smoothed: {bridge['anchor_smoothed_mastery']:.3f})")
print(f"    -> {bridge['mastered_anchor'].question}")
print(f"  Unmastered Gap  (Downhill): Card #{bridge['target_gap'].card_id} (Smoothed: {bridge['gap_smoothed_mastery']:.3f})")
print(f"    -> {bridge['target_gap'].question}")

print("\n" + "=" * 70)
print("TUTOR SUMMARY: Surface updated smoothly, total mass strictly conserved.")
print("=" * 70)
