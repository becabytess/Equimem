"""Equimem Command Line Interface for Socratic AI Agents and Terminal Sessions."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from equimem.engine import MassFieldEngine
from equimem.models import FieldConfig
from equimem.embedder import CachedGeminiEmbedder, NGramEmbedder

DEFAULT_STATE_FILE = "equimem_state.npz"
DEFAULT_CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "lab", "cache_embeddings_60.npz")


def _get_embedder(cache_path: str = DEFAULT_CACHE_FILE):
    """Load cached embedder with automatic fallback."""
    if os.path.exists(cache_path):
        return CachedGeminiEmbedder(cache_path=cache_path, dim=3072)
    # Check if API key exists
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        return CachedGeminiEmbedder(cache_path=cache_path, dim=3072)
    return NGramEmbedder(dim=512)


def get_or_create_engine(state_path: str = DEFAULT_STATE_FILE) -> MassFieldEngine:
    """Load existing engine from state file or initialize from standard 60-card deck."""
    if not state_path.endswith(".npz"):
        state_path += ".npz"

    embedder = _get_embedder()

    if os.path.exists(state_path):
        return MassFieldEngine.load(state_path, embedder=embedder)

    # Initialize from standard 60-card deck
    from lab.deck_50 import CARDS_DATA
    cfg = FieldConfig(tau=0.15, alpha_gain=0.70, alpha_loss=0.85)
    engine = MassFieldEngine(embedder=embedder, config=cfg)

    cards = [(c["question"], c["answer"], c.get("topic", "general")) for c in CARDS_DATA]
    engine.add_cards(cards)
    engine.save(state_path)
    return engine


def cmd_state(args) -> None:
    """Print current MentalState snapshot as JSON."""
    engine = get_or_create_engine(args.state)
    state = engine.get_mental_state(top_k=args.top_k)
    print(json.dumps(state, indent=2))


def cmd_review(args) -> None:
    """Apply batch evaluations for cards touched in a conversation session."""
    engine = get_or_create_engine(args.state)
    try:
        scores = json.loads(args.scores)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON string in --scores: {e}"}), file=sys.stderr)
        sys.exit(1)

    result = engine.batch_review(scores)
    engine.save(args.state)
    print(json.dumps(result, indent=2))


def cmd_cards(args) -> None:
    """List deck cards with their current mass and smoothed mastery."""
    engine = get_or_create_engine(args.state)
    snap = engine.surface_state()
    items = []
    for i, c in enumerate(engine.cards):
        if args.topic and args.topic.lower() not in c.topic.lower():
            continue
        items.append({
            "id": i,
            "topic": c.topic,
            "question": c.question,
            "mass": round(float(engine.m[i]), 4),
            "smoothed_mastery": round(float(snap.m_bar[i]), 4),
            "reviews": c.reviews,
        })
    print(json.dumps(items, indent=2))


def cmd_trajectory(args) -> None:
    """Run gradient drift search on the manifold."""
    engine = get_or_create_engine(args.state)
    traj = engine.trajectory_search(args.query, k=args.k, direction=args.direction)
    hits = [
        {
            "card_id": h.card_id,
            "question": h.question,
            "topic": h.topic,
            "score": round(float(h.score), 4),
            "smoothed_mass": round(float(h.smoothed_mass), 4),
        }
        for h in traj.hits
    ]
    print(json.dumps({"query": traj.query, "direction": traj.direction, "hits": hits}, indent=2))


def cmd_init(args) -> None:
    """Initialize a fresh state file."""
    if os.path.exists(args.state) and not args.force:
        print(json.dumps({"message": f"State file {args.state} already exists. Use --force to reinitialize."}))
        return

    from lab.deck_50 import CARDS_DATA
    embedder = _get_embedder()
    cfg = FieldConfig(tau=0.15, alpha_gain=0.70, alpha_loss=0.85)
    engine = MassFieldEngine(embedder=embedder, config=cfg)
    cards = [(c["question"], c["answer"], c.get("topic", "general")) for c in CARDS_DATA]
    engine.add_cards(cards)
    engine.save(args.state)
    print(json.dumps({"message": f"Initialized fresh state file {args.state} with {engine.n} cards."}))


def cmd_add(args) -> None:
    """Add a new concept card to the manifold."""
    engine = get_or_create_engine(args.state)
    new_ids = engine.add_cards([(args.question, args.answer, args.topic)])
    engine.save(args.state)
    cid = new_ids[0]
    print(json.dumps({
        "status": "added",
        "card_id": cid,
        "question": args.question,
        "answer": args.answer,
        "topic": args.topic,
        "mass": round(float(engine.m[cid]), 4),
        "deck_size": engine.n,
        "total_mass": round(float(engine.m.sum()), 4),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Equimem CLI: Continuous Conserved-Memory Engine")
    parser.add_argument("--state", default=DEFAULT_STATE_FILE, help="Path to engine state file (.npz)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # State subcommand
    p_state = subparsers.add_parser("state", help="Export ranked mental state (valleys, anchors, frontier)")
    p_state.add_argument("--top-k", type=int, default=4, help="Number of cards per category")
    p_state.set_defaults(func=cmd_state)

    # Review subcommand
    p_review = subparsers.add_parser("review", help="Apply batch evaluations for touched cards")
    p_review.add_argument("--scores", required=True, help="JSON map of card_id to score: '{\"12\": 0.95}'")
    p_review.set_defaults(func=cmd_review)

    # Add subcommand
    p_add = subparsers.add_parser("add", help="Add a new concept card to the manifold")
    p_add.add_argument("--question", required=True, help="Question that elicits the concept")
    p_add.add_argument("--answer", required=True, help="Canonical answer in student's own words")
    p_add.add_argument("--topic", default="General", help="Topic domain")
    p_add.set_defaults(func=cmd_add)

    # Cards subcommand
    p_cards = subparsers.add_parser("cards", help="List cards in deck")
    p_cards.add_argument("--topic", help="Filter by topic")
    p_cards.set_defaults(func=cmd_cards)

    # Trajectory subcommand
    p_traj = subparsers.add_parser("trajectory", help="Run spherical gradient drift search")
    p_traj.add_argument("query", help="Semantic inquiry string")
    p_traj.add_argument("--direction", choices=["downhill", "uphill"], default="downhill")
    p_traj.add_argument("--k", type=int, default=4)
    p_traj.set_defaults(func=cmd_trajectory)

    # Init subcommand
    p_init = subparsers.add_parser("init", help="Initialize fresh state file")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing state")
    p_init.set_defaults(func=cmd_init)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
