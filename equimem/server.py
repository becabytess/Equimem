"""FastAPI server for Equimem: Continuous Conserved-Memory Engine.

Provides REST API endpoints for AI assistants, web clients, and agentic workflows
to interact with the conserved-mass knowledge manifold.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from equimem.models import FieldConfig
from equimem.engine import MassFieldEngine
from equimem.embedder import GeminiEmbedder, NGramEmbedder
from equimem.grader import DecisionGrader
from equimem.mutator import CardMutator
from lab.deck_50 import CARDS_DATA

# Load environment credentials
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# =============================================================================
# Global Application State
# =============================================================================

class AppState:
    engine: Optional[MassFieldEngine] = None
    grader: Optional[DecisionGrader] = None
    mutator: Optional[CardMutator] = None
    embedder_type: str = "gemini"
    cooldown_history: List[int] = []


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Equimem engine and models on startup if not already initialized."""
    if state.engine is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            print("[EQUIMEM SERVER] Initializing GeminiEmbedder, DecisionGrader & CardMutator...")
            try:
                cache_file = os.path.join(os.path.dirname(__file__), "..", "lab", "cache_embeddings_60.npz")
                if os.path.exists(cache_file):
                    from lab.simulate_50_cards import CachedGeminiEmbedder
                    embedder = CachedGeminiEmbedder(cache_file, model="gemini-embedding-001", dim=3072)
                else:
                    embedder = GeminiEmbedder(model="gemini-embedding-001", dim=3072)
                state.embedder_type = "gemini-embedding-001"
            except Exception as e:
                print(f"[EQUIMEM SERVER] Warning: Falling back to NGramEmbedder ({e})")
                embedder = NGramEmbedder(dim=512)
                state.embedder_type = "ngram-fallback"
            state.grader = DecisionGrader(model="gemini-decision-1")
            try:
                state.mutator = CardMutator()
            except Exception as e:
                print(f"[EQUIMEM SERVER] Warning: Mutator initialization skipped ({e})")
                state.mutator = None
        else:
            print("[EQUIMEM SERVER] No API key found. Using NGramEmbedder fallback.")
            embedder = NGramEmbedder(dim=512)
            state.embedder_type = "ngram-fallback"
            state.grader = None
            state.mutator = None

        cfg = FieldConfig(tau=0.15, alpha_gain=0.70, alpha_loss=0.85, select_temp=0.10)
        state.engine = MassFieldEngine(embedder=embedder, config=cfg)

        # Pre-populate with default 60-card concept deck if empty
        card_tuples = [
            (c["question"], c["answer"], c.get("topic", "General"), {"card_id": c["id"]})
            for c in CARDS_DATA
        ]
        state.engine.add_cards(card_tuples)
        print(f"[EQUIMEM SERVER] Loaded {state.engine.n} concept cards. Total Mass: {state.engine.m.sum():.4f}")

    yield


app = FastAPI(
    title="Equimem API",
    description="REST API for Continuous Conserved-Memory Engine (Equimem)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request / Response Schemas
# =============================================================================

class CardCreateRequest(BaseModel):
    question: str
    answer: str
    topic: str = "General"
    meta: Optional[Dict[str, Any]] = None


class CardResponse(BaseModel):
    id: int
    question: str
    answer: str
    topic: str
    reviews: int
    mass: float
    smoothed_mass: float


class NextCardRequest(BaseModel):
    strategy: str = Field(default="boundary", description="'boundary', 'valley', or 'flat_step'")
    temperature: Optional[float] = Field(default=0.10, description="Softmax selection temperature")
    exclude: Optional[List[int]] = Field(default=None, description="Card IDs to temporarily exclude")
    cooldown: int = Field(default=8, description="Number of recent cards to automatically exclude")


class NextCardResponse(BaseModel):
    id: int
    question: str
    topic: str
    reviews: int
    mass: float
    smoothed_mass: float


class ReviewRequest(BaseModel):
    card_id: int
    student_answer: str


class ReviewResponse(BaseModel):
    card_id: int
    score: float
    level_probabilities: Dict[str, float]
    mass_before: float
    mass_after: float
    delta_mass: float
    smoothed_mass: float
    reference_answer: str
    total_deck_mass: float


class TrajectoryRequest(BaseModel):
    query: str
    direction: str = Field(default="downhill", description="'downhill' (gaps) or 'uphill' (anchors)")
    k: int = Field(default=5, description="Number of top hits to return")


class TrajectoryHitResponse(BaseModel):
    card_id: int
    question: str
    score: float
    mass: float
    smoothed_mass: float


class TrajectoryResponse(BaseModel):
    query: str
    direction: str
    hits: List[TrajectoryHitResponse]


class BatchReviewRequest(BaseModel):
    scores: Dict[str, float] = Field(..., description="Map of card_id to score in [0.0, 1.0]")


class HealthResponse(BaseModel):
    status: str
    deck_size: int
    total_mass: float
    embedder: str
    grader_available: bool


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/api/health", response_model=HealthResponse)
def get_health():
    eng = state.engine
    if not eng:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return HealthResponse(
        status="healthy",
        deck_size=eng.n,
        total_mass=float(eng.m.sum()),
        embedder=state.embedder_type,
        grader_available=state.grader is not None,
    )


@app.get("/api/state")
def get_state():
    eng = state.engine
    if not eng:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    snap = eng.surface_state()
    topics = sorted(list({c.topic for c in eng.cards}))
    topic_breakdown = []
    for t in topics:
        idxs = [i for i, c in enumerate(eng.cards) if c.topic == t]
        topic_breakdown.append({
            "topic": t,
            "count": len(idxs),
            "avg_mass": float(snap.m[idxs].mean()),
            "avg_smoothed_mass": float(snap.m_bar[idxs].mean()),
            "min_smoothed_mass": float(snap.m_bar[idxs].min()),
            "max_smoothed_mass": float(snap.m_bar[idxs].max()),
        })

    return {
        "deck_size": eng.n,
        "total_mass": float(snap.m.sum()),
        "surface_variance": snap.variance,
        "entropy": snap.entropy,
        "mean_gradient_norm": snap.mean_grad_norm,
        "topics": topic_breakdown,
    }


@app.get("/api/cards", response_model=List[CardResponse])
def get_cards():
    eng = state.engine
    if not eng:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    snap = eng.surface_state()
    return [
        CardResponse(
            id=c.id,
            question=c.question,
            answer=c.answer,
            topic=c.topic,
            reviews=c.reviews,
            mass=float(snap.m[c.id]),
            smoothed_mass=float(snap.m_bar[c.id]),
        )
        for c in eng.cards
    ]


@app.post("/api/cards", status_code=status.HTTP_201_CREATED)
def add_cards(cards: List[CardCreateRequest]):
    eng = state.engine
    if not eng:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    tuples = [(c.question, c.answer, c.topic, c.meta or {}) for c in cards]
    new_ids = eng.add_cards(tuples)
    return {"message": f"Successfully added {len(new_ids)} cards", "card_ids": new_ids}


@app.post("/api/session/next", response_model=NextCardResponse)
def select_next(req: NextCardRequest):
    eng = state.engine
    if not eng or eng.n == 0:
        raise HTTPException(status_code=400, detail="Deck is empty")

    exclude_set = set(req.exclude or [])
    if req.cooldown > 0 and state.cooldown_history:
        exclude_set.update(state.cooldown_history[-req.cooldown:])

    card_id = eng.select_next(
        strategy=req.strategy,
        temperature=req.temperature,
        exclude=exclude_set,
    )
    state.cooldown_history.append(card_id)

    card = eng.cards[card_id]
    snap = eng.surface_state()
    return NextCardResponse(
        id=card.id,
        question=card.question,
        topic=card.topic,
        reviews=card.reviews,
        mass=float(snap.m[card_id]),
        smoothed_mass=float(snap.m_bar[card_id]),
    )


@app.post("/api/session/review", response_model=ReviewResponse)
def review_card(req: ReviewRequest):
    eng = state.engine
    if not eng or not (0 <= req.card_id < eng.n):
        raise HTTPException(status_code=404, detail=f"Card ID {req.card_id} not found")

    card = eng.cards[req.card_id]
    m_before = float(eng.m[req.card_id])

    # Grade with gemini-decision-1 if available, otherwise heuristic fallback
    if state.grader:
        grade_res = state.grader.grade(card=card, student_answer=req.student_answer)
        score = grade_res.score
        probs = grade_res.level_probabilities
    else:
        # Fallback scoring
        score = 0.5
        probs = {"0": 0.25, "1": 0.50, "2": 0.25}

    eng.review(req.card_id, score=score)

    snap = eng.surface_state()
    m_after = float(snap.m[req.card_id])
    mb_after = float(snap.m_bar[req.card_id])

    return ReviewResponse(
        card_id=card.id,
        score=score,
        level_probabilities=probs,
        mass_before=m_before,
        mass_after=m_after,
        delta_mass=m_after - m_before,
        smoothed_mass=mb_after,
        reference_answer=card.answer,
        total_deck_mass=float(snap.m.sum()),
    )


@app.post("/api/search/trajectory", response_model=TrajectoryResponse)
def trajectory_search(req: TrajectoryRequest):
    eng = state.engine
    if not eng:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    res = eng.trajectory_search(query=req.query, k=req.k, direction=req.direction)
    hits = [
        TrajectoryHitResponse(
            card_id=h.card_id,
            question=h.question,
            score=h.score,
            mass=h.mass,
            smoothed_mass=h.smoothed_mass,
        )
        for h in res.hits
    ]
    return TrajectoryResponse(query=res.query, direction=res.direction, hits=hits)


@app.get("/api/session/mental-state")
def get_mental_state(top_k: int = 4):
    """Retrieve ranked mental state snapshot (valleys, anchors, frontier) for Socratic tutoring."""
    eng = state.engine
    if not eng:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return eng.get_mental_state(top_k=top_k)


@app.post("/api/session/batch-review")
def batch_review(req: BatchReviewRequest):
    """Apply session review scores across multiple concept cards and conserve mass."""
    eng = state.engine
    if not eng:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return eng.batch_review(req.scores)


# =============================================================================
# On-Demand Evolution Endpoints
# =============================================================================

@app.get("/api/cards/{card_id}/diagnostics")
def get_card_diagnostics(card_id: int):
    eng = state.engine
    if not eng or not (0 <= card_id < eng.n):
        raise HTTPException(status_code=404, detail="Card not found")
    return eng.diagnose_card(card_id)


@app.post("/api/cards/{card_id}/evolve")
def evolve_card(card_id: int):
    eng = state.engine
    if not eng or not (0 <= card_id < eng.n):
        raise HTTPException(status_code=404, detail="Card not found")
    if not state.mutator:
        raise HTTPException(status_code=503, detail="Card mutator not configured")
    try:
        return state.mutator.evolve_card(eng, card_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evolution failed: {e}")


@app.post("/api/cards/{card_id}/expand")
def expand_cluster(card_id: int, count: int = 2):
    eng = state.engine
    if not eng or not (0 <= card_id < eng.n):
        raise HTTPException(status_code=404, detail="Card not found")
    if not state.mutator:
        raise HTTPException(status_code=503, detail="Card mutator not configured")
    try:
        return state.mutator.expand_plateau_cluster(eng, card_id, count=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expansion failed: {e}")


def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[EQUIMEM SERVER] Starting FastAPI server on http://{host}:{port}")
    uvicorn.run("equimem.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
