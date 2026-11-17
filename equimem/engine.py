"""Conserved-mass geometric flashcard engine over the unit hypersphere."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

from .models import Card, FieldConfig, FieldSnapshot, TrajectoryHit, TrajectoryResult
from .embedder import Embedder, NGramEmbedder, _normalize_rows
from .template import format_card


class MassFieldEngine:
    """Conserved-mass geometric flashcard and knowledge-state engine.

    Total mass across N cards is strictly conserved at N (average mass = 1.0).
    - Initial cards enter flat at mass 1.0.
    - New cards added mid-study enter at mass 0.0 (an immediate valley),
      with existing cards smoothly re-normalized so sum(m) == N.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        config: Optional[FieldConfig] = None,
        card_formatter: Optional[Callable[[str, str, Optional[str]], str]] = None,
    ):
        self.emb: Embedder = embedder or NGramEmbedder()
        self.cfg: FieldConfig = config or FieldConfig()
        self.dim: int = int(self.emb.dim)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.format_card = card_formatter or format_card

        self.cards: List[Card] = []
        self._V = np.zeros((0, self.dim), dtype=np.float64)
        self._m = np.zeros(0, dtype=np.float64)
        self.step: int = 0

    @property
    def n(self) -> int:
        return len(self.cards)

    @property
    def V(self) -> np.ndarray:
        return self._V

    @property
    def masses(self) -> np.ndarray:
        return self._m.copy()

    @property
    def m(self) -> np.ndarray:
        return self._m.copy()

    @property
    def m_bar(self) -> np.ndarray:
        return self.surface_state().m_bar

    # ------------------------------------------------------------------ cards
    def add_card(
        self,
        question: str,
        answer: str,
        topic: str = "general",
        vector: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> int:
        ids = self.add_cards(
            [(question, answer, topic, meta)],
            vectors=None if vector is None else np.asarray(vector, dtype=np.float64)[None, :],
        )
        return ids[0]

    def add_cards(
        self,
        items: Sequence[Union[Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, Optional[Dict[str, object]]]]],
        vectors: Optional[np.ndarray] = None,
    ) -> List[int]:
        if not items:
            return []

        qs: List[str] = []
        ans: List[str] = []
        topics: List[str] = []
        metas: List[Dict[str, object]] = []

        for it in items:
            q = it[0]
            a = it[1]
            t = it[2] if len(it) > 2 and isinstance(it[2], str) else "general"
            m = it[3] if len(it) > 3 and it[3] is not None else {}
            qs.append(q)
            ans.append(a)
            topics.append(t)
            metas.append(dict(m))

        if vectors is None:
            texts = [self.format_card(q, a, t) for q, a, t in zip(qs, ans, topics)]
            vecs = _normalize_rows(self.emb.encode(texts))
        else:
            vecs = _normalize_rows(vectors)
            if vecs.shape != (len(items), self.dim):
                raise ValueError(f"Expected vectors shape {(len(items), self.dim)}, got {vecs.shape}")

        start_id = self.n
        new_ids = list(range(start_id, start_id + len(items)))
        for cid, q, a, t, meta in zip(new_ids, qs, ans, topics, metas):
            self.cards.append(Card(id=cid, question=q, answer=a, topic=t, meta=meta))

        if self._V.shape[0] == 0:
            self._V = vecs.copy()
        else:
            self._V = np.vstack([self._V, vecs])

        # Before studying starts (step == 0), deck initializes flat at 1.0.
        # Once study is underway (step > 0), new cards enter at 0.0 (valley).
        if self.step == 0:
            self._m = np.concatenate([self._m, np.ones(len(items), dtype=np.float64)])
        else:
            self._m = np.concatenate([self._m, np.zeros(len(items), dtype=np.float64)])
            self._conserve()

        return new_ids

    # ------------------------------------------------------------------ kernel & conservation
    def _kernel(self, x: np.ndarray) -> np.ndarray:
        """Smooth radial similarity kernel K(x, v_i) = exp((x . v_i - 1) / tau)."""
        sims = np.asarray(x, dtype=np.float64) @ self._V.T
        return np.exp((sims - 1.0) / self.cfg.tau)

    def _conserve(self) -> None:
        n = self.n
        if n == 0:
            return
        total = float(self._m.sum())
        if total <= 1e-12:
            self._m[:] = 1.0
        else:
            self._m *= n / total

    # ------------------------------------------------------------------ learning update
    def review(self, card_id: int, score: Union[bool, float]) -> Dict[str, float]:
        """Apply a review outcome for `card_id`.

        score: 1.0 / True = fully recalled, 0.0 / False = forgotten, or continuous [0.0, 1.0].
        """
        if not 0 <= card_id < self.n:
            raise IndexError(f"Invalid card_id {card_id}")

        y = float(np.clip(float(score), 0.0, 1.0))
        self.step += 1

        card = self.cards[card_id]
        card.reviews += 1
        card.correct_sum += y
        card.last_step = self.step

        m_before = float(self._m[card_id])
        k = self._kernel(self._V[card_id])  # shape (N,), k[card_id] == 1.0

        # Unified smooth mass update (no hard branching):
        gain_term = self.cfg.alpha_gain * y * k
        loss_factor = np.exp(-self.cfg.alpha_loss * (1.0 - y) * k)
        self._m = (self._m + gain_term) * loss_factor
        self._conserve()

        return {
            "step": float(self.step),
            "card_id": float(card_id),
            "score": y,
            "mass_before": m_before,
            "mass_after": float(self._m[card_id]),
        }

    # ------------------------------------------------------------------ surface & gradients
    def surface_at(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate (m_bar, potential, log_surface_grad) at vector(s) x on S^(D-1)."""
        single = x.ndim == 1
        X = x[None, :] if single else x
        if self.n == 0:
            z = np.zeros(X.shape[0], dtype=np.float64)
            g = np.zeros_like(X, dtype=np.float64)
            return (z[0], z[0], g[0]) if single else (z, z, g)

        K = self._kernel(X)                              # (B, N)
        rho0 = K.sum(axis=1)                             # Concept density
        P = K / np.maximum(rho0[:, None], 1e-12)         # Semantic distribution p_i(x)

        W_unnorm = K * self._m[None, :]                  # (B, N)
        phi = W_unnorm.sum(axis=1)                       # Raw potential Phi(x)
        W = W_unnorm / np.maximum(phi[:, None], 1e-12)   # Mass-weighted distribution w_i(x)

        m_bar = phi / np.maximum(rho0, 1e-12)            # Density-normalized surface

        # Exact analytical log-gradient: (1 / tau) * (E_w[v] - E_p[v])
        grad = ((W @ self._V) - (P @ self._V)) / self.cfg.tau
        if single:
            return m_bar[0], phi[0], grad[0]
        return m_bar, phi, grad

    def surface_state(self) -> FieldSnapshot:
        """Compute full geometric diagnostics of the current knowledge surface."""
        if self.n == 0:
            return FieldSnapshot(
                m=np.zeros(0),
                m_bar=np.zeros(0),
                potential=np.zeros(0),
                gradients=np.zeros((0, self.dim)),
                grad_norm=np.zeros(0),
                variance=0.0,
                entropy=1.0,
                mean_grad_norm=0.0,
                min_to_max_ratio=1.0,
            )

        m_bar, phi, grads = self.surface_at(self._V)
        grad_norm = np.linalg.norm(grads, axis=1)

        p = self._m / max(float(self._m.sum()), 1e-12)
        nz = p > 1e-15
        if self.n > 1:
            raw_ent = float(-np.sum(p[nz] * np.log(p[nz])))
            norm_ent = raw_ent / math.log(self.n)
        else:
            norm_ent = 1.0

        max_mb = float(m_bar.max())
        min_mb = float(m_bar.min())

        return FieldSnapshot(
            m=self._m.copy(),
            m_bar=m_bar,
            potential=phi,
            gradients=grads,
            grad_norm=grad_norm,
            variance=float(np.var(m_bar)),
            entropy=float(np.clip(norm_ent, 0.0, 1.0)),
            mean_grad_norm=float(grad_norm.mean()),
            min_to_max_ratio=(min_mb / max_mb) if max_mb > 1e-12 else 1.0,
        )

    # ------------------------------------------------------------------ card selection
    def card_priorities(self, strategy: str = "valley") -> np.ndarray:
        """Compute selection priority across all cards (higher = stronger review need)."""
        if self.n == 0:
            return np.zeros(0, dtype=np.float64)

        snap = self.surface_state()

        if strategy == "valley":
            return -snap.m_bar
        if strategy == "raw_valley":
            return -snap.m
        if strategy == "flat_step":
            K = self._kernel(self._V)
            P = K / np.maximum(K.sum(axis=1, keepdims=True), 1e-12)
            deficit = 1.0 - snap.m_bar
            return P.T @ deficit
        if strategy == "boundary":
            gn = snap.grad_norm / max(float(snap.grad_norm.max()), 1e-12)
            return -snap.m_bar + 0.5 * gn
        raise ValueError(f"Unknown selection strategy: {strategy!r}")

    def select_next(
        self,
        strategy: str = "valley",
        temperature: Optional[float] = None,
        exclude: Iterable[int] = (),
    ) -> int:
        """Select the next flashcard to review using smooth Boltzmann softmax."""
        if self.n == 0:
            raise ValueError("Cannot select from an empty deck")

        scores = self.card_priorities(strategy=strategy)
        excl = set(exclude)
        if excl and len(excl) < self.n:
            for idx in excl:
                if 0 <= idx < self.n:
                    scores[idx] = -np.inf

        temp = self.cfg.select_temp if temperature is None else float(temperature)
        if temp <= 0.0:
            best = float(scores.max())
            ties = np.where(np.abs(scores - best) <= 1e-12)[0]
            return int(self.rng.choice(ties))

        logits = scores / temp
        finite = np.isfinite(logits)
        if not np.any(finite):
            return int(self.rng.integers(self.n))
        logits = logits - float(logits[finite].max())
        probs = np.where(finite, np.exp(logits), 0.0)
        probs /= float(probs.sum())
        return int(self.rng.choice(self.n, p=probs))

    # ------------------------------------------------------------------ query trajectories
    def trajectory_search(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        branches: Optional[int] = None,
        steps: Optional[int] = None,
        direction: str = "downhill",
        lam: Optional[float] = None,
        eta: Optional[float] = None,
    ) -> TrajectoryResult:
        """Multi-branch spherical gradient drift retrieval from a query."""
        if self.n == 0:
            q_str = query if isinstance(query, str) else "<vector>"
            return TrajectoryResult(q_str, direction, [], [], [])

        if isinstance(query, str):
            q_str = query
            q_vec = _normalize_rows(self.emb.encode([query])[0])
        else:
            q_str = "<vector>"
            q_vec = _normalize_rows(np.asarray(query, dtype=np.float64))

        b = min(branches or self.cfg.traj_branches, self.n)
        n_steps = self.cfg.traj_steps if steps is None else int(steps)
        lmb = self.cfg.traj_lambda if lam is None else float(lam)
        step_eta = self.cfg.traj_eta if eta is None else float(eta)

        base_sims = self._V @ q_vec
        seeds = [int(i) for i in np.argsort(-base_sims)[:b]]

        accum = np.zeros(self.n, dtype=np.float64)
        paths: List[List[int]] = []

        for s in seeds:
            qt = _normalize_rows(q_vec + lmb * self._V[s])
            path: List[int] = []
            for _ in range(n_steps + 1):
                K_t = self._kernel(qt)
                accum = np.maximum(accum, K_t)
                path.append(int(np.argmax(K_t)))

                rho0 = float(K_t.sum())
                p = K_t / max(rho0, 1e-12)
                w_num = K_t * self._m
                phi = float(w_num.sum())
                w = w_num / max(phi, 1e-12)

                c_mass = w @ self._V
                if direction == "uphill":
                    g = c_mass
                elif direction == "downhill":
                    c_sem = p @ self._V
                    g = -(c_mass - c_sem)
                else:
                    raise ValueError("direction must be 'uphill' or 'downhill'")

                # Project onto tangent plane of S^(D-1) for spherical step
                g_tan = g - float(g @ qt) * qt
                qt = _normalize_rows(qt + step_eta * g_tan)
            paths.append(path)

        m_bar = self.surface_state().m_bar
        top_ids = np.argsort(-accum)[: min(k, self.n)]
        hits = [
            TrajectoryHit(
                card_id=int(i),
                question=self.cards[int(i)].question,
                answer=self.cards[int(i)].answer,
                score=float(accum[i]),
                mass=float(self._m[i]),
                smoothed_mass=float(m_bar[i]),
            )
            for i in top_ids
        ]
        return TrajectoryResult(
            query=q_str,
            direction=direction,
            hits=hits,
            branch_seeds=seeds,
            paths=paths,
        )

    # ------------------------------------------------------------------ regional diagnostics & mutation
    def region_diagnostics(self) -> Dict[str, np.ndarray]:
        """Compare each card's mass m_i against its leave-one-out neighborhood m_bar_neighbors.

        Returns:
            m: Raw card masses (N,)
            m_bar_neighbors: Smoothed mastery of surrounding cards excluding self (N,)
            spike_residual: m_i - m_bar_neighbors (positive = isolated spike, ~0 = uniform plateau)
            neighbor_density: Effective surrounding concept density rho_0 - 1 (N,)
        """
        if self.n <= 1:
            return {
                "m": self._m.copy(),
                "m_bar_neighbors": self._m.copy(),
                "spike_residual": np.zeros(self.n, dtype=np.float64),
                "neighbor_density": np.zeros(self.n, dtype=np.float64),
            }

        K = self._kernel(self._V).copy()                 # (N, N), diagonal is 1.0
        np.fill_diagonal(K, 0.0)                         # Exclude self (leave-one-out)

        rho_neighbors = K.sum(axis=1)                    # Surrounding concept density
        phi_neighbors = K @ self._m                      # Surrounding mass potential

        m_bar_neighbors = phi_neighbors / np.maximum(rho_neighbors, 1e-12)
        spike_residual = self._m - m_bar_neighbors

        return {
            "m": self._m.copy(),
            "m_bar_neighbors": m_bar_neighbors,
            "spike_residual": spike_residual,
            "neighbor_density": rho_neighbors,
        }

    @staticmethod
    def _classify_geometry(m_i: float, m_neigh: float, delta_i: float) -> Tuple[str, str, str]:
        """Map a card's geometric signature to a state, action and reason."""
        # Rule 1: High card mass, but weak surrounding region -> ISOLATED SPIKE
        if m_i >= 1.25 and delta_i >= 0.30:
            return (
                "isolated_spike",
                "replace",
                "Student has memorized this specific card, but adjacent neighbor concepts remain weak. Needs in-place replacement.",
            )
        # Rule 2: High card mass AND high surrounding region -> CONSOLIDATED PLATEAU
        if m_i >= 1.15 and m_neigh >= 1.05 and abs(delta_i) <= 0.35:
            return (
                "consolidated_plateau",
                "expand",
                "Entire local cluster is mastered. Ready for outward frontier expansion with new concepts.",
            )
        # Rule 3: Low card mass in a strong neighborhood -> ISOLATED POTHOLE
        if m_i <= 0.75 and m_neigh >= 1.05 and delta_i <= -0.30:
            return (
                "isolated_pothole",
                "reframe",
                "Student understands the surrounding domain but repeatedly fails this card. Needs reframing.",
            )
        return (
            "unmastered",
            "review",
            "Region is still developing. Normal review recommended.",
        )

    def diagnose_card(self, card_id: int) -> Dict[str, Any]:
        """Classify a card into a geometric state: isolated_spike, consolidated_plateau, isolated_pothole, or unmastered."""
        if not 0 <= card_id < self.n:
            raise IndexError(f"Invalid card_id {card_id}")

        diag = self.region_diagnostics()
        m_i = float(diag["m"][card_id])
        m_neigh = float(diag["m_bar_neighbors"][card_id])
        delta_i = float(diag["spike_residual"][card_id])
        density = float(diag["neighbor_density"][card_id])
        state, action, reason = self._classify_geometry(m_i, m_neigh, delta_i)

        return {
            "card_id": card_id,
            "state": state,
            "recommended_action": action,
            "reason": reason,
            "mass": round(m_i, 4),
            "neighbor_mastery": round(m_neigh, 4),
            "spike_residual": round(delta_i, 4),
            "neighbor_density": round(density, 4),
        }

    def diagnose_all(self) -> List[Dict[str, Any]]:
        """Scan the entire deck in one pass and return every card's geometric signature.

        This is the deck-wide view of the manifold: it computes leave-one-out
        neighbourhood mastery once, then classifies each card so the caller can
        surface the regions that actually deserve mutation / expansion / reframing.
        """
        if self.n == 0:
            return []
        diag = self.region_diagnostics()
        results: List[Dict[str, Any]] = []
        for i in range(self.n):
            m_i = float(diag["m"][i])
            m_neigh = float(diag["m_bar_neighbors"][i])
            delta_i = float(diag["spike_residual"][i])
            density = float(diag["neighbor_density"][i])
            state, action, reason = self._classify_geometry(m_i, m_neigh, delta_i)
            results.append(
                {
                    "card_id": i,
                    "question": self.cards[i].question,
                    "topic": self.cards[i].topic,
                    "state": state,
                    "recommended_action": action,
                    "reason": reason,
                    "mass": round(m_i, 4),
                    "neighbor_mastery": round(m_neigh, 4),
                    "spike_residual": round(delta_i, 4),
                    "neighbor_density": round(density, 4),
                }
            )
        return results

    def get_mental_state(self, top_k: int = 4) -> Dict[str, Any]:
        """Extract continuous, ranked Mental State snapshot for Socratic AI tutoring.

        Returns top ranked concept nodes across four continuous topological roles:
        - valleys: lowest smoothed mastery m_bar (primary learning gaps)
        - anchors: highest smoothed mastery m_bar (mastered knowledge to scaffold on)
        - frontier: highest gradient norms ||nabla log m_bar|| (active learning boundaries)
        - spikes: highest positive residuals Delta_i = m_i - m_bar_neighbors (phrasing memorization)
        """
        if self.n == 0:
            return {
                "valleys": [],
                "anchors": [],
                "frontier": [],
                "spikes": [],
                "deck_summary": {"total_cards": 0, "total_mass": 0.0, "step": self.step},
            }

        k = min(int(top_k), self.n)
        snap = self.surface_state()
        diag = self.region_diagnostics()

        def _card_dict(idx: int, **kwargs) -> Dict[str, Any]:
            c = self.cards[idx]
            d = {
                "card_id": idx,
                "question": c.question,
                "answer": c.answer,
                "topic": c.topic,
                "mass": round(float(self._m[idx]), 4),
                "smoothed_mastery": round(float(snap.m_bar[idx]), 4),
            }
            d.update(kwargs)
            return d

        # 1. Valleys: lowest smoothed mastery m_bar
        valley_order = np.argsort(snap.m_bar)
        valleys = [_card_dict(int(i)) for i in valley_order[:k]]

        # 2. Anchors: highest smoothed mastery m_bar
        anchor_order = np.argsort(-snap.m_bar)
        anchors = [_card_dict(int(i)) for i in anchor_order[:k]]

        # 3. Frontier boundaries: highest gradient norms
        frontier_order = np.argsort(-snap.grad_norm)
        frontier = [
            _card_dict(int(i), grad_norm=round(float(snap.grad_norm[i]), 4))
            for i in frontier_order[:k]
        ]

        # 4. Spikes: highest positive residual discrepancy Delta_i
        spike_order = np.argsort(-diag["spike_residual"])
        spikes = [
            _card_dict(
                int(i),
                spike_residual=round(float(diag["spike_residual"][i]), 4),
                neighbor_mastery=round(float(diag["m_bar_neighbors"][i]), 4),
            )
            for i in spike_order[:k]
            if float(diag["spike_residual"][i]) > 0.0
        ]

        return {
            "valleys": valleys,
            "anchors": anchors,
            "frontier": frontier,
            "spikes": spikes,
            "deck_summary": {
                "total_cards": self.n,
                "total_mass": round(float(self._m.sum()), 6),
                "variance": round(float(snap.variance), 6),
                "entropy": round(float(snap.entropy), 4),
                "step": self.step,
            },
        }

    def batch_review(self, scores: Dict[Union[int, str], float]) -> Dict[str, Any]:
        """Apply evaluations for multiple concepts touched in a Socratic session and conserve mass."""
        if not scores:
            return {"reviews_applied": 0, "total_mass": float(self._m.sum()), "deltas": {}}

        masses_before = self._m.copy()
        applied = 0
        deltas = {}

        for k, score in scores.items():
            cid = int(k)
            if 0 <= cid < self.n:
                y = float(score)
                self.review(cid, y)
                applied += 1

        for k, score in scores.items():
            cid = int(k)
            if 0 <= cid < self.n:
                deltas[str(cid)] = {
                    "question": self.cards[cid].question,
                    "topic": self.cards[cid].topic,
                    "score": float(score),
                    "mass_before": round(float(masses_before[cid]), 4),
                    "mass_after": round(float(self._m[cid]), 4),
                    "delta": round(float(self._m[cid] - masses_before[cid]), 4),
                }

        return {
            "reviews_applied": applied,
            "step": self.step,
            "total_mass": round(float(self._m.sum()), 6),
            "deltas": deltas,
        }

    def get_card_cluster(self, card_id: int, k: int = 5) -> Dict[str, Any]:
        """Retrieve target card and its k nearest semantic neighbors."""
        if not 0 <= card_id < self.n:
            raise IndexError(f"Invalid card_id {card_id}")

        v = self._V[card_id]
        sims = self._V @ v
        indices = np.argsort(-sims)
        neighbor_ids = [int(idx) for idx in indices if idx != card_id][:k]

        target = {
            "card_id": card_id,
            "question": self.cards[card_id].question,
            "answer": self.cards[card_id].answer,
            "topic": self.cards[card_id].topic,
            "mass": float(self._m[card_id]),
            "generation": int(self.cards[card_id].meta.get("generation", 0)),
        }

        neighbors = []
        for nid in neighbor_ids:
            neighbors.append({
                "card_id": nid,
                "question": self.cards[nid].question,
                "answer": self.cards[nid].answer,
                "topic": self.cards[nid].topic,
                "mass": float(self._m[nid]),
                "similarity": float(sims[nid]),
            })

        return {
            "target": target,
            "diagnostics": self.diagnose_card(card_id),
            "neighbors": neighbors,
        }

    def replace_card(
        self,
        card_id: int,
        new_question: str,
        new_answer: str,
        new_topic: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """In-place mutation of an isolated-spike card into a deeper probe."""
        if not 0 <= card_id < self.n:
            raise IndexError(f"Invalid card_id {card_id}")

        diag = self.region_diagnostics()
        target_mass = float(diag["m_bar_neighbors"][card_id])
        old_mass = float(self._m[card_id])

        card = self.cards[card_id]
        topic = new_topic if new_topic is not None else card.topic

        generation = int(card.meta.get("generation", 0)) + 1
        history = list(card.meta.get("history", []))
        history.append({
            "generation": generation - 1,
            "question": card.question,
            "answer": card.answer,
            "mass_before_replacement": old_mass,
        })
        card.meta["generation"] = generation
        card.meta["history"] = history
        card.meta["prev_question"] = card.question

        card.question = new_question
        card.answer = new_answer
        card.topic = topic
        card.reviews = 0
        card.correct_sum = 0.0

        if vector is None:
            text = self.format_card(new_question, new_answer, topic)
            new_vec = _normalize_rows(self.emb.encode([text])[0])
        else:
            new_vec = _normalize_rows(np.asarray(vector, dtype=np.float64))

        self._V[card_id] = new_vec

        # Relax mass from isolated spike down to neighborhood reality, then conserve
        self._m[card_id] = max(target_mass, 0.5)
        self._conserve()

        return {
            "card_id": card_id,
            "generation": generation,
            "old_question": card.meta["prev_question"],
            "new_question": new_question,
            "mass_before": old_mass,
            "mass_after": float(self._m[card_id]),
            "total_mass": float(self._m.sum()),
        }

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        meta = {
            "cfg": asdict(self.cfg),
            "step": self.step,
            "cards": [asdict(c) for c in self.cards],
        }
        np.savez_compressed(
            path,
            V=self._V,
            m=self._m,
            meta=np.array(json.dumps(meta)),
        )

    @classmethod
    def load(cls, path: str, embedder: Optional[Embedder] = None) -> "MassFieldEngine":
        if not path.endswith(".npz"):
            path += ".npz"
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        cfg = FieldConfig(**meta["cfg"])
        eng = cls(embedder=embedder, config=cfg)
        V = np.asarray(data["V"], dtype=np.float64)
        if V.shape[0] > 0 and V.shape[1] != eng.dim:
            raise ValueError(f"Embedder dim {eng.dim} != stored dim {V.shape[1]}")
        eng._V = V if V.size else np.zeros((0, eng.dim), dtype=np.float64)
        eng._m = np.asarray(data["m"], dtype=np.float64)
        eng.step = int(meta["step"])
        eng.cards = [Card(**c) for c in meta["cards"]]
        return eng
