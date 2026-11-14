"""Card Mutator and Knowledge Frontier Expander powered by Gemini Flash models.

Implements in-place card replacement for isolated spikes, outward additions
for consolidated plateaus, and reframing for isolated potholes.

All evolution operations are strictly manual and on-demand.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from google import genai

from equimem.engine import MassFieldEngine

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _clean_json_markdown(text: str) -> str:
    """Strip markdown code fence ```json ... ``` if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class CardMutator:
    """Evolves knowledge cards on demand using Google's Flash models via the official GenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.5-flash-lite",
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment or .env file")
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    def _call_model(self, prompt: str) -> str:
        """Call Gemini generate_content via the official Google GenAI SDK with fallback on 503."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text or ""
        except Exception as e:
            err_msg = str(e)
            if ("503" in err_msg or "UNAVAILABLE" in err_msg) and self.model != "gemini-3.5-flash-lite":
                response = self.client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                )
                return response.text or ""
            raise

    def mutate_spike_card(self, engine: MassFieldEngine, card_id: int) -> Dict[str, Any]:
        """Upgrade an isolated spike card in-place into a deeper probe connecting to weak neighbors."""
        cluster = engine.get_card_cluster(card_id, k=4)
        target = cluster["target"]
        neighbors = cluster["neighbors"]

        # Find weak neighbors (m < 1.0)
        weak_neighbors = [n for n in neighbors if n["mass"] < 1.0]
        context_neighbors = weak_neighbors if weak_neighbors else neighbors[:2]

        neighbor_text = "\n".join([
            f"- Related Concept: {n['question']}\n  Details: {n['answer']} (Mass: {n['mass']:.2f})"
            for n in context_neighbors
        ])

        prompt = f"""You are the Equimem Knowledge Manifold Mutator.
The student has memorized the target flashcard below (high recall score), but struggles with adjacent concepts in the same domain (low neighborhood mastery).

[TARGET CARD TO UPGRADE IN-PLACE]
- Question: {target['question']}
- Answer: {target['answer']}
- Domain: {target['topic']}

[WEAK ADJACENT NEIGHBORS]
{neighbor_text}

TASK:
Upgrade the Target Card IN-PLACE into a deeper, mechanism-level inquiry.
The new question must require understanding how the target concept connects with, governs, or interacts with the weak neighbor concepts. Do not make it a trivial definition; make it an applied or structural question.

Return ONLY a valid JSON object with exact keys:
{{
  "new_question": "string",
  "new_answer": "string",
  "explanation": "string explaining how this bridges to weak neighbors"
}}"""

        raw_output = self._call_model(prompt)
        cleaned = _clean_json_markdown(raw_output)
        data = json.loads(cleaned)

        new_q = data["new_question"].strip()
        new_a = data["new_answer"].strip()
        explanation = data.get("explanation", "")

        # Perform in-place card replacement in the conserved mass field
        result = engine.replace_card(
            card_id=card_id,
            new_question=new_q,
            new_answer=new_a,
            new_topic=target["topic"],
        )
        result["action"] = "mutated_in_place"
        result["explanation"] = explanation
        return result

    def expand_plateau_cluster(
        self,
        engine: MassFieldEngine,
        card_id: int,
        count: int = 2,
    ) -> Dict[str, Any]:
        """Spawn brand new concept cards outward from a thoroughly mastered plateau cluster."""
        cluster = engine.get_card_cluster(card_id, k=5)
        target = cluster["target"]
        neighbors = cluster["neighbors"]

        cluster_items = [f"- {target['question']} => {target['answer']}"]
        for n in neighbors:
            cluster_items.append(f"- {n['question']} => {n['answer']}")

        prompt = f"""You are the Equimem Knowledge Manifold Expander.
The student has thoroughly mastered the following conceptual cluster (all cards in this neighborhood have high mass):

[MASTERED CLUSTER DOMAIN: {target['topic']}]
{chr(10).join(cluster_items)}

TASK:
Generate {count} brand-new, advanced concept flashcards that push into the conceptual frontier directly adjacent to this cluster.
These cards must explore deeper mechanics, advanced theorems, or next-horizon applications that logically follow from this mastered base.

Return ONLY a valid JSON array of objects:
[
  {{
    "question": "string",
    "answer": "string",
    "topic": "{target['topic']}"
  }}
]"""

        raw_output = self._call_model(prompt)
        cleaned = _clean_json_markdown(raw_output)
        cards_data = json.loads(cleaned)

        tuples = [
            (c["question"].strip(), c["answer"].strip(), c.get("topic", target["topic"]), {"parent_card": card_id})
            for c in cards_data
        ]
        new_ids = engine.add_cards(tuples)

        return {
            "action": "expanded_outward",
            "parent_card_id": card_id,
            "new_card_ids": new_ids,
            "new_cards": cards_data,
            "deck_size": engine.n,
            "total_mass": float(engine.m.sum()),
        }

    def reframe_pothole_card(self, engine: MassFieldEngine, card_id: int) -> Dict[str, Any]:
        """Reframe a card that the student repeatedly misses despite knowing surrounding concepts."""
        cluster = engine.get_card_cluster(card_id, k=4)
        target = cluster["target"]
        neighbors = cluster["neighbors"]

        mastered_neighbors = [n for n in neighbors if n["mass"] >= 1.0]
        context_neighbors = mastered_neighbors if mastered_neighbors else neighbors[:2]

        neighbor_text = "\n".join([
            f"- Mastered Anchor: {n['question']} => {n['answer']}"
            for n in context_neighbors
        ])

        prompt = f"""You are the Equimem Knowledge Manifold Mutator.
The student understands the surrounding domain well, but repeatedly misses this specific card (isolated pothole). The phrasing may be ambiguous or poorly anchored.

[FAILING CARD TO REFRAME]
- Question: {target['question']}
- Answer: {target['answer']}
- Domain: {target['topic']}

[SURROUNDING MASTERED ANCHORS]
{neighbor_text}

TASK:
Reframe the question and answer so it explicitly anchors to the concepts the student already knows. Eliminate ambiguity and make the prompt crystal clear.

Return ONLY a valid JSON object:
{{
  "new_question": "string",
  "new_answer": "string",
  "explanation": "string"
}}"""

        raw_output = self._call_model(prompt)
        cleaned = _clean_json_markdown(raw_output)
        data = json.loads(cleaned)

        new_q = data["new_question"].strip()
        new_a = data["new_answer"].strip()

        result = engine.replace_card(
            card_id=card_id,
            new_question=new_q,
            new_answer=new_a,
            new_topic=target["topic"],
        )
        result["action"] = "reframed_pothole"
        result["explanation"] = data.get("explanation", "")
        return result

    def evolve_card(self, engine: MassFieldEngine, card_id: int) -> Dict[str, Any]:
        """Diagnose a card and perform the appropriate on-demand evolution."""
        diag = engine.diagnose_card(card_id)
        action = diag["recommended_action"]

        if action == "replace":
            return self.mutate_spike_card(engine, card_id)
        elif action == "expand":
            return self.expand_plateau_cluster(engine, card_id, count=2)
        elif action == "reframe":
            return self.reframe_pothole_card(engine, card_id)
        else:
            return {
                "action": "none",
                "card_id": card_id,
                "diagnostics": diag,
                "message": "Card is in an unmastered or consolidating region. Normal review recommended before evolution.",
            }
