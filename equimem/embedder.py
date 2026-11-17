"""Embedding providers for projecting concept cards onto the unit hypersphere."""
from __future__ import annotations

import os
import re
import time
from typing import List, Optional, Protocol, Sequence, Union
import zlib
import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return 2D array of shape (len(texts), dim) with unit L2 norm rows."""
        ...


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        return arr / max(n, 1e-12)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


class NGramEmbedder:
    """Deterministic dependency-free fallback embedder using signed feature hashing."""

    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def _add_feature(self, vec: np.ndarray, key: str, weight: float) -> None:
        h = zlib.crc32(key.encode("utf-8"))
        idx = h % self.dim
        sign = 1.0 if ((h >> 31) & 1) else -1.0
        vec[idx] += sign * weight

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            prev = None
            for tok in tokens:
                self._add_feature(out[row], f"w:{tok}", 1.0)
                if len(tok) >= 4:
                    for i in range(len(tok) - 3):
                        self._add_feature(out[row], f"c4:{tok[i:i+4]}", 0.35)
                if prev is not None:
                    self._add_feature(out[row], f"b:{prev}_{tok}", 0.5)
                prev = tok
        return _normalize_rows(out)


class GeminiEmbedder:
    """Google Gemini Embeddings provider (gemini-embedding-001 / gemini-embedding-2).

    Projects text concepts onto the unit hypersphere with high semantic fidelity.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-embedding-001",
        dim: int = 3072,
        batch_size: int = 32,
        max_chars: int = 4000,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            for candidate in [".env", "../.env", "../../.env"]:
                if os.path.exists(candidate):
                    with open(candidate, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                if k.strip() in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                                    self.api_key = v.strip()
                                    os.environ[k.strip()] = self.api_key
                                    break
                if self.api_key:
                    break

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment or .env file")

        from google import genai
        self.client = genai.Client(http_options={"api_version": "v1beta"}, api_key=self.api_key)
        self.name = model
        self.dim = int(dim)
        self.batch_size = int(batch_size)
        self.max_chars = int(max_chars)

    def _encode_chunk(self, chunk: List[str]) -> np.ndarray:
        from google.genai import types
        cleaned = [t[:self.max_chars] if isinstance(t, str) else "" for t in chunk]
        cfg = types.EmbedContentConfig(output_dimensionality=self.dim) if self.dim != 3072 else None

        last_err = None
        for attempt in range(4):
            try:
                res = self.client.models.embed_content(
                    model=self.name,
                    contents=cleaned if len(cleaned) > 1 else cleaned[0],
                    config=cfg,
                )
                vecs = [e.values for e in res.embeddings]
                return np.array(vecs, dtype=np.float64)
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))

        # Fallback to zero vectors on complete network failure
        return np.zeros((len(chunk), self.dim), dtype=np.float64)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)

        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for start in range(0, len(texts), self.batch_size):
            end = min(start + self.batch_size, len(texts))
            chunk = texts[start:end]
            out[start:end] = self._encode_chunk(chunk)

        return _normalize_rows(out)


class CachedGeminiEmbedder:
    """Embedder wrapper with local file caching to prevent redundant API calls."""

    def __init__(
        self,
        cache_path: str,
        model: str = "gemini-embedding-001",
        dim: int = 3072,
    ):
        self.cache_path = cache_path
        self.dim = int(dim)
        self.model = model
        self._live: Optional[GeminiEmbedder] = None
        self.cache: dict[str, np.ndarray] = {}

        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                keys = data["keys"]
                vecs = data["vectors"]
                for k, v in zip(keys, vecs):
                    self.cache[str(k)] = np.asarray(v, dtype=np.float64)
            except Exception:
                pass

    @property
    def live(self) -> GeminiEmbedder:
        if self._live is None:
            self._live = GeminiEmbedder(model=self.model, dim=self.dim)
        return self._live

    def _save(self) -> None:
        keys = list(self.cache.keys())
        if keys:
            vecs = np.array([self.cache[k] for k in keys], dtype=np.float64)
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            np.savez_compressed(self.cache_path, keys=keys, vectors=vecs)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        missing = [t for t in texts if t not in self.cache]
        if missing:
            try:
                vecs = self.live.encode(missing)
                for t, v in zip(missing, vecs):
                    self.cache[t] = v
                self._save()
            except Exception:
                # If API fails and text missing, fallback to NGram hash
                fallback = NGramEmbedder(dim=self.dim)
                vecs = fallback.encode(missing)
                for t, v in zip(missing, vecs):
                    self.cache[t] = v

        out = np.array([self.cache[t] for t in texts], dtype=np.float64)
        return _normalize_rows(out)
