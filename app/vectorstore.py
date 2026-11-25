"""Index vectoriel simple basé sur NumPy (similarité cosinus), sauvegardé sur disque."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from .ingest import Chunk


class VectorStore:
    def __init__(self, path: str | None = None):
        self.path = path
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        if path and os.path.exists(os.path.join(path, "index.npy")):
            self.load()

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        v = _normalize(np.asarray(vectors, dtype=np.float32))
        self.matrix = v if len(self) == 0 else np.vstack([self.matrix, v])
        self.chunks.extend(chunks)
        self.save()

    def search(self, query_vector: list[float], k: int = 4) -> list[tuple[Chunk, float]]:
        """Renvoie les k passages les plus proches de la question, avec leur score."""
        if len(self) == 0:
            return []
        q = _normalize(np.asarray([query_vector], dtype=np.float32))[0]
        scores = self.matrix @ q
        best = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in best]

    def sources(self) -> dict[str, int]:
        """Nombre de passages indexés par document."""
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.source] = out.get(c.source, 0) + 1
        return out

    def remove(self, source: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        self.chunks = [self.chunks[i] for i in keep]
        self.matrix = self.matrix[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        self.save()
        return removed

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(self.path, exist_ok=True)
        np.save(os.path.join(self.path, "index.npy"), self.matrix)
        with open(os.path.join(self.path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)

    def load(self) -> None:
        self.matrix = np.load(os.path.join(self.path, "index.npy"))
        with open(os.path.join(self.path, "chunks.json"), encoding="utf-8") as f:
            self.chunks = [Chunk(**c) for c in json.load(f)]


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
