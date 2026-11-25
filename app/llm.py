"""Clients LLM : Ollama en production, un faux client déterministe pour les tests."""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx


class LLMClient(Protocol):
    def chat(self, messages: list[dict]) -> str: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaClient:
    """Appelle l'API HTTP d'Ollama (chat + embeddings)."""

    def __init__(self, base_url: str, chat_model: str, embed_model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.http = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict]) -> str:
        # format=json oblige le modèle à répondre avec un objet JSON valide
        r = self.http.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.chat_model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self.http.post(f"{self.base_url}/api/embed", json={"model": self.embed_model, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]

    def is_up(self) -> bool:
        try:
            return self.http.get(f"{self.base_url}/api/tags", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False


class HashingEmbedder:
    """Embeddings « sac de mots » hachés : sans modèle, utile pour les tests et le mode hors ligne."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = [0.0] * self.dim
            for word in _tokens(text):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class ScriptedLLM(HashingEmbedder):
    """Faux LLM : renvoie des réponses prévues à l'avance, dans l'ordre (tests de l'agent)."""

    def __init__(self, replies: list[str]):
        super().__init__()
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.replies.pop(0)


def _tokens(text: str) -> list[str]:
    word, out = [], []
    for ch in text.lower():
        if ch.isalnum():
            word.append(ch)
        elif word:
            out.append("".join(word))
            word = []
    if word:
        out.append("".join(word))
    return out
