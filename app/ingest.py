"""Lecture des documents (PDF, TXT, Markdown) et découpage en passages."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

SUPPORTED = (".pdf", ".txt", ".md")


@dataclass
class Chunk:
    text: str
    source: str  # nom du fichier
    page: int    # numéro de page (1 pour les fichiers texte)


def read_document(filename: str, data: bytes) -> list[tuple[int, str]]:
    """Retourne la liste (page, texte) d'un document."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if name.endswith((".txt", ".md")):
        return [(1, data.decode("utf-8", errors="ignore"))]
    raise ValueError(f"Format non supporté : {filename} (formats acceptés : {', '.join(SUPPORTED)})")


def split_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Découpe un texte en morceaux d'environ `size` caractères avec un chevauchement.

    On coupe de préférence à la fin d'une phrase pour garder des passages lisibles.
    """
    if overlap >= size:
        raise ValueError("Le chevauchement doit être plus petit que la taille des morceaux")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # recule jusqu'à la dernière fin de phrase dans la seconde moitié du morceau
            cut = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(filename: str, data: bytes, size: int = 800, overlap: int = 150) -> list[Chunk]:
    chunks = []
    for page, text in read_document(filename, data):
        for part in split_text(text, size, overlap):
            chunks.append(Chunk(text=part, source=filename, page=page))
    return chunks
