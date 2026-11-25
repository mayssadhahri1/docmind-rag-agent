"""API FastAPI de DocMind : import de documents, questions à l'agent, interface web."""
from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .agent import Agent
from .config import settings
from .ingest import SUPPORTED, chunk_document
from .llm import OllamaClient
from .vectorstore import VectorStore

STATIC = os.path.join(os.path.dirname(__file__), "static")


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = []


def create_app(llm=None, store: VectorStore | None = None) -> FastAPI:
    llm = llm or OllamaClient(settings.ollama_url, settings.chat_model, settings.embed_model)
    store = store if store is not None else VectorStore(settings.data_dir)
    agent = Agent(llm, store, settings.top_k, settings.max_steps)

    app = FastAPI(title="DocMind", description="Agent RAG qui répond aux questions à partir de vos documents.", version="1.0.0")

    @app.exception_handler(httpx.HTTPError)
    async def llm_unavailable(_: Request, exc: httpx.HTTPError):
        # Ollama arrêté ou modèle absent : message clair plutôt qu'une erreur 500
        return JSONResponse(status_code=503, content={"detail": f"LLM indisponible : {exc}. Vérifiez qu'Ollama est lancé."})

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/health")
    def health():
        up = llm.is_up() if hasattr(llm, "is_up") else True
        return {"status": "ok", "llm": "up" if up else "down", "chunks": len(store)}

    @app.get("/documents")
    def list_documents():
        return [{"name": n, "chunks": c} for n, c in store.sources().items()]

    @app.post("/documents", status_code=201)
    async def upload(file: UploadFile = File(...)):
        if not file.filename or not file.filename.lower().endswith(SUPPORTED):
            raise HTTPException(400, f"Formats acceptés : {', '.join(SUPPORTED)}")
        if file.filename in store.sources():
            store.remove(file.filename)  # ré-import : on remplace l'ancienne version
        chunks = chunk_document(file.filename, await file.read(), settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise HTTPException(422, "Aucun texte n'a pu être extrait de ce document")
        store.add(chunks, llm.embed([c.text for c in chunks]))
        return {"name": file.filename, "chunks": len(chunks)}

    @app.delete("/documents/{name}")
    def delete(name: str):
        if not store.remove(name):
            raise HTTPException(404, "Document introuvable")
        return {"deleted": name}

    @app.post("/ask")
    def ask(q: Question):
        result = agent.run(q.question, [m.model_dump() for m in q.history[-6:]])
        return {
            "answer": result.answer,
            "sources": result.sources,
            "steps": [s.__dict__ for s in result.steps],
        }

    return app


app = create_app()
