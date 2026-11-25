import json

from fastapi.testclient import TestClient

from app.llm import ScriptedLLM
from app.main import create_app
from app.vectorstore import VectorStore


def client(replies=()):
    llm = ScriptedLLM([json.dumps(r) for r in replies])
    return TestClient(create_app(llm=llm, store=VectorStore()))


def test_health():
    r = client().get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_import_liste_et_suppression_de_document():
    c = client()
    r = c.post("/documents", files={"file": ("cours.txt", "Le design pattern Observer notifie ses abonnés.".encode(), "text/plain")})
    assert r.status_code == 201 and r.json()["chunks"] == 1
    assert c.get("/documents").json() == [{"name": "cours.txt", "chunks": 1}]
    assert c.delete("/documents/cours.txt").status_code == 200
    assert c.get("/documents").json() == []


def test_format_refuse():
    r = client().post("/documents", files={"file": ("photo.png", b"x", "image/png")})
    assert r.status_code == 400


def test_ask():
    c = client([{"thought": "simple", "final_answer": "Bonjour !"}])
    r = c.post("/ask", json={"question": "Salut"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Bonjour !"


def test_page_web():
    r = client().get("/")
    assert r.status_code == 200 and "DocMind" in r.text


def test_llm_indisponible_renvoie_503():
    from app.llm import OllamaClient

    c = TestClient(create_app(llm=OllamaClient("http://127.0.0.1:9", "m", "e", timeout=1), store=VectorStore()))
    r = c.post("/documents", files={"file": ("a.txt", b"Du texte.", "text/plain")})
    assert r.status_code == 503
    assert c.get("/health").json()["llm"] == "down"
