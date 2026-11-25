from app.ingest import Chunk
from app.llm import HashingEmbedder
from app.vectorstore import VectorStore

DOCS = [
    "Le remboursement est possible sous 30 jours avec le ticket de caisse.",
    "La livraison standard prend entre 3 et 5 jours ouvrés.",
    "Le service client est joignable du lundi au vendredi de 9h à 18h.",
]


def build(path=None):
    emb = HashingEmbedder()
    store = VectorStore(path)
    chunks = [Chunk(t, "faq.txt", 1) for t in DOCS]
    store.add(chunks, emb.embed(DOCS))
    return emb, store


def test_recherche_trouve_le_bon_passage():
    emb, store = build()
    best, score = store.search(emb.embed(["combien de jours pour la livraison"])[0], k=1)[0]
    assert "livraison" in best.text
    assert 0 < score <= 1


def test_sauvegarde_et_rechargement(tmp_path):
    build(str(tmp_path))
    reloaded = VectorStore(str(tmp_path))
    assert len(reloaded) == 3
    assert reloaded.sources() == {"faq.txt": 3}


def test_suppression_document():
    _, store = build()
    assert store.remove("faq.txt") == 3
    assert len(store) == 0
    assert store.search([0.1] * 256) == []
