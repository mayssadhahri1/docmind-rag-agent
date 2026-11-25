import pytest

from app.ingest import chunk_document, read_document, split_text


def test_split_respecte_la_taille_et_couvre_tout_le_texte():
    text = " ".join(f"Phrase numéro {i}." for i in range(200))
    chunks = split_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].startswith("Phrase numéro 0.")
    assert chunks[-1].endswith("Phrase numéro 199.")


def test_split_coupe_en_fin_de_phrase():
    text = "A" * 120 + ". " + "B" * 200
    first = split_text(text, size=200, overlap=20)[0]
    assert first.endswith(".")


def test_texte_vide():
    assert split_text("   \n  ") == []


def test_chevauchement_invalide():
    with pytest.raises(ValueError):
        split_text("abc", size=100, overlap=100)


def test_format_non_supporte():
    with pytest.raises(ValueError):
        read_document("image.png", b"...")


def test_chunk_document_texte():
    chunks = chunk_document("notes.md", b"# Titre\n\nContenu du cours.", 100, 10)
    assert len(chunks) == 1
    assert chunks[0].source == "notes.md" and chunks[0].page == 1
