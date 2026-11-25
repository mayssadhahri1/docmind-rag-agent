import json

from app.agent import Agent, calculate, parse_json
from app.ingest import Chunk
from app.llm import ScriptedLLM
from app.vectorstore import VectorStore


def make_agent(replies):
    llm = ScriptedLLM([json.dumps(r) for r in replies])
    store = VectorStore()
    text = "Les congés payés sont de 30 jours ouvrables par an."
    store.add([Chunk(text, "reglement.pdf", 4)], llm.embed([text]))
    return Agent(llm, store, top_k=2, max_steps=4), llm


def test_agent_utilise_la_recherche_puis_repond():
    agent, llm = make_agent([
        {"thought": "Je cherche dans le règlement", "action": "search_documents", "input": "congés payés"},
        {"thought": "J'ai trouvé", "final_answer": "30 jours ouvrables par an.", "sources": ["reglement.pdf p.4"]},
    ])
    result = agent.run("Combien de jours de congés ?")
    assert result.answer == "30 jours ouvrables par an."
    assert result.sources == ["reglement.pdf p.4"]
    assert result.steps[0].action == "search_documents"
    assert "reglement.pdf p.4" in result.steps[0].observation
    # le résultat de l'outil a bien été renvoyé au modèle
    assert "Résultat de l'outil" in llm.calls[1][-1]["content"]


def test_agent_redemande_si_json_invalide():
    agent, _ = make_agent([])
    agent.llm.replies = ["je ne sais pas", json.dumps({"thought": "ok", "final_answer": "Réponse"})]
    assert agent.run("Question ?").answer == "Réponse"


def test_agent_outil_inconnu():
    agent, _ = make_agent([
        {"thought": "...", "action": "internet", "input": "x"},
        {"thought": "...", "final_answer": "fin"},
    ])
    result = agent.run("Question ?")
    assert "Outil inconnu" in result.steps[0].observation


def test_agent_s_arrete_apres_max_steps():
    step = {"thought": "encore", "action": "list_documents", "input": ""}
    agent, _ = make_agent([step] * 4)
    result = agent.run("Question ?")
    assert len(result.steps) == 4
    assert "pas réussi" in result.answer


def test_calculatrice():
    assert calculate("(12,5 * 4) / 100") == "0.5"
    assert calculate("2^10") == "1024"
    assert calculate("__import__('os')").startswith("Erreur")


def test_parse_json_avec_texte_autour():
    assert parse_json('Voici : {"a": 1} merci') == {"a": 1}
    assert parse_json("pas de json") is None
