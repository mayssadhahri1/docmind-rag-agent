"""Agent de type ReAct : le LLM raisonne, choisit un outil, lit le résultat, puis répond.

À chaque étape, le modèle répond en JSON :
  {"thought": "...", "action": "<outil>", "input": "..."}      -> on exécute l'outil
  {"thought": "...", "final_answer": "...", "sources": [...]}    -> réponse finale
"""
from __future__ import annotations

import ast
import json
import operator
from collections.abc import Callable
from dataclasses import dataclass, field

from .vectorstore import VectorStore

SYSTEM_PROMPT = """Tu es DocMind, un assistant qui répond aux questions à partir des documents de l'utilisateur.

Outils disponibles :
{tools}

Règles :
- Pour toute question sur le contenu des documents, utilise d'abord "search_documents".
- Pour un calcul, utilise "calculator" au lieu de calculer de tête.
- Réponds uniquement avec ce que disent les documents. Si l'information n'y est pas, dis-le clairement.
- Cite tes sources sous la forme "fichier p.X".
- Réponds dans la langue de la question.

Réponds TOUJOURS avec un seul objet JSON, sans texte autour :
- pour utiliser un outil : {{"thought": "ton raisonnement", "action": "nom_outil", "input": "argument"}}
- pour répondre : {{"thought": "ton raisonnement", "final_answer": "ta réponse", "sources": ["fichier p.X"]}}
"""


@dataclass
class Step:
    thought: str
    action: str | None = None
    input: str | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[str], str]


class Agent:
    def __init__(self, llm, store: VectorStore, top_k: int = 4, max_steps: int = 5):
        self.llm = llm
        self.store = store
        self.top_k = top_k
        self.max_steps = max_steps
        self.tools = {
            t.name: t
            for t in [
                Tool("search_documents", "recherche sémantique dans les documents. input = la question ou des mots-clés", self._search),
                Tool("list_documents", "liste les documents indexés. input = chaîne vide", self._list),
                Tool("calculator", "évalue une expression mathématique, ex. '(12.5 * 4) / 100'", calculate),
            ]
        }

    def run(self, question: str, history: list[dict] | None = None) -> AgentResult:
        tools_txt = "\n".join(f'- "{t.name}" : {t.description}' for t in self.tools.values())
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(tools=tools_txt)}]
        messages += history or []
        messages.append({"role": "user", "content": question})

        steps: list[Step] = []
        for _ in range(self.max_steps):
            raw = self.llm.chat(messages)
            data = parse_json(raw)
            if data is None:
                # le modèle n'a pas respecté le format : on lui redemande
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Réponds uniquement avec un objet JSON valide."})
                continue

            thought = str(data.get("thought", ""))
            if "final_answer" in data:
                steps.append(Step(thought=thought))
                return AgentResult(str(data["final_answer"]), [str(s) for s in data.get("sources", [])], steps)

            action, arg = str(data.get("action", "")), str(data.get("input", ""))
            tool = self.tools.get(action)
            observation = tool.run(arg) if tool else f"Outil inconnu : {action}. Outils possibles : {', '.join(self.tools)}"
            steps.append(Step(thought, action, arg, observation))
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Résultat de l'outil {action} :\n{observation}"})

        return AgentResult("Je n'ai pas réussi à trouver une réponse fiable en un nombre limité d'étapes.", [], steps)

    # ---- outils ----
    def _search(self, query: str) -> str:
        hits = self.store.search(self.llm.embed([query])[0], self.top_k)
        if not hits:
            return "Aucun document indexé."
        return "\n\n".join(f"[{c.source} p.{c.page}] (score {s:.2f})\n{c.text}" for c, s in hits)

    def _list(self, _: str) -> str:
        docs = self.store.sources()
        if not docs:
            return "Aucun document indexé."
        return "\n".join(f"- {name} ({n} passages)" for name, n in docs.items())


# ---- utilitaires ----
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def calculate(expression: str) -> str:
    """Calculatrice sûre : n'accepte que des nombres et des opérateurs (pas d'eval)."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("expression non autorisée")

    try:
        result = ev(ast.parse(expression.replace(",", ".").replace("^", "**"), mode="eval").body)
        return str(round(result, 6))
    except Exception as exc:  # noqa: BLE001
        return f"Erreur de calcul : {exc}"


def parse_json(raw: str) -> dict | None:
    """Extrait le premier objet JSON d'une réponse du modèle."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None
