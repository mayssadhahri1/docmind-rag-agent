"""Configuration de l'application, lue depuis les variables d'environnement."""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # Adresse du serveur Ollama (LLM local)
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    # Modèle de chat utilisé par l'agent
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "llama3.2"))
    # Modèle d'embeddings pour la recherche sémantique
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text"))
    # Dossier où l'index vectoriel est sauvegardé
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    # Découpage des documents : taille d'un morceau et chevauchement (en caractères)
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))
    # Nombre de passages renvoyés par une recherche
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    # Nombre maximum d'étapes de raisonnement de l'agent
    max_steps: int = field(default_factory=lambda: int(os.getenv("MAX_STEPS", "5")))


settings = Settings()
