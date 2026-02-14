from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# -----------------------------
# Réponse du moteur RAG
class RAGResponse(BaseModel):
    answer: str
    contexts: List[str]


# -----------------------------
# Réponse du SQL Tool
class SQLResponse(BaseModel):
    answer: str
    contexts: List[str] = []
