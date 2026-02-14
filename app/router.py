import logging
from app.prompts import get_classification_prompt
from app.mistral_client import mistral_chat
from src.sql_tool import nl_2_sql
from src.rag_engine import rag_answer
from src.validation_pydantic import RAGResponse, SQLResponse

import logfire

logfire.configure()


@logfire.instrument()
def classify_question(question: str) -> str:
    """
    Utilise le LLM pour déterminer si la question doit être traitée
    par le SQL Tool ou par le moteur RAG.
    Retourne 'SQL' ou 'RAG'.
    """
    system_prompt = get_classification_prompt()

    response = mistral_chat(
        system_prompt=system_prompt,
        user_message=question,
        temperature=0
    )

    if not response:
        logging.warning("[ROUTER] Réponse vide du classificateur, fallback RAG.")
        return "RAG"

    mode = response.strip().upper()

    if mode not in ("SQL", "RAG"):
        logging.warning(f"[ROUTER] Réponse inattendue : {mode}. Fallback RAG.")
        return "RAG"

    logging.info(f"[ROUTER] Mode détecté : {mode}")
    logfire.info("Classification", question=question, mode=mode)
    return mode


@logfire.instrument()
def route_question(question: str) -> dict:
    """
    Route la question vers le bon moteur :
    - SQL → nl_2_sql
    - RAG → rag_answer
    """
    logfire.info("Routing question", question=question)
    mode = classify_question(question)

    try:
        if mode == "SQL":
            logfire.info("Routing to SQL Tool")
            return nl_2_sql(question)

        logfire.info("Routing to RAG Engine")
        return rag_answer(question)

    except Exception as e: 
        logging.error(f"[ROUTER] Erreur {mode} : {e}") 
        logfire.error("Routing error", error=str(e))
        return RAGResponse( answer="Une erreur est survenue lors du traitement.", contexts=[] ).model_dump()
