from utils.vector_store import VectorStoreManager
from src.prompt_builder import build_rag_prompt
from app.mistral_client import mistral_chat
from utils.config import SEARCH_K
from src.validation_pydantic import RAGResponse

import logfire

_vector_store = VectorStoreManager()
logfire.configure()

@logfire.instrument()
def rag_answer(question: str) -> dict:
    """
    Pipeline RAG complet :
    1. Recherche vectorielle (FAISS)
    2. Construction du contexte
    3. Construction du prompt final via prompt_builder
    4. Appel au LLM
    5. Retourne :
    {
        "answer": str,
        "contexts": [list de chunks utilisés]
    }
    """
    logfire.info("RAG start", question=question)

    # Recherche FAISS
    results = _vector_store.search(question, k=SEARCH_K)
    logfire.info("FAISS results", count=len(results))

    if not results:
        return RAGResponse( 
            answer="Je n’ai trouvé aucune information pertinente dans les documents.", 
            contexts=[] 
            ).model_dump()

    # Construction du contexte
    context_chunks = [r["text"] for r in results]
    context_str = "\n\n---\n\n".join(context_chunks)
    logfire.info("Context chunks", chunks=context_chunks)

    # Prompt final
    system_prompt = build_rag_prompt(context_str, question)

    # Appel Mistral
    answer = mistral_chat(
        system_prompt=system_prompt,
        user_message="",
        temperature=0.1
    )

    logfire.info("LLM answer", answer=answer)
    return RAGResponse( answer=answer, contexts=context_chunks ).model_dump()




