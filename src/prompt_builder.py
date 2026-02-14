from app.prompts import get_rag_system_prompt

def build_rag_prompt(context: str, question: str) -> str:
    """
    Construit le prompt final utilisé par le moteur RAG.
    Le prompt système est récupéré depuis app/prompts.py
    et formaté avec le contexte et la question.
    """
    system_prompt = get_rag_system_prompt()

    final_prompt = system_prompt.format(
        context=context,
        question=question
    )

    return final_prompt
