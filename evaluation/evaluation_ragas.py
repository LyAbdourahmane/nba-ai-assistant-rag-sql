import sys
import os

# ================================================================
# Path & environnement

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import logging
import json
import pandas as pd
from datasets import Dataset

from app.mistral_client import mistral_chat
from app.router import route_question
from utils.vector_store import VectorStoreManager
from utils.config import MISTRAL_API_KEY
from evaluation.mistral_ragas_embeddings import MistralRagasEmbeddings

from ragas import evaluate
from langchain_mistralai import ChatMistralAI
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

logging.basicConfig(level=logging.INFO)

# ================================================================
# Vector store

vector_store = VectorStoreManager()
chunks = vector_store.chunks
logging.info(f"{len(chunks)} chunks chargés pour l'évaluation.")

ragas_llm = ChatMistralAI(
    mistral_api_key=MISTRAL_API_KEY,
    model="mistral-tiny-latest",
    temperature=0.1,
)
ragas_embeddings = MistralRagasEmbeddings()

# ================================================================
# Sampling de context consécutif

def sample_context(n=3):
    if len(chunks) < n:
        raise ValueError("Pas assez de chunks.")
    start = random.randint(0, len(chunks) - n)
    selected = chunks[start : start + n]
    merged_context = "\n\n---\n\n".join([c["text"] for c in selected])
    return merged_context

# ================================================================
# Parsing JSON robuste

def safe_json_extract(text: str):
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None

# ================================================================
# Génération question + ground truth

def generate_question_and_gt(context: str):
    prompt = f"""
Tu es un générateur de données pour l'évaluation d'un système RAG.

Voici un extrait de document :

{context}

Ta tâche :
1. Générer UNE question basée uniquement sur cet extrait.
2. Générer la réponse exacte basée uniquement sur cet extrait.

Tu dois répondre STRICTEMENT en JSON valide.

Format EXACT :

{{
  "question": "...",
  "ground_truth": "..."
}}
"""

    raw = mistral_chat(
        system_prompt=prompt,
        user_message="",
        temperature=0.1,
    )

    data = safe_json_extract(raw)
    if not data:
        logging.warning("JSON invalide, sample ignoré.")
        return None, None

    q = data.get("question")
    gt = data.get("ground_truth")

    if not q or not gt:
        logging.warning("JSON incomplet, sample ignoré.")
        return None, None

    return q.strip(), gt.strip()

# ================================================================
# Interrogation du système RAG

def ask_system(question: str):
    result = route_question(question)

    if isinstance(result, str):
        return result, []

    answer = result.get("answer", "")
    contexts = result.get("contexts", [])

    if contexts and isinstance(contexts[0], dict):
        contexts = [c.get("text", "") for c in contexts]

    return answer, contexts

# ================================================================
# Construction des samples + sauvegarde des contexts de génération

def build_samples(n_samples: int = 20):
    ragas_samples = []
    generation_contexts = []

    for i in range(n_samples):
        generation_context = sample_context()
        q, gt = generate_question_and_gt(generation_context)

        if not q or not gt:
            continue

        answer, retrieved_contexts = ask_system(q)

        ragas_samples.append(
            {
                "question": q,
                "answer": answer,
                "contexts": retrieved_contexts,
                "ground_truth": gt,
            }
        )

        generation_contexts.append(
            {
                "generation_context": generation_context
            }
        )

        logging.info(f"[OK] {i+1}/{n_samples} – {q}")

    return ragas_samples, generation_contexts

# ================================================================
# Évaluation RAGAS + exports CSV

def run_evaluation(n_samples: int = 5):
    samples, generation_contexts = build_samples(n_samples)

    if not samples:
        logging.error("Aucun sample généré.")
        return

    # --- Dataset RAGAS ---
    df = pd.DataFrame(samples)
    hf_dataset = Dataset.from_pandas(df)

    result = evaluate(
        hf_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        embeddings=ragas_embeddings,
        llm=ragas_llm,
        raise_exceptions=False,
    )

    df_scores = result.to_pandas()
    df_scores.to_csv("data/Apres_sqltool/ragas_results.csv", index=False)

    # --- CSV des contexts utilisés pour la génération ---
    df_gen_ctx = pd.DataFrame(generation_contexts)
    df_gen_ctx.to_csv("data/Apres_sqltool/generation_contexts.csv", index=False)

    print("Évaluation terminée.")
    print("data/Apres_sqltool/ragas_results.csv")
    print("data/Apres_sqltool/generation_contexts.csv (1 colonne : generation_context)")



if __name__ == "__main__":
    run_evaluation(n_samples=5)
