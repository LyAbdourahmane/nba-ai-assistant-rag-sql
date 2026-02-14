import sys
import os

# ================================================================
# Path & environnement

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import pandas as pd
from datasets import Dataset
from app.router import route_question
from evaluation.mistral_ragas_embeddings import MistralRagasEmbeddings
from langchain_mistralai import ChatMistralAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from utils.config import MISTRAL_API_KEY

# Chargons les questions métier
with open("data/test_questions.json", "r", encoding="utf-8") as f:
    test_questions = json.load(f)

ragas_llm = ChatMistralAI(
    mistral_api_key=MISTRAL_API_KEY,
    model="mistral-tiny-latest",
    temperature=0.1,
)

ragas_embeddings = MistralRagasEmbeddings()

samples = []

for item in test_questions:
    q = item["question"]
    result = route_question(q)

    if isinstance(result, str):
        answer = result
        contexts = []
    else:
        answer = result.get("answer", "")
        contexts = result.get("contexts", [])

    samples.append({
        "question": q,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ""
    })

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
df_scores.to_csv("data/Apres_sqltool/ragas_results_metier.csv", index=False)

print("Évaluation métier terminée.")
