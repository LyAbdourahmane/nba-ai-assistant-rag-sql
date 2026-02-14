import streamlit as st
import pandas as pd
import ast

# -------------------------------------------------------------------
# Config de la page
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    layout="wide"
)

# -------------------------------------------------------------------
# Sélecteur de mode
mode = st.sidebar.radio(
    "Type d'évaluation",
    [
        "Évaluation automatique",
        "Évaluation métier",
        "Comparaison Avant / Après SQL Tool"
    ]
)

# -------------------------------------------------------------------
# Chargement des données
@st.cache_data
def load_auto():
    df = pd.read_csv("../data/Apres_sqltool/ragas_results.csv")
    ctx = pd.read_csv("../data/Apres_sqltool/generation_contexts.csv")

    min_len = min(len(df), len(ctx))
    df = df.iloc[:min_len].reset_index(drop=True)
    ctx = ctx.iloc[:min_len].reset_index(drop=True)

    df["generation_context"] = ctx["generation_context"]
    return df

@st.cache_data
def load_metier():
    df = pd.read_csv("../data/Apres_sqltool/ragas_results_metier.csv")
    df["generation_context"] = ""
    if "ground_truth" not in df.columns:
        df["ground_truth"] = ""
    return df

@st.cache_data
def load_avant():
    df = pd.read_csv("../data/Avant_sqltool/ragas_results.csv")
    ctx = pd.read_csv("../data/Avant_sqltool/generation_contexts.csv")
    df["generation_context"] = ctx["generation_context"]
    return df

@st.cache_data
def load_apres():
    df = pd.read_csv("../data/Apres_sqltool/ragas_results.csv")
    ctx = pd.read_csv("../data/Apres_sqltool/generation_contexts.csv")
    df["generation_context"] = ctx["generation_context"]
    return df

# -------------------------------------------------------------------
# Mode COMPARAISON AVANT / APRÈS SQL TOOL
if mode == "Comparaison Avant / Après SQL Tool":

    st.title("📊 Comparaison Avant / Après SQL Tool")

    df_avant = load_avant()
    df_apres = load_apres()

    st.header("Vue globale des scores")

    global_scores = pd.DataFrame({
        "Metric": ["faithfulness", "answer_relevancy", "context_recall", "context_precision"],
        "Avant SQL Tool": [
            df_avant["faithfulness"].mean(),
            df_avant["answer_relevancy"].mean(),
            df_avant["context_recall"].mean(),
            df_avant["context_precision"].mean(),
        ],
        "Après SQL Tool": [
            df_apres["faithfulness"].mean(),
            df_apres["answer_relevancy"].mean(),
            df_apres["context_recall"].mean(),
            df_apres["context_precision"].mean(),
        ]
    })

    global_scores["Gain (%)"] = (
        (global_scores["Après SQL Tool"] - global_scores["Avant SQL Tool"])
        / global_scores["Avant SQL Tool"]
    ) * 100

    st.dataframe(global_scores, use_container_width=True)

    st.subheader("📈 Visualisation des gains")
    st.bar_chart(global_scores.set_index("Metric")[["Avant SQL Tool", "Après SQL Tool"]])

    st.subheader("📌 Analyse automatique")

    for _, row in global_scores.iterrows():
        metric = row["Metric"]
        gain = row["Gain (%)"]

        if gain > 0:
            st.success(f"**{metric}** a augmenté de **{gain:.1f}%** après intégration du SQL Tool.")
        else:
            st.warning(f"**{metric}** a diminué de **{abs(gain):.1f}%** après intégration du SQL Tool.")

    st.stop()

# -------------------------------------------------------------------
# Modes existants (automatique / métier)
if mode == "Évaluation automatique":
    df = load_auto()
else:
    df = load_metier()

# -------------------------------------------------------------------
# Titre
st.title("📊 RAG Evaluation Dashboard")
st.subheader(f"Mode : {mode}")

# -------------------------------------------------------------------
# Vue globale
st.header("Vue globale des scores RAGAS")

st.dataframe(
    df[[
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision"
    ]].describe(),
    use_container_width=True
)

# -------------------------------------------------------------------
# Analyse par question
st.header("Analyse détaillée par question")

selected_question = st.selectbox(
    "Choisir une question",
    df["question"]
)

row = df[df["question"] == selected_question].iloc[0]

# -------------------------------------------------------------------
# Question / Réponses
st.markdown("### Question")
st.write(row["question"])

st.markdown("### Réponse générée par le système")
st.write(row["answer"])

st.markdown("### Ground Truth")
st.write(row["ground_truth"] if row["ground_truth"] else "— (pas de GT en mode métier)")

# -------------------------------------------------------------------
# Contexts côte à côte
st.markdown("### Analyse des contextes")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Contexte de génération (automatique)")
    if row["generation_context"]:
        st.info(row["generation_context"])
    else:
        st.info("— (pas de contexte de génération en mode métier)")

with col2:
    st.markdown("#### Contexte récupéré par le RAG")

    contexts = row["contexts"]

    if isinstance(contexts, str):
        try:
            contexts = ast.literal_eval(contexts)
        except:
            contexts = [contexts]

    if not contexts:
        st.info("🗄️ Réponse issue du module SQL (aucun contexte vectoriel utilisé).")
    else:
        for i, ctx in enumerate(contexts, 1):
            st.markdown(f"**Chunk {i}**")
            st.warning(ctx)

# -------------------------------------------------------------------
# Métriques par question
st.markdown("### Scores RAGAS pour cette question")

metrics_df = pd.DataFrame({
    "metric": [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision"
    ],
    "score": [
        row["faithfulness"],
        row["answer_relevancy"],
        row["context_recall"],
        row["context_precision"]
    ]
})

st.bar_chart(metrics_df.set_index("metric"))

# -------------------------------------------------------------------
# Analyse des cas faibles
st.header("🔎 Identifier les cas problématiques")

threshold = st.slider(
    "Seuil de Faithfulness",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

low_cases = df[df["faithfulness"] < threshold]

st.write(
    f"Nombre de cas avec Faithfulness < {threshold} : {len(low_cases)}"
)

st.dataframe(
    low_cases[[
        "question",
        "answer",
        "ground_truth",
        "faithfulness",
        "context_recall"
    ]],
    use_container_width=True
)

# -------------------------------------------------------------------
# Corrélation des métriques
st.header("Relations entre métriques")

st.dataframe(
    df[[
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision"
    ]].corr(),
    use_container_width=True
)
