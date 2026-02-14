import sys
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Ajouter la racine au PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Fix OpenMP conflict (FAISS + PyTorch)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import logging

from utils.config import APP_TITLE, NAME, MODEL_NAME

# ================================================================
# Configuration Streamlit

st.set_page_config(page_title=APP_TITLE, layout="centered")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)

# ================================================================
# Reconstruction automatique de la base vectorielle si absente
from utils.vector_store import VectorStoreManager
from utils.data_loader import load_and_parse_files

FAISS_INDEX = "vector_db/faiss_index.idx"
CHUNKS_FILE = "vector_db/document_chunks.pkl"

def rebuild_vector_db():
    st.info("Reconstruction de la base vectorielle…")
    docs = load_and_parse_files("inputs/pdf")
    vsm = VectorStoreManager()
    vsm.build_index(docs)
    st.success("Base vectorielle reconstruite avec succès !")

# Si la base vectorielle n'existe pas → on la reconstruit
if not (os.path.exists(FAISS_INDEX) and os.path.exists(CHUNKS_FILE)):
    rebuild_vector_db()

# IMPORTANT : recharger FAISS après reconstruction
from utils.vector_store import VectorStoreManager
GLOBAL_VECTOR_STORE = VectorStoreManager()

# ================================================================
# Router avec vector store global
from app.router import route_question

# ================================================================
# Initialisation de la session

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Je suis votre assistant d’analyse basketball pour {NAME}. Posez votre question."
    }]

# ================================================================
# Affichage du chat

st.title(APP_TITLE)
st.caption(f"{NAME} — Assistant IA | Modèle : {MODEL_NAME}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ================================================================
# Input utilisateur

if question := st.chat_input(f"Posez votre question sur {NAME}..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("Analyse en cours...")

        answer = route_question(question)

        placeholder.write(answer["answer"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer["answer"]
    })

# ================================================================
# Footer

st.markdown("---")
st.caption("Powered by Mistral AI • FAISS • RAG • SQL Tooling")