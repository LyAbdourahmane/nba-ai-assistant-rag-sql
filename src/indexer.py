# indexer.py
import argparse
import logging
import time
from typing import Optional

from utils.config import INPUT_DIR
from utils.data_loader import download_and_extract_zip, load_and_parse_files
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_indexing(input_directory: str, data_url: Optional[str] = None):
    logging.info("=== Démarrage du processus d'indexation ===")
    start_time = time.time()

    # Étape 1 : Téléchargement optionnel
    if data_url:
        logging.info(f"Téléchargement depuis : {data_url}")
        if not download_and_extract_zip(data_url, input_directory):
            logging.error("Échec du téléchargement ou de l'extraction. Arrêt.")
            return
    else:
        logging.info(f"Utilisation des fichiers locaux dans : {input_directory}")

    # Étape 2 : Parsing
    logging.info(f"Chargement et parsing des fichiers...")
    documents = load_and_parse_files(input_directory)

    if not documents:
        logging.warning("Aucun document trouvé. Arrêt.")
        return

    logging.info(f"{len(documents)} documents chargés.")

    # Étape 3 : Indexation
    logging.info("Initialisation du Vector Store...")
    vector_store = VectorStoreManager()

    logging.info("Construction de l'index FAISS...")
    try:
        vector_store.build_index(documents)
    except Exception as e:
        logging.error(f"Erreur lors de la construction de l'index : {e}")
        return

    # Résumé
    duration = time.time() - start_time
    logging.info("=== Indexation terminée avec succès ===")
    logging.info(f"Durée totale : {duration:.2f} secondes")
    logging.info(f"Documents traités : {len(documents)}")
    logging.info(f"Chunks indexés : {vector_store.index.ntotal if vector_store.index else 0}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument("--input-dir", type=str, default=INPUT_DIR)
    parser.add_argument("--data-url", type=str, default=None)
    args = parser.parse_args()

    run_indexing(input_directory=args.input_dir, data_url=args.data_url)
