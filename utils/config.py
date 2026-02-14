import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ Attention: MISTRAL_API_KEY manquante dans .env")

# --- Models ---
EMBEDDING_MODEL = "mistral-embed"
MODEL_NAME = "mistral-small-latest"

# --- Directories ---
INPUT_DIR = "inputs"
VECTOR_DB_DIR = "vector_db"
DATABASE_DIR = "database"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)

# --- Vector Store Files ---
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")

# --- Chunking ---
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 32

# --- Retrieval ---
SEARCH_K = 5

# --- Database ---
DATABASE_FILE = os.path.join(DATABASE_DIR, "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

# --- App ---
APP_TITLE = "NBA Analyst AI"
NAME = "NBA"

# --- Debug / Logging ---
ENABLE_LOGFIRE = True
DEBUG = True
