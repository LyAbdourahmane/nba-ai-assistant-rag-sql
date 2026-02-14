from langchain_core.embeddings import Embeddings
from mistralai.client import MistralClient
import numpy as np
from utils.config import MISTRAL_API_KEY, EMBEDDING_MODEL

class MistralRagasEmbeddings(Embeddings):
    """Wrapper embeddings compatible RAGAS."""

    def __init__(self):
        self.client = MistralClient(api_key=MISTRAL_API_KEY)

    def embed_documents(self, texts):
        response = self.client.embeddings(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [np.array(d.embedding, dtype="float32") for d in response.data]

    def embed_query(self, text):
        response = self.client.embeddings(
            model=EMBEDDING_MODEL,
            input=[text]
        )
        return np.array(response.data[0].embedding, dtype="float32")
