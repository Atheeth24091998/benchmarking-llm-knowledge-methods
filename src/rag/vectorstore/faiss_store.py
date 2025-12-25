import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.utils.config_loader import load_config
from src.rag.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

EMBED_DIR = Path(config["embeddings_data_path"])


class FaissVectorStore:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = None

    def load(self):
        """Load embeddings and metadata, build FAISS index."""
        logger.info("Loading embeddings and metadata")

        embeddings = np.load(EMBED_DIR / "embeddings.npy")
        with (EMBED_DIR / "metadata.json").open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        dim = embeddings.shape[1]

        # Cosine similarity via inner product
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k most similar chunks."""
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            item = self.metadata[idx]
            results.append({
                "chunk_id": item["chunk_id"],
                "score": float(score),
                "text": item["text"], 
                "metadata": item["metadata"]
            })

        return results
