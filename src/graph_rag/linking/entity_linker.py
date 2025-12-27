import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.graph_rag.graph_store.neo4j_store import Neo4jStore
from src.graph_rag.utils.logger import get_logger

class EntityLinker:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=self.device
        )

        self.store = Neo4jStore()
        self.problem_names = self._load_problems()
        self.problem_embeddings = self._embed(self.problem_names)

        self.logger.info(
            f"Loaded {len(self.problem_names)} problems on {self.device}"
        )

    def _load_problems(self):
        query = "MATCH (p:Problem) RETURN p.name AS name"
        with self.store.driver.session() as session:
            result = session.run(query)
            return [r["name"].replace("_", " ") for r in result]

    def _embed(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def link(self, question: str, threshold: float = 0.4):
        q_emb = self._embed([question])[0]
        sims = cosine_similarity([q_emb], self.problem_embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < threshold:
            self.logger.warning("No strong entity match found")
            return None, best_score

        problem = self.problem_names[best_idx]
        self.logger.info(
            f"Linked question → '{problem}' (score={best_score:.3f})"
        )
        return problem.replace(" ", "_"), best_score
