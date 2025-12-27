import json
import time
from pathlib import Path
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.utils.config_loader import load_config
from src.rag.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

PROCESSED_DIR = Path(config["processed_data_path"])
OUT_DIR = Path(config["embeddings_data_path"])
OUT_DIR.mkdir(parents=True, exist_ok=True)
batch_size = config["embeddings"]["embeddings_batch_size"]

def load_chunks():
    """Load all chunk JSONL files."""
    chunks = []

    for file in sorted(PROCESSED_DIR.glob("*.jsonl")):
        logger.info(f"Loading chunks from {file.name}")
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

    return chunks


def create_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Encode texts into embeddings."""
    logger.info(f"Loading embedding model: {model_name}")
    
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = SentenceTransformer(model_name)

    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    elapsed = time.time() - start

    logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
    return embeddings


def main():
    chunks = load_chunks()

    if not chunks:
        logger.warning("No chunks found. Exiting.")
        return

    texts = [c["text"] for c in chunks]
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "metadata": c["metadata"]
        }
        for c in chunks
    ]

    embeddings = create_embeddings(texts)

    # Save outputs
    np.save(OUT_DIR / "embeddings.npy", embeddings)

    with (OUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved embeddings to {OUT_DIR / 'embeddings.npy'}")
    logger.info(f"Saved metadata to {OUT_DIR / 'metadata.json'}")
    logger.info(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
