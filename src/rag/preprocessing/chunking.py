###################################################################################################################################################################
"""
DOCUMENTATION & HELPER LOGIC
----------------------------

1. estimate_tokens(text)
   - Inp: Raw text string.
   - Out: Integer (estimated token count).
   - Desc: A fast, free proxy for token counting (approx. 1 token = 4 chars) to avoid slow API calls.

2. normalize_text(text)
   - Inp: Raw text with PDF artifacts (e.g., 'n<br>').
   - Out: Cleaned, human-readable text.
   - Desc: Removes extraction artifacts, fixes broken newlines, and standardizes bullet points.

3. chunk_table(section_text, metadata)
   - Inp: A Markdown string containing a table + section metadata.
   - Out: List of chunk dictionaries.
   - Desc: Splits long tables into multiple chunks (e.g., 10 rows each) while ensuring *every* chunk keeps the table header row.

4. chunk_text(text, metadata)
   - Inp: Standard paragraph text + section metadata.
   - Out: List of chunk dictionaries.
   - Desc: Splits long text into chunks based on token limits, maintaining a "sliding window" overlap so context isn't cut off.

5. make_chunk(text, metadata, idx, ...)
   - Inp: Text content, metadata, and chunk index.
   - Out: A standardized dictionary object ready for embedding.
   - Desc: Formats the final JSON object, creating a unique ID (doc_id + section + chunk_id) and adding lineage metadata.

6. merge_tiny_sections(sections)
   - Inp: List of raw sections from the extraction step.
   - Out: List of merged sections.
   - Desc: Detects sections that are too small (like orphaned headers) and merges them into the next section to preserve context.

7. process_document(sections)
   - Inp: Full list of sections for one document.
   - Out: Final list of chunks.
   - Desc: Master pipeline that first merges tiny sections, then routes them to specific text or table chunkers.

8. load_jsonl(path) / save_jsonl(data, path)
   - Inp: File path / Data list.
   - Out: List of Dicts / None (writes file).
   - Desc: Simple file I/O helpers to read and write line-delimited JSON.
"""
###################################################################################################################################################################

import json
import re
from pathlib import Path
from typing import List, Dict

from src.rag.utils.config_loader import load_config
from src.rag.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

EXTRACTED_DIR = Path(config["extracted_data_path"])
OUT_DIR = Path(config["processed_data_path"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chunking parameters
max_rows = config["chunking"]["max_rows"]
max_tokens = config["chunking"]["max_tokens"]
min_tokens = config["chunking"]["min_tokens"]

# -------------------------
# Helpers
# -------------------------

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def normalize_text(text: str) -> str:
    """Clean PDF artifacts and normalize bullets."""
    text = re.sub(r'<br>|n<br>', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    lines = []
    for line in text.splitlines():
        line = line.strip()

        # Normalize bullet markers
        line = re.sub(r'^(n|u|Ú)\s+', '- ', line)

        if line:
            lines.append(line)

    return "\n".join(lines)


# -------------------------
# Chunking logic
# -------------------------

def chunk_table(section_text: str, metadata: Dict) -> List[Dict]:
    """Split markdown tables by rows."""
    lines = section_text.split("\n")
    table = [l for l in lines if l.strip().startswith("|")]
    header, separator, *rows = table

    if len(rows) <= max_rows:
        return [make_chunk(section_text, metadata, 0)]

    chunks = []
    for i in range(0, len(rows), max_rows):
        chunk_text = "\n".join([header, separator] + rows[i:i + max_rows])
        chunks.append(make_chunk(chunk_text, metadata, len(chunks), table_rows=len(rows[i:i + max_rows])))

    return chunks


def chunk_text(text: str, metadata: Dict) -> List[Dict]:
    """Paragraph-based chunking with light overlap."""
    paras = [p for p in text.split("\n\n") if p.strip()]

    if estimate_tokens(text) <= max_tokens:
        return [make_chunk(text, metadata, 0)]

    chunks, current, tokens = [], [], 0
    heading = paras[0]

    for para in paras:
        t = estimate_tokens(para)
        if tokens + t > max_tokens and current:
            chunks.append(make_chunk("\n\n".join(current), metadata, len(chunks)))
            current = [heading, current[-1]]
            tokens = estimate_tokens("\n\n".join(current))

        current.append(para)
        tokens += t

    if current:
        chunks.append(make_chunk("\n\n".join(current), metadata, len(chunks)))

    return chunks


def make_chunk(text: str, metadata: Dict, idx: int, **extra_meta) -> Dict:
    return {
        "chunk_id": f'{metadata["doc_id"]}__{metadata["section_full"]}__chunk_{idx:03d}',
        "text": normalize_text(text),
        "metadata": {**metadata, "chunk_index": idx, **extra_meta}
    }


def merge_tiny_sections(sections: List[Dict]) -> List[Dict]:
    merged, i = [], 0

    while i < len(sections):
        cur = sections[i]
        if estimate_tokens(cur["text"]) < min_tokens and i + 1 < len(sections):
            nxt = sections[i + 1]
            merged.append({
                "text": cur["text"] + "\n\n" + nxt["text"],
                "metadata": {**nxt["metadata"], "merged_with": cur["metadata"]["section_full"]}
            })
            i += 2
        else:
            merged.append(cur)
            i += 1

    return merged


def process_document(sections: List[Dict]) -> List[Dict]:
    sections = merge_tiny_sections(sections)
    chunks = []

    for sec in sections:
        if sec["metadata"].get("has_table"):
            chunks.extend(chunk_table(sec["text"], sec["metadata"]))
        else:
            chunks.extend(chunk_text(sec["text"], sec["metadata"]))

    return chunks


# -------------------------
# File I/O
# -------------------------

def load_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -------------------------
# Main
# -------------------------

def main():
    for file in sorted(EXTRACTED_DIR.glob("*.jsonl")):
        logger.info(f"Processing {file.name}")
        sections = load_jsonl(file)
        chunks = process_document(sections)
        save_jsonl(chunks, OUT_DIR / file.name)
        logger.info(f"Saved {len(chunks)} chunks")


if __name__ == "__main__":
    main()
