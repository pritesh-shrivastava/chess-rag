"""Build FAISS index from data/sources/patterns.md. Run once; output committed to repo."""

import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
PATTERNS_PATH = DATA_DIR / "sources" / "patterns.md"
FAISS_INDEX_PATH = DATA_DIR / "chess_index.faiss"
PATTERNS_JSON_PATH = DATA_DIR / "chess_patterns.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def parse_patterns(text: str) -> list[dict]:
    """Split markdown into one dict per ## section with keys: title, body, text."""
    sections = re.split(r"\n## ", text)
    patterns = []
    for section in sections[1:]:  # skip preamble before first ##
        lines = section.strip().split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        patterns.append({"title": title, "body": body, "text": f"{title}\n{body}"})
    return patterns


def main() -> None:
    text = PATTERNS_PATH.read_text(encoding="utf-8")
    patterns = parse_patterns(text)
    print(f"Parsed {len(patterns)} patterns from {PATTERNS_PATH.name}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode([p["text"] for p in patterns], show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)  # cosine similarity via inner product
    print(f"Encoded {len(patterns)} patterns — embedding dim: {embeddings.shape[1]}")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine after L2-norm
    index.add(embeddings)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    PATTERNS_JSON_PATH.write_text(json.dumps(patterns, indent=2), encoding="utf-8")
    print(f"FAISS index → {FAISS_INDEX_PATH}")
    print(f"Patterns metadata → {PATTERNS_JSON_PATH}")


if __name__ == "__main__":
    main()
