"""Retrieval functions: ECO opening lookup, position description, pattern search."""

import json
from functools import lru_cache
from pathlib import Path

import chess
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.ingestion import FAISS_INDEX_PATH, MODEL_NAME, PATTERNS_JSON_PATH

DATA_DIR = Path(__file__).parent.parent / "data"
ECO_PATH = DATA_DIR / "sources" / "eco.json"

# loaded once at module level to avoid cold-starts on every Streamlit rerun
_model = SentenceTransformer(MODEL_NAME)
_index = faiss.read_index(str(FAISS_INDEX_PATH))
_patterns: list[dict] = json.loads(PATTERNS_JSON_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_eco() -> dict:
    return json.loads(ECO_PATH.read_text(encoding="utf-8"))


def retrieve_opening_theory(moves: list[str]) -> list[dict]:
    """Return ECO entries matching any prefix of the move list (longest match first)."""
    if not moves:
        return []
    eco = _load_eco()
    results = []
    for length in range(len(moves), 0, -1):
        key = " ".join(moves[:length])
        if key in eco:
            results.append(eco[key])
            if len(results) >= 3:
                break
    return results


def describe_position(board: chess.Board) -> str:
    """Convert board state to a natural-language query for semantic search."""
    if not any(board.piece_at(sq) for sq in chess.SQUARES):
        return "Empty board."

    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9}
    white_mat = sum(v * len(board.pieces(pt, chess.WHITE)) for pt, v in piece_values.items())
    black_mat = sum(v * len(board.pieces(pt, chess.BLACK)) for pt, v in piece_values.items())
    diff = white_mat - black_mat
    if diff > 0:
        material = f"White is up {diff} points of material."
    elif diff < 0:
        material = f"Black is up {abs(diff)} points of material."
    else:
        material = "Material is equal."

    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    open_files = sum(
        1 for f in range(8)
        if not any(chess.square_file(sq) == f for sq in white_pawns | black_pawns)
    )
    if open_files >= 4:
        structure = "The position is open with many open files."
    elif open_files >= 2:
        structure = "The position is semi-open."
    else:
        structure = "The position is closed with few open files."

    def king_safety(color: chess.Color) -> str:
        king_sq = board.king(color)
        if king_sq is None:
            return "king not present"
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)
        side = "White" if color == chess.WHITE else "Black"
        if color == chess.WHITE and rank == 0 and file in (6, 2):
            return f"{side} king safely castled"
        if color == chess.BLACK and rank == 7 and file in (6, 2):
            return f"{side} king safely castled"
        if (color == chess.WHITE and rank >= 2) or (color == chess.BLACK and rank <= 5):
            return f"{side} king exposed in center"
        return f"{side} king position unclear"

    safety = f"{king_safety(chess.WHITE)}; {king_safety(chess.BLACK)}."
    return f"{material} {structure} {safety}"


def retrieve_pattern_explanation(board: chess.Board, top_k: int = 3) -> list[str]:
    """Embed a NL description of the position and return top-k matching patterns."""
    query = describe_position(board)
    embedding = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embedding)
    _, indices = _index.search(embedding, top_k)
    return [_patterns[i]["text"] for i in indices[0] if i < len(_patterns)]
