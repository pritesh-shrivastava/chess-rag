"""Retrieval functions: ECO opening lookup, position description, pattern search."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import chess
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.ingestion import FAISS_INDEX_PATH, MODEL_NAME, PATTERNS_JSON_PATH

DATA_DIR = Path(__file__).parent.parent / "data"
SOURCES_DIR = DATA_DIR / "sources"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _load_index_and_patterns() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    patterns = json.loads(PATTERNS_JSON_PATH.read_text(encoding="utf-8"))
    return index, patterns


@lru_cache(maxsize=1)
def _load_eco() -> dict[str, dict]:
    eco: dict[str, dict] = {}
    for path in sorted(SOURCES_DIR.glob("*.tsv")):
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                moves = (row.get("pgn") or "").strip()
                if not moves:
                    continue
                eco[moves] = {
                    "eco": (row.get("eco") or "").strip(),
                    "name": (row.get("name") or "").strip(),
                    "pgn": moves,
                    "source": path.name,
                }
    return eco


def _prefix_keys(moves: list[str]) -> list[str]:
    keys: list[str] = []
    parts: list[str] = []
    for idx, move in enumerate(moves, start=1):
        if idx % 2 == 1:
            parts.append(f"{(idx + 1) // 2}. {move}")
        else:
            parts[-1] += f" {move}"
        keys.append(" ".join(parts))
    return keys


def retrieve_opening_theory(moves: list[str]) -> list[dict]:
    """Return ECO entries matching any prefix of the move list (longest match first)."""
    if not moves:
        return []
    eco = _load_eco()
    results: list[dict] = []
    for key in reversed(_prefix_keys(moves)):
        if key in eco:
            results.append(eco[key])
            if len(results) >= 3:
                break
    return results


def describe_position(board: chess.Board) -> str:
    """Convert board state to a natural-language query for semantic search."""
    if not any(board.piece_at(sq) for sq in chess.SQUARES):
        return "Empty board."

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
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
    all_pawns = white_pawns | black_pawns
    open_files = sum(
        1 for file_idx in range(8)
        if not any(chess.square_file(sq) == file_idx for sq in all_pawns)
    )
    if open_files >= 4:
        structure = "The position is open with many open files."
    elif open_files >= 2:
        structure = "The position is semi-open."
    else:
        structure = "The position is closed with few open files."

    def pawn_structure(color: chess.Color) -> str:
        pawns = board.pieces(chess.PAWN, color)
        if not pawns:
            return "no pawns"
        isolated = 0
        passed = 0
        for sq in pawns:
            file_idx = chess.square_file(sq)
            neigh_files = [f for f in (file_idx - 1, file_idx + 1) if 0 <= f < 8]
            if not any(chess.square_file(other) in neigh_files for other in pawns if other != sq):
                isolated += 1
            rank = chess.square_rank(sq)
            same_or_adjacent_files = {file_idx - 1, file_idx, file_idx + 1}
            enemy_pawns = board.pieces(chess.PAWN, not color)
            if color == chess.WHITE:
                blockers = [
                    other for other in enemy_pawns
                    if chess.square_file(other) in same_or_adjacent_files and chess.square_rank(other) > rank
                ]
            else:
                blockers = [
                    other for other in enemy_pawns
                    if chess.square_file(other) in same_or_adjacent_files and chess.square_rank(other) < rank
                ]
            if not blockers:
                passed += 1
        parts = []
        if isolated:
            parts.append(f"{isolated} isolated pawn(s)")
        if passed:
            parts.append(f"{passed} passed pawn(s)")
        return ", ".join(parts) if parts else "healthy pawn structure"

    def king_safety(color: chess.Color) -> str:
        king_sq = board.king(color)
        if king_sq is None:
            return "king not present"
        file_idx = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)
        side = "White" if color == chess.WHITE else "Black"
        if color == chess.WHITE and rank == 0 and file_idx in (6, 2):
            return f"{side} king safely castled"
        if color == chess.BLACK and rank == 7 and file_idx in (6, 2):
            return f"{side} king safely castled"
        if (color == chess.WHITE and rank >= 2) or (color == chess.BLACK and rank <= 5):
            return f"{side} king exposed in center"
        return f"{side} king position unclear"

    safety = f"{king_safety(chess.WHITE)}; {king_safety(chess.BLACK)}."
    pawn_notes = f"White pawns: {pawn_structure(chess.WHITE)}. Black pawns: {pawn_structure(chess.BLACK)}."
    return f"{material} {structure} {pawn_notes} {safety}"


def retrieve_pattern_explanation(board: chess.Board, top_k: int = 3) -> list[str]:
    """Embed a NL description of the position and return top-k matching patterns."""
    model = _load_model()
    index, patterns = _load_index_and_patterns()
    query = describe_position(board)
    embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embedding)
    _, indices = index.search(embedding, top_k)
    return [patterns[i]["text"] for i in indices[0] if i < len(patterns)]
