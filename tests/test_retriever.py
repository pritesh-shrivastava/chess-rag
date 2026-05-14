from __future__ import annotations

import numpy as np
import chess

import rag.retriever as retriever


def test_retrieve_opening_theory_finds_king_pawn_game() -> None:
    results = retriever.retrieve_opening_theory(["e4"])
    assert results
    assert results[0]["name"]
    assert results[0]["eco"] == "B00" or results[0]["eco"].startswith("C") or results[0]["eco"].startswith("A")


def test_retrieve_opening_theory_finds_italian_game() -> None:
    results = retriever.retrieve_opening_theory(["e4", "e5", "Nf3", "Nc6", "Bc4"])
    assert any("Italian" in result["name"] for result in results)


def test_retrieve_opening_theory_finds_london_system() -> None:
    results = retriever.retrieve_opening_theory(["d4", "d5", "Bf4"])
    assert any("London" in result["name"] for result in results)


def test_describe_position_mentions_material_and_structure() -> None:
    board = chess.Board()
    text = retriever.describe_position(board)
    assert "Material is equal." in text
    assert "closed" in text.lower()


def test_retrieve_pattern_explanation_returns_empty_when_index_missing(monkeypatch) -> None:
    monkeypatch.setattr(retriever, "_load_model", lambda: (_ for _ in ()).throw(FileNotFoundError()))

    board = chess.Board()
    assert retriever.retrieve_pattern_explanation(board) == []


def test_retrieve_pattern_explanation_filters_invalid_indices(monkeypatch) -> None:
    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.ones((1, 3), dtype="float32")

    class DummyIndex:
        def search(self, embedding, top_k):
            return None, np.array([[1, -1, 0]])

    monkeypatch.setattr(retriever, "_load_model", lambda: DummyModel())
    monkeypatch.setattr(retriever, "_load_index_and_patterns", lambda: (DummyIndex(), [
        {"text": "pattern 0"},
        {"text": "pattern 1"},
    ]))

    board = chess.Board()
    assert retriever.retrieve_pattern_explanation(board, top_k=3) == ["pattern 1", "pattern 0"]
