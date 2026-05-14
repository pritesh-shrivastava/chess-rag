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


def test_retrieve_pattern_explanation_context_distinguishes_failure_from_no_match(monkeypatch) -> None:
    monkeypatch.setattr(retriever, "_load_model", lambda: (_ for _ in ()).throw(RuntimeError("model load failed")))

    board = chess.Board()
    context = retriever.retrieve_pattern_context(board)

    assert context["patterns"] == []
    assert "Pattern retrieval is unavailable" in context["warning"]



def test_retrieve_pattern_explanation_context_reports_no_match_without_warning(monkeypatch) -> None:
    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.ones((1, 3), dtype="float32")

    class DummyIndex:
        ntotal = 2

        def search(self, embedding, top_k):
            return None, np.array([[-1, -1, -1]])

    monkeypatch.setattr(retriever, "_load_model", lambda: DummyModel())
    monkeypatch.setattr(retriever, "_load_index_and_patterns", lambda: (DummyIndex(), [
        {"text": "pattern 0"},
        {"text": "pattern 1"},
    ]))

    board = chess.Board()
    context = retriever.retrieve_pattern_context(board)

    assert context == {"patterns": [], "warning": None}



def test_retrieve_pattern_context_reports_search_failures_as_warning(monkeypatch) -> None:
    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            raise RuntimeError("embedding backend failed")

    class DummyIndex:
        ntotal = 2

        def search(self, embedding, top_k):
            return None, np.array([[0]])

    monkeypatch.setattr(retriever, "_load_model", lambda: DummyModel())
    monkeypatch.setattr(retriever, "_load_index_and_patterns", lambda: (DummyIndex(), [{"text": "pattern 0"}]))

    context = retriever.retrieve_pattern_context(chess.Board())

    assert context["patterns"] == []
    assert "Pattern retrieval is unavailable" in context["warning"]



def test_retrieve_pattern_context_reports_unexpected_search_failures_as_warning(monkeypatch) -> None:
    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.ones((1, 3), dtype="float32")

    class DummyIndex:
        ntotal = 2

        def search(self, embedding, top_k):
            raise TypeError("faiss returned unexpected shape")

    monkeypatch.setattr(retriever, "_load_model", lambda: DummyModel())
    monkeypatch.setattr(retriever, "_load_index_and_patterns", lambda: (DummyIndex(), [{"text": "pattern 0"}]))

    context = retriever.retrieve_pattern_context(chess.Board())

    assert context["patterns"] == []
    assert "Pattern retrieval is unavailable" in context["warning"]



def test_retrieve_pattern_context_warns_when_index_is_empty(monkeypatch) -> None:
    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.ones((1, 3), dtype="float32")

    class DummyIndex:
        ntotal = 0

        def search(self, embedding, top_k):
            return None, np.array([[]], dtype=int)

    monkeypatch.setattr(retriever, "_load_model", lambda: DummyModel())
    monkeypatch.setattr(retriever, "_load_index_and_patterns", lambda: (DummyIndex(), []))

    context = retriever.retrieve_pattern_context(chess.Board())

    assert context["patterns"] == []
    assert "rebuild" in context["warning"]
