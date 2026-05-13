from __future__ import annotations

import chess

from rag.prompts import build_prompt


def test_build_prompt_handles_empty_context() -> None:
    board = chess.Board()
    messages = build_prompt([], [], board, [])
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "No opening theory match" in messages[1]["content"]
    assert "No tactical pattern match" in messages[1]["content"]


def test_build_prompt_includes_openings_and_patterns() -> None:
    board = chess.Board()
    messages = build_prompt(
        [{"eco": "B00", "name": "King's Pawn Game", "pgn": "1. e4"}],
        ["Back Rank Mate"],
        board,
        ["e4"],
    )
    content = messages[1]["content"]
    assert "B00 King's Pawn Game" in content
    assert "Back Rank Mate" in content
    assert "e4" in content
