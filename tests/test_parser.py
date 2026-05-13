from __future__ import annotations

import pytest

from rag.parser import extract_board_at_ply, moves_to_san, parse_multi_game_pgn


def test_parse_multi_game_pgn_returns_games(sample_pgn_text: str) -> None:
    games = parse_multi_game_pgn(sample_pgn_text)
    assert len(games) == 2
    assert games[0].headers["White"] == "Alice"


def test_parse_multi_game_pgn_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_multi_game_pgn("   ")


def test_moves_to_san(sample_pgn_text: str) -> None:
    game = parse_multi_game_pgn(sample_pgn_text)[0]
    assert moves_to_san(game) == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]


def test_extract_board_at_ply_clamps_to_end(sample_pgn_text: str) -> None:
    game = parse_multi_game_pgn(sample_pgn_text)[0]
    board = extract_board_at_ply(game, 999)
    assert board.fullmove_number >= 4
    assert board.turn is True or board.turn is False


def test_extract_board_at_ply_rejects_negative(sample_pgn_text: str) -> None:
    game = parse_multi_game_pgn(sample_pgn_text)[0]
    with pytest.raises(ValueError):
        extract_board_at_ply(game, -1)
