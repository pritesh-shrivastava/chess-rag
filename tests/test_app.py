from __future__ import annotations

from pathlib import Path

import app
from rag.parser import extract_board_at_ply, moves_to_san, parse_multi_game_pgn
from rag.prompts import build_prompt


def test_app_import_smoke() -> None:
    assert callable(app.main)
    assert app.DEFAULT_MODELS


def test_groq_setup_notice_reflects_api_key_state(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert "No GROQ_API_KEY" in app._groq_setup_notice()

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert app._groq_setup_notice() is None


def test_sample_pgn_pipeline_builds_commentary_prompt(sample_pgn_text: str) -> None:
    games = parse_multi_game_pgn(sample_pgn_text)
    game = games[0]
    moves = moves_to_san(game)
    board = extract_board_at_ply(game, 4)
    prior_moves = moves[:4]

    openings = app.retrieve_opening_theory(prior_moves)
    assert openings, "expected the sample PGN to match at least one ECO opening"

    messages = build_prompt(openings, ["Back Rank Mate"], board, prior_moves)
    user_content = messages[1]["content"]

    assert "Board FEN:" in user_content
    assert "Retrieved opening theory:" in user_content
    assert "Back Rank Mate" in user_content


def test_decode_uploaded_pgn_prefers_utf8_variants() -> None:
    assert app.decode_uploaded_pgn("[Event \"UTF8\"]\n".encode("utf-8")) == "[Event \"UTF8\"]\n"
    assert app.decode_uploaded_pgn("[Event \"BOM\"]\n".encode("utf-8-sig")) == "[Event \"BOM\"]\n"


def test_decode_uploaded_pgn_falls_back_to_latin1() -> None:
    text = "[White \"Jos\xe9\"]\n\n1. e4 e5 1-0\n"
    payload = text.encode("latin-1")
    assert app.decode_uploaded_pgn(payload) == text


def test_call_groq_streams_successful_response_once(monkeypatch) -> None:
    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return [object()]

    captured = {"write_stream_calls": 0, "writes": []}

    monkeypatch.setattr(app, "_groq_client", lambda api_key: DummyClient())
    monkeypatch.setattr(app.st, "write_stream", lambda stream: captured.__setitem__("write_stream_calls", captured["write_stream_calls"] + 1) or "streamed text")
    monkeypatch.setattr(app.st, "write", lambda value: captured["writes"].append(value))

    streamed = app.call_groq([{"role": "user", "content": "hi"}])

    assert streamed is True
    assert captured["write_stream_calls"] == 1
    assert captured["writes"] == []


def test_sample_pgn_example_file_exists() -> None:
    example = Path("examples/sample_game.pgn")
    assert example.exists()
    text = example.read_text(encoding="utf-8")
    assert "[Event \"Game 1\"]" in text
    assert "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6" in text
