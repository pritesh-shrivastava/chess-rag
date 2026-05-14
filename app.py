"""Streamlit UI for chess_rag."""

from __future__ import annotations

import os
from typing import Iterable

import chess
import streamlit as st

from rag.parser import extract_board_at_ply, moves_to_san, parse_multi_game_pgn
from rag.prompts import build_prompt
from rag.retriever import (
    describe_position,
    pattern_knowledge_base_ready,
    retrieve_opening_theory,
    retrieve_pattern_explanation,
)

DEFAULT_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
]


def _groq_setup_notice() -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return None
    return (
        "No GROQ_API_KEY is configured, so the app will use deterministic fallback commentary. "
        "Copy .env.example to .env and add your Groq key to enable live model responses."
    )


def decode_uploaded_pgn(payload: bytes) -> str:
    """Decode uploaded PGN bytes using common export encodings."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", payload, 0, 1, "Unable to decode uploaded PGN")


def _game_label(game: chess.pgn.Game) -> str:
    white = game.headers.get("White", "White")
    black = game.headers.get("Black", "Black")
    result = game.headers.get("Result", "?")
    date = game.headers.get("Date", "?")
    return f"{white} vs. {black} ({result}) — {date[:10]}"


def _selected_game_index(games: list[chess.pgn.Game]) -> int:
    labels = [_game_label(game) for game in games]
    return st.selectbox("Select game", range(len(games)), format_func=lambda i: labels[i])


def _selected_ply(moves: list[str]) -> int:
    options = list(range(0, len(moves) + 1))

    def fmt(ply: int) -> str:
        if ply == 0:
            return "Start position"
        return f"{ply}. {moves[ply - 1]}"

    return st.selectbox("Select move", options, format_func=fmt)


def stream_groq_response(stream: Iterable) -> Iterable[str]:
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            yield delta


@st.cache_resource
def _groq_client(api_key: str | None):
    if not api_key:
        return None
    from groq import Groq  # local import so app still loads without the package at runtime

    return Groq(api_key=api_key)


def call_groq(messages: list[dict]) -> bool:
    api_key = os.getenv("GROQ_API_KEY")
    client = _groq_client(api_key)
    if client is None:
        return False

    last_error: Exception | None = None
    models = [os.getenv("GROQ_MODEL")] if os.getenv("GROQ_MODEL") else DEFAULT_MODELS
    for model in models:
        if not model:
            continue
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                stream=True,
            )
            st.write_stream(stream_groq_response(response))
            return True
        except Exception as exc:  # noqa: BLE001 - fallback chain on provider errors
            last_error = exc
            continue
    if last_error:
        st.warning(f"Groq unavailable right now: {last_error}")
    return False


def fallback_commentary(openings: list[dict], patterns: list[str], board: chess.Board, moves: list[str]) -> str:
    opening_text = ", ".join(f"{o.get('eco', '')} {o.get('name', '')}".strip() for o in openings) or "No opening match"
    pattern_text = patterns[0] if patterns else "No strong pattern match"
    return (
        f"Position summary: {describe_position(board)}\n\n"
        f"Opening theory: {opening_text}\n"
        f"Pattern: {pattern_text}\n\n"
        f"Moves so far: {', '.join(moves) if moves else 'None'}\n\n"
        "The key idea is to compare the current board against the retrieved opening and pattern context, "
        "then choose the most forcing plan: attack the king, win material, or simplify if the position is safe."
    )


def main() -> None:
    st.set_page_config(page_title="Chess RAG", page_icon="♟️", layout="wide")
    st.title("♟️ Chess RAG")
    st.caption("Upload a PGN, choose a move, and get commentary grounded in opening theory and chess patterns.")

    notice = _groq_setup_notice()
    if notice:
        st.info(notice)

    uploaded = st.file_uploader("Upload a PGN file", type=["pgn", "txt"])
    if not uploaded:
        st.info("Upload a PGN to start. You can use the sample file in examples/sample_game.pgn.")
        return

    try:
        pgn_text = decode_uploaded_pgn(uploaded.read())
        games = parse_multi_game_pgn(pgn_text)
    except UnicodeDecodeError:
        st.error("Could not decode PGN file. Try exporting as UTF-8 PGN or plain text.")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not parse PGN: {exc}")
        return

    st.success(f"Loaded {len(games)} game(s)")
    game = games[_selected_game_index(games)] if len(games) > 1 else games[0]

    moves = moves_to_san(game)
    if not moves:
        st.warning("This game has no moves to analyze.")
        return

    ply = _selected_ply(moves)
    board = extract_board_at_ply(game, ply)
    prior_moves = moves[:ply]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Position snapshot")
        st.code(board.fen(), language="text")
    with col2:
        st.subheader("Retrieval context")
        openings = retrieve_opening_theory(prior_moves)
        if pattern_knowledge_base_ready():
            patterns = retrieve_pattern_explanation(board)
        else:
            patterns = []
            st.warning("Pattern index is missing; run `uv run python -m rag.ingestion` to rebuild it.")
        st.write("**Opening theory**")
        st.write(openings if openings else ["No opening match"])
        st.write("**Pattern matches**")
        st.write(patterns if patterns else ["No pattern match"])

    messages = build_prompt(openings, patterns, board, prior_moves)
    st.subheader("Commentary")
    streamed = call_groq(messages)
    if not streamed:
        st.write(fallback_commentary(openings, patterns, board, prior_moves))


if __name__ == "__main__":
    main()
