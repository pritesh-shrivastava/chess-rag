"""Prompt builder for chess commentary generation."""

from __future__ import annotations

from textwrap import dedent

import chess


def _format_openings(openings: list[dict]) -> str:
    if not openings:
        return "No opening theory match was found."
    lines = []
    for opening in openings:
        name = opening.get("name", "Unknown opening")
        eco = opening.get("eco", "")
        pgn = opening.get("pgn", "")
        lines.append(f"- {eco} {name}: {pgn}")
    return "\n".join(lines)


def _format_patterns(patterns: list[str]) -> str:
    if not patterns:
        return "No tactical pattern match was found."
    return "\n".join(f"- {pattern}" for pattern in patterns)


def build_prompt(
    openings: list[dict],
    patterns: list[str],
    board: chess.Board,
    moves: list[str],
) -> list[dict]:
    """Build a Groq/OpenAI-style messages list for chess commentary."""
    system = dedent(
        """
        You are an expert chess coach.
        Explain the position in plain English, grounded in the provided retrieval context.
        Always cite the opening theory and tactical patterns when relevant.
        If the retrieved context is weak or empty, say so instead of inventing facts.
        Keep the answer concise, useful, and educational.
        """
    ).strip()

    user = dedent(
        f"""
        Analyze this chess position.

        Moves played so far (SAN): {', '.join(moves) if moves else 'None'}
        Board FEN: {board.fen()}

        Retrieved opening theory:
        {_format_openings(openings)}

        Retrieved tactical patterns:
        {_format_patterns(patterns)}

        What should the player understand about the position, plan, and tactical idea?
        """
    ).strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
