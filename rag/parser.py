"""PGN parsing helpers for chess_rag."""

from __future__ import annotations

from io import StringIO

import chess
import chess.pgn

MAX_GAMES = 100


def parse_multi_game_pgn(pgn_text: str) -> list[chess.pgn.Game]:
    """Parse a PGN blob containing one or more games.

    Returns at most 100 games. Raises ValueError if no valid games are found.
    """
    if not pgn_text or not pgn_text.strip():
        raise ValueError("PGN text is empty")

    handle = StringIO(pgn_text)
    games: list[chess.pgn.Game] = []
    while len(games) < MAX_GAMES:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        games.append(game)

    if not games:
        raise ValueError("No valid PGN games found")
    return games


def moves_to_san(game: chess.pgn.Game) -> list[str]:
    """Return the mainline moves in SAN notation."""
    board = game.board()
    san_moves: list[str] = []
    node = game
    while node.variations:
        node = node.variation(0)
        san_moves.append(board.san(node.move))
        board.push(node.move)
    return san_moves


def extract_board_at_ply(game: chess.pgn.Game, ply: int) -> chess.Board:
    """Return the board after ``ply`` half-moves, clamped to the game length."""
    if ply < 0:
        raise ValueError("ply must be >= 0")

    board = game.board()
    for idx, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        if idx >= ply:
            break
    return board
