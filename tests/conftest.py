from __future__ import annotations

import pytest

SAMPLE_PGN = """[Event "Game 1"]
[Site "?"]
[Date "2026.01.01"]
[Round "1"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0

[Event "Game 2"]
[Site "?"]
[Date "2026.01.02"]
[Round "1"]
[White "Carol"]
[Black "Dan"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1
"""


@pytest.fixture()
def sample_pgn_text() -> str:
    return SAMPLE_PGN


@pytest.fixture()
def single_game_pgn_text() -> str:
    return SAMPLE_PGN.split("\n\n[Event \"Game 2\"]", 1)[0]
