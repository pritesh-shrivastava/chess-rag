from __future__ import annotations

from rag.ingestion import parse_patterns


def test_parse_patterns_includes_new_training_material() -> None:
    text = open("data/sources/patterns.md", "r", encoding="utf-8").read()
    patterns = parse_patterns(text)
    titles = {pattern["title"] for pattern in patterns}

    assert "Open File Pressure" in titles
    assert "Half-Open File Pressure" in titles
    assert "Bishop Pair" in titles
    assert "Minority Attack" in titles
    assert "Weak Square" in titles
