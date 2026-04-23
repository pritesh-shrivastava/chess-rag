# TODOS

## In Progress

## Open

### `rag/parser.py`
- [ ] `parse_multi_game_pgn(pgn_text: str) -> list[chess.pgn.Game]` — up to 100 games, raises `ValueError` on bad PGN
- [ ] `moves_to_san(game: chess.pgn.Game) -> list[str]` — walk game tree, call `.san()` with board state at each ply
- [ ] `extract_board_at_ply(game: chess.pgn.Game, ply: int) -> chess.Board` — clamp to last position if ply > game length

### `rag/prompts.py`
- [ ] `build_prompt(openings, patterns, board, moves) -> list[dict]` — returns Groq messages list
- [ ] System prompt: cite retrieved sources, explain WHY not just WHAT
- [ ] Handle empty openings and empty patterns gracefully (no KeyError)

### `app.py`
- [ ] `load_index()` with `@st.cache_resource` — `FileNotFoundError` → `st.error` + `st.stop()`
- [ ] `stream_groq_response(stream)` — unwrap `ChatCompletionChunk` to string generator
- [ ] `call_groq(messages)` — try each model in `GROQ_MODEL_CHAIN`, catch `RateLimitError` + `NotFoundError`
- [ ] Game selector UI (vs. Opponent · result · date)
- [ ] Move selector + analysis panel (~80 lines total)

### Tests
- [ ] `tests/conftest.py` — shared fixtures: sample PGNs, sample boards
- [ ] `tests/test_parser.py` — 9 tests: parse, truncation, error paths
- [ ] `tests/test_retriever.py` — 8 tests: ECO lookup, describe_position, pattern retrieval
- [ ] `tests/test_prompts.py` — 3 tests: build_prompt edge cases

### Deploy
- [ ] Deploy to Hugging Face Spaces (free CPU tier, Streamlit SDK)
- [ ] Set `GROQ_API_KEY` as HF Space secret
- [ ] Update README demo link

## Completed
- [x] `pyproject.toml` — faiss-cpu, sentence-transformers, groq, python-dotenv, streamlit, chess
- [x] `rag/ingestion.py` — sentence-transformers + FAISS index builder
- [x] `rag/retriever.py` — retrieve_opening_theory, describe_position, retrieve_pattern_explanation
- [x] `data/chess_index.faiss` + `data/chess_patterns.json` — committed index (53 patterns)
- [x] `data/sources/patterns.md` — 53 annotated chess patterns
- [x] `data/sources/a.tsv` – `e.tsv` — full Lichess ECO database
