# chess_rag

RAG-based chess game analysis. Upload a PGN, pick a move, get AI commentary grounded in opening theory and tactical patterns.

## Stack

- Python 3.11, managed with `uv`
- Streamlit frontend
- scikit-learn TF-IDF for pattern retrieval
- ECO opening database (dict prefix match, no ML needed)
- Groq API (llama-3.3-70b-versatile, free tier)

## Commands

```bash
# Install deps
uv sync --group dev

# Run app locally (needs GROQ_API_KEY in .env)
uv run streamlit run app.py

# Build the pattern index (run once, output committed to repo)
uv run python -m rag.ingestion

# Run tests
uv run pytest

# Run tests excluding index-dependent tests
uv run pytest -m "not requires_index"
```

## Project structure

```
app.py              # Streamlit UI (~80 lines, pure orchestration)
rag/
  parser.py         # PGN parsing, SAN extraction, board-at-ply
  retriever.py      # ECO lookup, describe_position, pattern retrieval
  ingestion.py      # Build TF-IDF index from patterns.md (run once)
  prompts.py        # build_prompt: system prompt + citation instructions
data/
  chess_index.pkl   # Serialized TF-IDF vectorizer + matrix (generated)
  sources/
    eco.json        # ECO opening codes (public domain)
    patterns.md     # 25-50 chess pattern descriptions (knowledge base)
tests/
  conftest.py
  test_parser.py
  test_retriever.py
  test_prompts.py
```

## Architecture

```
PGN upload → parse_multi_game_pgn() → game selector → move selector
  → retrieve_opening_theory(moves[:ply])     # ECO dict prefix match
  → describe_position(board)                  # NL description of position
  → retrieve_pattern_explanation(board)       # TF-IDF search over patterns.md
  → build_prompt(openings, patterns, board)   # rag/prompts.py
  → call_groq() with 3-model fallback chain   # Groq streaming API
  → st.write_stream()                         # display to user
```

## Testing

Framework: pytest. Test files in `tests/`. Two categories:

- Normal tests: run anywhere, no index needed
- `@pytest.mark.requires_index`: need `data/chess_index.pkl` (run `rag.ingestion` first)

```bash
# All tests
uv run pytest

# Skip index-dependent tests (fast, for CI without index)
uv run pytest -m "not requires_index"
```

## Environment

Copy `.env.example` to `.env` and add your Groq API key (free at console.groq.com).

The app falls back gracefully: if `GROQ_API_KEY` is missing, shows a setup message.

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.

Key routing rules:
- Product ideas, brainstorming, "is this worth building" → invoke office-hours
- Bugs, errors, "why is this broken" → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the feature, find bugs → invoke qa
- Code review, check my diff → invoke review
- Architecture review → invoke plan-eng-review
