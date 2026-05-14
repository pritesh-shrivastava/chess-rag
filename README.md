# chess_rag

> Upload a chess game, pick a move, get AI analysis that explains WHY — citing opening theory, naming the pattern, grounded in real chess knowledge.

Stockfish tells you "Nf6 was better (+0.5)." This tells you *why* and what pattern you walked into.

**[Live demo →](#)** *(link after HF Spaces deploy)*

## How it works

```mermaid
flowchart TD
    A[Upload PGN\nchess.com export, up to 100 games] --> B[parse_multi_game_pgn]
    B --> C{Multiple games?}
    C -->|yes| D[Game selector\nvs. Opponent · result · date]
    C -->|no| E[Move selector]
    D --> E
    E --> F[retrieve_opening_theory\nECO prefix match]
    E --> G[describe_position\nboard → natural language]
    G --> H[Semantic search\nFAISS + sentence-transformers]
    H --> I[retrieve_pattern_explanation]
    F --> J[build_prompt\nwith citations]
    I --> J
    J --> K[Groq API\nllama-3.3-70b-versatile]
    K --> L[Stream analysis to UI]
```

## RAG pipeline

1. **Retrieve** — ECO opening theory (dict prefix match on SAN moves) + tactical pattern (FAISS semantic search over 53 annotated pattern docs)
2. **Augment** — inject retrieved context into the prompt before the LLM sees it
3. **Generate** — Groq streams the analysis, citing the retrieved sources

The retrieval functions are named and structured so the agentic architecture is readable:

```python
openings = retrieve_opening_theory(moves[:ply])        # ECO: what opening is this?
patterns = retrieve_pattern_explanation(board)          # patterns.md: what's happening here?
prompt   = build_prompt(openings, patterns, board, moves)
response = call_groq(prompt, stream=True)
```

## Run locally

```bash
git clone <repo>
cd chess_rag
uv sync                          # installs all deps (Python 3.11, managed by uv)
uv run python -m rag.ingestion   # build the pattern index (run once)
cp .env.example .env             # add GROQ_API_KEY only if you want live Groq commentary
uv run streamlit run app.py      # works with or without an API key (fallback mode available)
```

If `data/chess_index.faiss` or `data/chess_patterns.json` is missing, the app now stays usable: it shows opening theory, skips pattern matches, and surfaces a warning telling you to rebuild the index.

## Demo with the bundled sample PGN

If you want to sanity-check the full pipeline without exporting your own game yet, open `examples/sample_game.pgn`.
It contains two short games that exercise:

- multi-game parsing
- game selection
- move selection
- opening lookup
- pattern retrieval
- local fallback commentary when `GROQ_API_KEY` is missing

The sample file is also used by the test suite as a smoke check.

## Run tests

```bash
uv run pytest                              # all tests
uv run pytest -m "not requires_index"     # skip index-dependent tests
```

## Stack

- The project uses `pyproject.toml` for ranges and `uv.lock` for exact pinned dependency versions.

| Layer | Tech |
|---|---|
| Frontend | Streamlit |
| Pattern retrieval | sentence-transformers + FAISS cosine similarity |
| Opening lookup | ECO database (dict prefix match, zero ML) |
| LLM | Groq free tier (llama-3.3-70b-versatile, 14,400 req/day) |
| PGN parsing | python-chess |
| Hosting | Hugging Face Spaces (free CPU tier) |
| Deps | uv |

## Knowledge base

- **ECO openings** — ~100 opening codes with move sequences (public domain, from lichess-org/eco)
- **Chess patterns** — 53 annotated pattern descriptions (Greek Gift, Back Rank Mate, Isolated Queen Pawn, etc.) — each ~200 words covering position indicators, strategic idea, and example line

## Why not Stockfish?

Stockfish gives you evaluation numbers. This gives you *explanations*. "Your bishop on c4 targets the f7 pawn, a weakness in the Italian Game — this is the Italian Attack setup" is not something any engine can say. RAG makes it possible.
