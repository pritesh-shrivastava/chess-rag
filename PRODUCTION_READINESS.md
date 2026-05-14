# Production readiness review

Reviewed on: 2026-05-14 17:16:25 IST

## Verdict

**Shipable for demo / portfolio use, not fully production-hardened yet.**

Core app flow is in good shape:
- test suite passes
- missing FAISS artifacts fail soft instead of breaking the UI
- local fallback commentary works without a Groq key

This pass fixed two concrete runtime issues:
- uploaded PGNs now decode with `utf-8-sig`, `utf-8`, or `latin-1`
- streamed Groq commentary is no longer rendered twice in the UI

## Fixed in this pass

### 1) PGN decoding robustness
- **Files:** `app.py`, `tests/test_app.py`
- **Risk before:** non-UTF-8 PGNs could fail before parsing even when the content was otherwise valid
- **Change:** added `decode_uploaded_pgn()` with common export encoding fallbacks and a clearer user-facing decode error

### 2) Streamed commentary duplication
- **Files:** `app.py`, `tests/test_app.py`
- **Risk before:** successful Groq responses were streamed once via `st.write_stream()` and then written again via `st.write(commentary)`
- **Change:** only fallback commentary is explicitly written; streamed responses render once

### 3) Groq client cache safety
- **Files:** `app.py`
- **Risk before:** the cached client did not vary with `GROQ_API_KEY`, so key rotation could require a restart/cache clear
- **Change:** `_groq_client()` now takes the API key as an argument so Streamlit cache keys track the configured secret

## Still recommended before a real public launch

### High priority
1. **Surface retrieval failures separately from true “no pattern match” cases**
   - **File:** `rag/retriever.py`
   - Right now corrupted artifacts / model load failures can collapse into an empty result list.
   - Add structured logging plus a UI warning for retrieval-system failures.

2. **Differentiate retryable vs non-retryable Groq errors**
   - **File:** `app.py`
   - The current model fallback loop retries on any exception.
   - Stop early on auth / validation errors and show a cleaner user-facing message.

### Medium priority
3. **Add basic observability for deploys**
   - Log key app states: upload received, PGN decoded, game count, retrieval failure reason, provider fallback used.
   - Even lightweight structured logs would make Hugging Face Spaces debugging much easier.

4. **Add a deployment smoke test**
   - A small test covering sample PGN upload → prompt build → fallback commentary path would protect the demo flow.

5. **Document operational limits**
   - Spell out expected cold-start behavior for `sentence-transformers`
   - Document memory/CPU expectations for Hugging Face Spaces
   - Document what happens when `GROQ_API_KEY` is absent, invalid, or rate-limited

## Suggested ship bar

### Good enough to demo now
- local `uv run pytest` passes
- app loads without `GROQ_API_KEY`
- sample PGN works end to end
- missing pattern index shows warning but app remains usable

### Good enough for a public portfolio deploy
- add retrieval error visibility
- tighten Groq error handling
- run one clean hosted smoke test on the target environment
