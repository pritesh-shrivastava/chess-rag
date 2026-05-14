# Deployment TODOs — chess_rag

Last updated: 2026-05-14 IST

## 1) Pending non-coding tasks from Pritesh's end

These are the things needed to get the app deployed publicly without changing code.

- [ ] Create a Hugging Face account if not already available.
- [ ] Create a new Hugging Face Space for this app.
  - Suggested name: `chess-rag`
  - Suggested visibility: public (best for portfolio use)
  - Runtime: Streamlit
- [ ] Decide the final public repo/Space naming you want to show on your resume/portfolio.
- [ ] Generate a Groq API key from `console.groq.com`.
- [ ] Add `GROQ_API_KEY` as a Hugging Face Space secret.
- [ ] Push the latest repo state to the GitHub repo that will back the Space.
- [ ] Connect the Hugging Face Space to this repo or upload/sync the code into the Space.
- [ ] Trigger the first deploy on Hugging Face Spaces.
- [ ] Open the live app and run one manual smoke test with `examples/sample_game.pgn`.
- [ ] Share the live Space URL back here once deploy succeeds.
- [ ] If deployment fails, paste the exact build/runtime error from Hugging Face so I can fix it quickly.

## 2) Subsequent coding tasks for Mercury

These are the next code/docs tasks I should handle after the above deploy steps are done or once you send me the first deploy error/URL.

- [ ] Add the real live demo URL to `README.md` (the current link is still a placeholder).
- [ ] Tighten Groq error handling in `app.py` so auth/config errors are shown differently from retryable transient failures.
- [ ] Add lightweight structured logging around deploy-critical states:
  - upload received
  - PGN decode result
  - game count parsed
  - retrieval warning/failure reason
  - fallback path used
- [ ] Add a deployment smoke-test path/checklist for the hosted app.
- [ ] Document deploy-time operational limits more explicitly:
  - cold start expectations
  - Hugging Face Spaces CPU/memory caveats
  - behavior when `GROQ_API_KEY` is missing, invalid, or rate-limited
- [ ] If the Hugging Face deploy exposes environment/runtime issues, patch the repo for Space compatibility (for example dependency/bootstrap or config fixes) and re-verify.

## 3) Definition of done for public deploy

- [ ] App loads on a public Hugging Face Space.
- [ ] Sample PGN runs end-to-end.
- [ ] Pattern retrieval warning behavior is correct.
- [ ] Groq-backed commentary works when the secret is set.
- [ ] README contains the actual live demo link.

## 4) Notes

Grounding for this TODO file:
- `README.md` still shows `**[Live demo →](#)** *(link after HF Spaces deploy)*`
- `PRODUCTION_READINESS.md` says the app is shipable for demo/portfolio use, but still recommends follow-up work around Groq error handling, observability, deploy smoke testing, and operational docs.
