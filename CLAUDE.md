# Project guide for Claude Code

Research project: drift-aware federated HAR on PAMAP2 (target: IEEE journal).

**Read first, in order:**
1. `docs/decisions.md`  - running log of decisions, findings and current status. Update it after every heavy piece of work.
2. `docs/RESEARCH_PLAN.md` - the approved plan (phases, baselines, metrics, risks).

Rules:
- Use the venv at `D:\drift\venv` (`..\venv\Scripts\python.exe`). Run tests with `pytest`.
- `legacy/` holds retired code kept for reference. Do not import from it.
- Never report a number from mock or unverified code. Known flaws are listed in `docs/decisions.md`.
- Commit in small steps on a feature branch, never directly to `master`.
