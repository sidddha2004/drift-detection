# legacy/

Retired code kept only for reference. Nothing here is imported by the project.

| File | Why retired |
|---|---|
| `drift_aware_fl_MOCK.py` | Mock training: loss = `np.var(features)`, accuracy = label-overlap heuristic + noise. Results from it are not real (audit A1). |
| `improved_adapter.py` | Near-duplicate of `src/adaptation/adapter.py`, imported nowhere. |
| `exploratory_scripts/` | Former `tests/test_0x_*.py`: plotting/analysis scripts with no assertions. Reusable for figures. |
| `CONTEXT_HANDOFF_old.md` | Pre-audit handoff. Several claims are outdated (e.g. references to deleted tests, "54K params / project complete"). |
