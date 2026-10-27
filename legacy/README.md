# legacy/

Retired code kept only for reference. Nothing here is imported by the project.

| File | Why retired |
|---|---|
| `drift_aware_fl_MOCK.py` | Mock training: loss = `np.var(features)`, accuracy = label-overlap heuristic + noise. Results from it are not real (audit A1). |
| `improved_adapter.py` | Near-duplicate of `src/adaptation/adapter.py`, imported nowhere. |
| `exploratory_scripts/` | Former `tests/test_0x_*.py`: plotting/analysis scripts with no assertions. Reusable for figures. |
| `CONTEXT_HANDOFF_old.md` | Pre-audit handoff. Several claims are outdated (e.g. references to deleted tests, "54K params / project complete"). |

## Retired in Phase 2 (2026-10-27)
- `data_pipeline/` (preprocessing, windowing, drift injection): wrong channel columns (A14), majority-label windows (A7), `injection_point or default` bug (A11), constant-offset virtual drift only (A4). Replaced by `src/data` (bout windowing, leak-free splits) and `src/drift` (schedules, real + virtual drift, manifests).
- `tests/` : tests of the old pipeline, including the strict xfails for A7 and A11 (both are now covered by `tests/test_data_federation.py` and `tests/test_drift_generator.py`).
- `exploratory_scripts/run_random_client_drift_fl.py`: depended on the old pipeline.
