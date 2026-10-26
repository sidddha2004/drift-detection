# Decision & Progress Log

Persistent memory for this project. **Update after every heavy piece of work.**
Newest entries at the bottom of each section. Plan of record: `docs/RESEARCH_PLAN.md` (v0.2).

## 1. Fixed decisions (author-approved)

| # | Decision |
|---|---|
| D1 | The 4-way drift-type classifier (sudden / gradual / recurring / feature-level) is a **core contribution**. It must be rebuilt on real model-error streams and justified by an ablation against a type-agnostic adaptation policy. |
| D2 | Clustering is an **optional extension**, decided after Phase 5 results. |
| D3 | **PAMAP2 only**. A second dataset is deferred; state it as a limitation. |
| D4 | Baselines come from current published methods. Re-implement them on our backbone and also place results beside the numbers reported in the papers (flagging protocol differences). |
| D5 | Differential privacy / secure aggregation deferred to the end (Phase 6b). Until then claim **data locality only**. |
| D6 | **Native PyTorch simulator on the GPU** for experiments. Flower stays optional; byte accounting uses real serialised payload sizes. |
| D7 | Repo restructure and commits approved. Work on feature branches, small commits. |
| D8 | Statistics: 5 seeds (42, 123, 999, 2024, 7), bootstrap 95% CIs, Wilcoxon + Holm, hyperparameters tuned on S0 + a held-out validation schedule only. |

## 2. Audit findings still open (from the initial audit; ids used in the plan)

| id | Finding | Fix lands in |
|---|---|---|
| A1 | `drift_aware_fl.py` used mock training. **Retired to `legacy/` (Phase 0).** Real engine still to be written. | Phase 4 |
| A2 | No clustering code (extension, D2) | Phase 5 |
| A3 | No Page-Hinkley, entropy or MMD detector; only ADWIN on supervised error (ADWIN lives in `lstm_model.py`) | Phase 3 |
| A4 | Injector is label-preserving virtual drift only (constant offset). No real drift. | Phase 2 |
| A5 | Per-subject temporal train/test split on sequential protocol activities -> classes missing from train | Phase 1 |
| A6 | Running-EMA normalisation over the whole stream (leakage, masks offset drift) | Phase 1 |
| A7 | Majority-label windows with no purity filter (xfail test in `tests/test_windowing.py`) | Phase 1 |
| A8 | Drift re-sampled per round, no persistent stream / change-point manifest | Phase 2 |
| A9 | Detector/classifier/adapter not wired into any FL loop | Phase 4 |
| A10 | `LSTMDriftDetector` has no device handling (CPU only); GPU use needs the new engine | Phase 4 |
| A11 | `DriftInjector.get_drift_mask` uses `injection_point or default`, so an explicit 0 is ignored (xfail test) | Phase 2 |
| A12 | Drift classifier scored 69% overall and 0% on SUDDEN, only on synthetic error streams | Phase 3 (WP1b) |
| A13 | Modules use mixed import styles (`from src.x` and `sys.path` hacks such as `from drift_detection...`) | Phase 4 |

## 3. Facts verified (use these, not the docx)

- LSTM (9 -> 64 x2 -> 18) has **53,650 parameters** (~209.6 KB FP32). The plan docx's "77K / 308 KB" is wrong.
- ADWIN is `river` 0.23.0. Python 3.13.2. venv at `D:\drift\venv`.
- GPU: RTX 5050 Laptop, 8 GB, compute capability 12.0 (Blackwell). Needs **torch cu128 build**: now `torch 2.11.0+cu128`, verified `cuda.is_available()`; LSTM train step (batch 64, len 256) ~3 ms.
- PAMAP2: 9 `Protocol/` files; `Optional/` only has subjects 101, 105, 106, 108, 109.
- `docs/RESEARCH_PLAN.md` mentions `test_phase*.py`; those files no longer exist. The real tests are in `tests/`.

## 4. Progress log

### 2026-10-26 - Phase 0 (Foundations) - branch `phase-0-foundations`
Done:
- Snapshot commit `c65fccc` of the received state, before any change.
- Untracked `__pycache__` (19 files), rewrote `.gitignore`.
- Moved to `legacy/`: mock FL simulator, duplicate adapter, old handoff doc, 8 exploratory scripts (see `legacy/README.md`). Removed the mock from `src/federated/__init__.py`.
- Installed torch cu128 + pytest in the venv; pinned `requirements.txt`.
- New pytest suite: 18 passing, 2 strict xfails (A7, A11). Run: `..\venv\Scripts\python.exe -m pytest`.
- Fixed README (nonexistent paths), wrote `CLAUDE.md` pointing here.

Deliberately NOT done (and why):
- **Full directory re-layout from plan §8** (`src/data`, `src/detect`, ...). Moving modules that Phases 1-4 rewrite would only churn history. The layout migrates incrementally as each module is rewritten. Current package names stay until then.
- No config system or results schema yet. They are the first items of Phase 1, when their needs are concrete.
- Seeds/determinism utilities (set all RNGs, `torch.use_deterministic_algorithms`) -> Phase 1.

Next: Phase 1 (data & baseline lock): purity filter, stratified/randomised-block split, frozen normalisation fit on pre-drift train only, real E1/E2/E3 on stationary data, params/bytes table.
