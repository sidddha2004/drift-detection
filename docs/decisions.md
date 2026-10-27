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

| D9 | **8 clients, not 9**: subject 109 is excluded (a single 64 s rope-jumping bout, ~46 windows, 1 class). Subjects 101-108. |
| D10 | **Stationary control S0 = `blocked` split** (contiguous 16-window blocks assigned at random to train/val/test, purge between splits). The `temporal` split (test = last 25% of each bout) is kept as a **natural-drift probe**, not as the control. |
| D11 | **Locked baseline budget**: FedAvg 60 rounds x 2 local epochs; E1/E2 120 epochs; checkpoint chosen by mean client val macro-F1. The initial 30-round budget under-trained FedAvg (0.831 -> 0.851 at 2x). Use this budget for every method in later phases. |
| D12 | Model channels = the **9 correct +-16g accelerometer axes** (cols 4-6, 21-23, 38-40). Primary metric = macro-F1; report worst-client too. |

| D13 | **OPEN, needs author decision**: adopt a v2 data protocol that trims ~5 s from each bout edge? Recommended, because edge windows dominate the error and would cause spurious drift alarms in Phase 3. It changes the frozen baselines, so it would be a new versioned config and E1-E3b would be re-run (~50 min); the Phase 1 results stay frozen under tag `phase1-frozen`. |
| D14 | Analysis is pre-registered in `docs/analysis_plan.md` (primary endpoint, H1-H5, 10 seeds, Holm, non-inferiority margin 1 pt, separate tuning schedule). Change only via its Amendments section. |

## 2. Audit findings still open (from the initial audit; ids used in the plan)

| id | Finding | Fix lands in |
|---|---|---|
| A1 | `drift_aware_fl.py` used mock training. **Retired to `legacy/` (Phase 0).** Real engine still to be written. | Phase 4 |
| A2 | No clustering code (extension, D2) | Phase 5 |
| A3 | No Page-Hinkley, entropy or MMD detector; only ADWIN on supervised error (ADWIN lives in `lstm_model.py`) | Phase 3 |
| A4 | Injector is label-preserving virtual drift only (constant offset). No real drift. | done (Phase 2) |
| A5 | Per-subject temporal train/test split on sequential protocol activities -> classes missing from train | Phase 1 |
| A6 | Running-EMA normalisation over the whole stream (leakage, masks offset drift) | Phase 1 |
| A7 | Majority-label windows with no purity filter (xfail test in `tests/test_windowing.py`) | Phase 1 |
| A8 | Drift re-sampled per round, no persistent stream / change-point manifest | done (Phase 2) |
| A9 | Detector/classifier/adapter not wired into any FL loop | Phase 4 |
| A10 | `LSTMDriftDetector` has no device handling (CPU only); GPU use needs the new engine | Phase 4 |
| A11 | `DriftInjector.get_drift_mask` uses `injection_point or default`, so an explicit 0 is ignored (xfail test) | done (Phase 2; old injector retired) |
| A12 | Drift classifier scored 69% overall and 0% on SUDDEN, only on synthetic error streams | Phase 3 (WP1b) |
| A14 | **Wrong channel indices**: old code used columns 3/20/37 = IMU *temperature* (+ two acc axes per IMU, no z). **Fixed in new `src/data` loader** (old `src/data_pipeline` still has it; retire in Phase 2/4). Evidence: with the interleaved split, temperature-only gets macro-F1 0.836 and the old channel set (0.913) beats the correct accelerometers (0.874), i.e. temperature leaks the label via sensor warm-up. | done (new loader) |
| A15 | Temporal-vs-blocked test macro-F1 drops 6-8 pts for every method with no injection. **Phase 1b diagnosis: mostly NOT drift.** The first and last quarter of every activity bout are intrinsically hard (blocked-model error 0.17 / 0.14 vs 0.04-0.06 mid-bout: onset/offset/label-timing effects), and the temporal test set is exactly that last quarter. Genuine covariate shift is small (error 0.137 -> 0.175 at the same positions; train-vs-test MMD^2 2x higher). Class frequency (TV 0.009) and sensor warm-up (Spearman -0.06, p=0.52) are NOT causes. | handled: secondary sanity check only |
| A16 | Bout-edge windows have high error in every split; they inflate error in both S0 and temporal protocols and could trigger false drift alarms. Candidate fix: trim `trim_s` seconds at both ends of each bout (D13, open). | decision needed |
| A13 | Modules use mixed import styles (`from src.x` and `sys.path` hacks such as `from drift_detection...`) | Phase 4 |

## 3. Facts verified (use these, not the docx)

- LSTM (9 -> 64 x2 -> 18) has **53,650 parameters** (~209.6 KB FP32). The plan docx's "77K / 308 KB" is wrong.
- ADWIN is `river` 0.23.0. Python 3.13.2. venv at `D:\drift\venv`.
- GPU: RTX 5050 Laptop, 8 GB, compute capability 12.0 (Blackwell). Needs **torch cu128 build**: now `torch 2.11.0+cu128`, verified `cuda.is_available()`; LSTM train step (batch 64, len 256) ~3 ms.
- Correct accelerometer columns (0-indexed): hand 4-6, chest 21-23, ankle 38-40. Temperature: 3, 20, 37.
- 12-class label set (activity ids 1,2,3,4,5,6,7,12,13,16,17,24). 12-class LSTM = 53,260 params = 213,040 B FP32 (18-class head = 53,650).
- Data after Phase 1 pipeline: train/val/test windows (blocked) 8247/2404/3607 over 8 clients; class imbalance 5.7x (rope jumping smallest); clients lack classes (103 has 8, 104/106 have 10) = natural non-IID.
- Each subject does each activity once (stairs twice), so every bout is one long segment; within-bout splitting is unavoidable.
- Training speed (RTX 5050): one E1/E2/E3 run at the locked budget ~90-95 s.
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

### 2026-10-26 - Phase 1 (Data & baseline lock) - branch `phase-1-data-baselines`
Built: `src/data` (bout-based windowing, blocked/temporal splits with purge, train-only normalisation), `src/fl/engine.py` + `baselines.py` (GPU, E1/E2/E3), `src/eval` (metrics, bootstrap CI, Wilcoxon+Holm), `src/utils` (seed, config, result I/O with git hash), `configs/phase1_baselines.yaml`, experiments `phase1_{data_audit,baselines,channel_check,report}.py`. 46 tests pass (+2 xfail). Reports: `results/phase1/summary.md`, `data_audit.md`.

Locked-budget results, 5 seeds (42,123,999,2024,7), test split, mean macro-F1:

| method | blocked (S0 control) | temporal (natural-drift probe) |
|---|---|---|
| E1 local-only | 0.883 +- 0.003 | 0.815 +- 0.005 |
| E2 centralised | 0.871 +- 0.008 | 0.810 +- 0.004 |
| E3 FedAvg | 0.851 +- 0.015 | 0.774 +- 0.013 |

What this means / caveats (do not over-claim):
- The old "95% accuracy" is not reproducible under a leak-free protocol; realistic stationary macro-F1 is ~0.85-0.88. Old numbers must not appear in the paper.
- **Local-only (personalised) >= pooled/global models** on within-subject test data; FedAvg is worst and has a weak worst-client (0.68; client 103, which only has 8 classes, is the problem). So FedAvg is a *beatable* baseline, but the proposed system must be compared against personalised baselines (add FedAvg + local fine-tuning in Phase 6, already in plan as E3b/FedAvg+FT).
- With 5 seeds the minimum Wilcoxon p is 0.0625: no pairwise test can reach p<0.05. Plan: report CIs and effect sizes; consider 10 seeds for the headline comparisons in Phase 6.
- FedAvg temporal runs still select rounds 43-58 of 60 (not fully converged). Revisit if FedAvg becomes the main comparison.
- Val macro-F1 (0.87-0.93 blocked, 0.94-0.98 temporal) overstates test; never report val.
- Weights initialised/seeded deterministically (same seed => identical results, tested). GPU determinism relies on cuDNN deterministic flags.
- Results were produced at commit `4393dbd` + uncommitted config/runner edits (committed right after); numbers are in `results/phase1/*.json` with the full config.

Phase 1 exit gate: met (verified config + data audit table; E1-E3 reproduce across 5 seeds with std <= 0.015).

Next: Phase 2 (drift generator): real + virtual drift injectors, S0-S3 schedules as YAML, ground-truth change-point manifest, built on the `src/data` pipeline (so the old `data_pipeline` can be retired), stream visualisations and manifest tests. Also fix A11.

### 2026-10-27 - Phase 1 follow-ups: freeze, natural drift, FedAvg + fine-tuning
- Tag `phase1-frozen` = commit `a235e34` (code, configs, 5-seed results, audit). Do not change the protocol under that tag; use versioned configs.
- **Phase 1b** (`experiments/phase1b_natural_drift.py`, `results/phase1/natural_drift.{md,json,png}`): see A15/A16. Paper consequence: do **not** present the temporal split as natural drift evidence; present injected drift with ground truth as the main evidence, and mention mild natural covariate shift only as a secondary check.
- **Phase 1c** (`experiments/phase1c_fedavg_ft.py`, `results/phase1/fedavg_ft_*.json`): FedAvg part reproduces the frozen E3 numbers **exactly** (reproducibility check passed, both splits, 5 seeds).

| (blocked, 5 seeds) | macro-F1 | worst client | accuracy |
|---|---|---|---|
| E1 local-only | 0.883 +- 0.003 | 0.830 | 0.892 |
| E3 FedAvg | 0.851 +- 0.015 | 0.676 | 0.897 |
| **E3b FedAvg + local FT** | **0.888 +- 0.009** | 0.814 | 0.910 |

  E3b - E1 = +0.005, 95% CI [-0.002, +0.012]; temporal split: +0.0003. **Personalisation closes the entire FedAvg gap**: any claim of the proposed system must be made against E3b (H1), not E3. The fine-tuning stays local (no extra bytes) but costs ~15 s/run for 8 clients; 20-epoch cap was hit by several clients (minor, revisit if E3b becomes the key baseline).
- Open question for the author: D13 (edge trimming). Also still open: venue/deadline, E7 method list.

Next: Phase 2 (drift generator), after D13 is decided, since trimming changes the stream the injectors build on.

### 2026-10-27 - Phase 1d: edge-trim protocol (D13) - branch `phase-1d-trim-protocol`
Step log (in order):
1. Author's input (pasted advice, adopted): validate the rule first; justification must NOT be "improves macro-F1" but "edge windows are outside the evaluation target"; keep frozen results; versioned config; apply to all methods; check trimming does not remove meaningful drift.
2. Rule fixed a priori: trim 5 s (500 samples, ~2 windows) from both ends of every bout. Reason = PAMAP2 label alignment at transitions is approximate. Not tuned.
3. Code: `DataConfig.trim_samples` (default 0 = v1 behaviour unchanged; provenance start samples stay in raw bout coordinates). Test added. `configs/phase1_v2_trimmed.yaml`.
4. Validation (`experiments/phase1d_trim_validation.py`, `results/phase1/trim_validation.{md,json}`), no training:
   - 93-95% of windows kept per client; 0 client x class pairs lost.
   - Edge windows are far more atypical than interior ones (median centroid distance 2.34 vs 0.71, farther in 99% of 84 pairs). Supporting only.
   - Covariate content of retained windows preserved: MMD^2 v1 vs v2 mean 0.038 (< 0.071 blocked train/test reference); max 0.57 is a small-class outlier.
   - Coverage floor (train>=20, val>=5, test>=5) is breached by 5 pairs in v2 vs 4 in v1: 105/cls8 train 19, 105/cls11 test 2 (was 10), 107/cls4 train 12 val 3 test 3 (rare, short bouts). Block assignment is random so counts also shift by chance. Caveat to mention in the paper; per-client results for these classes are noisy.
5. Decision D13: **ADOPT v2 (trim 5 s) as primary protocol**, recorded as Amendment A1 in `docs/analysis_plan.md` before the re-run. v1 stays as a sensitivity analysis.
6. Next in this phase: re-run E1/E2/E3 (+E3b) under v2 (blocked + temporal, 5 seeds, same budget) -> `results/phase1d/`; compare to v1; confirm ranking; then Phase 2.
7. **v2 re-run done** (`results/phase1d/`: baselines + FedAvg+FT, blocked and temporal, 5 seeds, same budget, commit `53e5de9`, all configs carry `trim_samples: 500`). Comparison script `experiments/phase1d_compare.py` -> `results/phase1d/comparison.md`. Rule used (fixed before comparing): ranking change = sign of paired difference differs between protocols AND 95% CI excludes 0 in at least one protocol.
8. **Result: 0 ranking changes** in both splits. Test macro-F1 v1 -> v2 (blocked): E1 .883->.939, E2 .871->.930, E3 .851->.893, E3b .888->.935. Temporal: E1 .815->.895, E2 .810->.884, E3 .774->.862, E3b .815->.903. Every method rises 4-9 points: this is a protocol effect (hard edge windows removed), NOT a better model, and v1/v2 absolute numbers must never be mixed in one table.
   - Conclusions that survive: FedAvg clearly worst (CI excludes 0 in both); fine-tuning closes the gap (E3b - E3 about +0.04 in both protocols); E3b ~ E1 on the blocked split (tie in both protocols).
   - Nuance: in v2 temporal E3b beats E1 by 0.0085 (CI [0.003, 0.015]); in v1 it was a tie. Same sign, so not a ranking change, but shows E3b is at least as strong as local-only. The proposed system must still beat E3b.
   - Natural-shift gap (blocked - temporal, E1): v1 0.068 -> v2 0.044. Trimming removes about a third of it, the rest remains (consistent with A15: edges explain part, not all, of the gap). Still not presented as natural drift.
   - Worst-client rises as well (E3 blocked 0.676 -> 0.730) but client 103 (8 classes) remains the weakest for FedAvg.
9. Final: **v2 is the primary protocol from Phase 2 on; v1 = sensitivity table in the paper.** No cause analysis was needed because nothing changed.
10. Housekeeping: the wrapper scripts reported exit 127 for the temporal steps, but the result JSONs are complete (15 / 10 runs, 5 seeds, correct commit and config) so the runs are valid; first launch with `nohup &` was killed when its shell ended, re-run via tracked background jobs (`run_rest*.sh`). 
Next: Phase 2 (drift generator) on v2 streams.

### 2026-10-27 - Phase 2: drift generator - branch `phase-2-drift-generator`
Step log (in order):
1. Design (written before coding): time in rounds; events = profile (sudden/gradual/recurring) x mode (virtual/real) x transform (offset g / gain / rotation deg / label_remap) x channel subset x coverage x timing; type4 label derived by rule (virtual transform on a channel subset => feature_level, else the profile). Virtual transforms act in RAW g (de-normalise with frozen train stats, transform, re-normalise); real drift only relabels (disjoint class swaps), gradual real drift remaps a growing fraction.
2. Code: `src/drift/{schedule,transforms,stream}.py`; scenario YAMLs in `configs/scenarios/` (`severity.yaml`, S0 none, S1 sudden: 6 variants, S2 gradual: 4, S3 recurring: 3, S4 feature-level: 3). Manifest = pure function of (scenario, variant, severity, seed, clients, client classes); it stores per-client events and ground-truth change points (onset / offset / full).
3. Pools (A2 in analysis plan): train clean; val = arrival stream (detector/adaptation input); test = per-round evaluation pool. Phase 4 must decide whether adaptation needs its own validation split carved out of the arrival stream (val is only ~15% of windows, about 4 windows/round).
4. Tests: `tests/test_drift_generator.py`, 24 tests on synthetic clients (alpha profiles, change points, exact offset in raw g, rotation preserves |a| and other sensors, feature-level touches only target channels, real drift never touches inputs, gradual remap fraction grows, coverage 3/8, timing modes, determinism, 4 type4 labels all present, A11 explicit start 0). All pass.
5. Visual check: `experiments/phase2_drift_catalog.py` -> `results/phase2/stream_<scenario>.png` + `manifest_catalog.{md,json}` (seed 42, med). Schedules look as specified (partial = clients 101, 103, 107 for seed 42). The raw arrival-stream trace is noisy because each round has about 4 windows of mixed activities; the schedule heatmaps are the reliable view.
6. Severity calibration on **tuning seed 1001** (`experiments/phase2_severity_check.py`, `results/phase2/severity_check.md`): frozen local model per client, clean macro-F1 0.940. All 7 transform/scope cases are monotone low > med > high. First run (seed 42, rotation 15/30/60 deg) showed 60 deg collapses the model (0.12), so rotation levels were changed to 10/20/30 and the check re-run on the tuning seed (design calibration must not use evaluation seeds; recorded in Amendment A2). Mean macro-F1 at low/med/high, all sensors: offset 0.913/0.797/0.462, gain 0.911/0.813/0.656, rotation 0.880/0.678/0.421, real label swaps 0.761/0.584/0.420. Hand-only offset is nearly invisible at low/med (0.932/0.912): the subtle end for the detector tests.
7. Retired the old pipeline: `src/data_pipeline`, its 3 test files and `run_random_client_drift_fl.py` moved to `legacy/` (A4, A7, A11, A14 now closed). Suite: 60 passed, 0 xfail.
Caveats: severity is ordinal per transform (not comparable across transforms); virtual drift is applied at round granularity (constant alpha within a round); rotation is about the sensor x axis only; label_remap classes are chosen at random, not by activity similarity.
Next: Phase 3 detectors (ADWIN / Page-Hinkley / entropy-MMD, two-level, drift-type classifier trained on real error streams from these scenarios) - after the user's go-ahead.

### 2026-10-27 - Phase 3: detectors and drift-type classifier - branch `phase-3-detectors`
Step log (in order):
1. Plan written first and registered as Amendment A3 (`docs/analysis_plan.md`): tuning seeds 1001-1003, validation/classifier seeds 1004-1010, evaluation seeds untouched; signals, D1-D5 detectors, matching rules, 1 false alarm per 1,000 clean windows bound, typing protocol.
2. Key design insight (shapes the paper's claims): (a) *real* drift leaves P(x) unchanged, so label-free detectors cannot detect it by construction; (b) the first onset of a *recurring* drift looks exactly like a sudden drift at alarm time; only a re-occurrence can be recognised (needs episode memory). The old classifier's 0% on SUDDEN (A12) came from synthetic streams; both effects must be handled explicitly when rebuilding it. We report recurring_first separately instead of letting it hurt or flatter the accuracy.
3. Old `src/drift_detection/{drift_classifier,integrated_detector}.py` (about 1,300 untested lines, A3/A12/A13) are NOT reused; new package `src/detect/` is written test-first.
4. **Phase 2 stream corrected** (found while designing detectors; it is a bug fix of Amendment A2, not a new design): the arrival stream shuffled individual windows (i.i.d., unrealistically easy for detectors and contrary to the plan's "randomised-block order"), and `ceil` made it cycle and repeat windows. Now: `val` is split by contiguous 16-window blocks (alternating per class) into a clean `cal` pool (detector references, never drifted) and the `arrival` pool (random block order, time order kept inside a block); `per_round = floor(n_arrival / n_rounds)` so no window repeats (about 3 windows/round, ~180 per client). Tests updated (24 pass). The Phase 2 severity check used the test pool and is unaffected.
5. New package `src/detect/`: `signals.py` (frozen local model, per-window err/entropy/pred/window statistics; cal + arrival), `detectors.py` (PH with frozen reference, ADWIN via river, entropy PH z-scored per predicted class, class-matched MMD with a calibration null, two-level with stage-B binomial confirmation, cooldown runner), `evaluate.py` (matching rules of A3). `tests/test_detectors.py`: 14 tests (PH fires on increase only, label-free detectors use 0 labels, MMD detects covariate shift, stage B dismisses benign covariate shift, no sentinel => blind to real drift, matching/aggregation arithmetic).
6. Findings while testing (to remember for the paper): (a) river ADWIN with delta 0.01 cannot detect a 0.06 -> 0.43 error jump within 80 post-drift windows (Hoeffding bound too conservative at this stream length), so the grid goes up to delta 0.6; (b) the frozen local models are very accurate (stream error rate about 2%), so error-bit detectors have little signal: label-free signals matter.
7. Signals cache: `results/raw/phase3/` (git-ignored); 55 s per seed (training 8 models + 392 streams).
8. **Detector run 1 (grids v1; `results/phase3/detectors.log`)** - weak, and honest about it: validation recall (med severity, 7 unseen tuning-side seeds) D1 0.20, D2 0.37, D3 0.48, D4 0.26, D5 0.31; FA/1000: 0.25, 0.37, 4.02 (bound 1.0 violated), 3.01 (violated), 0.91. Findings: (a) several optima sat on the grid edge (D3 lam 40, D4 q .999/persist 3) and D3/D4/D5 were infeasible under the bound; (b) recall of label-free detectors on REAL drift (0.16-0.25) is just chance: with FA about 4/1000 an alarm lands in a 60-window horizon by luck about 20% of the time, so a *placebo recall* (same onsets on the clean S0 stream) is needed to read any recall number; (c) a sentinel with 20% labels did not help (D5 = D5ns on virtual and slightly better on real); (d) activity-block structure makes error and entropy bursts (hard classes) the main false-alarm source.
9. **Design iteration after run 1 (disclosed).** Changes: class-conditioned error baseline (smoothed per-predicted-class calibration error rate) used in a new detector D2c and in stage B (Poisson test instead of a global binomial); grids extended in the direction of the edge; placebo recall added to the report. **Seeds 1004-1010 were looked at before these changes, so they are no longer a pristine validation set for the Phase 3 design; this is stated in the paper. The 10 evaluation seeds are untouched.**
10. **Detector run 2 and 3 (grids v2, D2c, D2s added; `results/phase3/detector_report.md`, earlier version kept as `detector_report_run2_before_D2s.md`).** Final validation (7 unseen tuning-side seeds, med severity): D2 PH-on-error is the best detector (recall 0.42, placebo 0.07, FA 0.39/1000, delay 9.2 rounds, needs 1000 labels/1000 windows). D2s (20% labels) recall 0.26 at 200 labels. Label-free D3/D4 are barely above chance (+0.09/+0.10) and violate the FA bound on validation (1.24, 1.53). **The two-level D5 is NOT better than a plain sparse-label detector at a similar label budget (0.19 at 376 labels vs 0.26 at 200)**; the sentinel adds almost nothing. Class-conditioning the error baseline (D2c) changed nothing. => RQ1 answer in this benchmark: supervised error signals give the best delay/false-alarm trade-off; the label-efficiency claim for a two-level detector is not supported and must not be made in the paper (or must be reported as a negative result).
11. **Drift typing (`results/phase3/typing_report_la5.md`, v1 without rise features in `typing_v1_before_rise_features/`).** 4-way typing with detected alarms (D2): gboost accuracy 0.46 / macro-F1 0.32 vs majority 0.51 / 0.17; rule classifier 0.38 / 0.25; oracle alarms gboost 0.63 / 0.59. Per-class recall (detected, gboost): sudden 0.72, gradual 0.22, recurring 0.19, feature-level 0.13. Caveats found: (a) `n_prior` was an artefact in oracle streams and was removed from the learned models, but the memory features still act as "an earlier alarm existed", so oracle recurring recall 1.00 is a shortcut; (b) a causality test caught future leakage in the smoothing of a new feature (fixed; test kept); (c) 10-round look-ahead and retrospective rise/step-vs-ramp features gave no gain, and iteration was stopped to avoid fishing on the tuning seeds. Likely causes: late alarms (a 10-round ramp is often finished when the error detector fires), weak per-sensor evidence at med severity, short streams.
12. **Choices made after seeing validation numbers (disclosed):** D2 was selected as the alarm source for typing because it is the best validated detector; seeds 1004-1010 are not pristine for the Phase 3 design (see item 9).
13. **Consequence for the plan (needs author's view):** typing accuracy alone does not currently justify the 4-way classifier. The decisive test stays the downstream one in the plan (typed response vs best fixed response, H5). Options: (A) go on to Phase 4 and let H5 decide; (B) first make the benchmark easier/longer for typing (more rounds, longer pre/post phases, higher severities) and re-run Phase 3; (C) reframe typing as a coarse 2-way decision (abrupt vs gradual) if the 4-way stays unidentifiable. Recommendation: A, with B's longer streams if Phase 4 shows the stream length is the bottleneck.
