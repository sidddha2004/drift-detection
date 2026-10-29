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

### 2026-10-27 - Decision D15 (author): Phase 3 follow-up = option A
Author chose option A: proceed to Phase 4 and let the downstream test (typed response vs best fixed response, H5) decide whether the 4-way drift typing earns its place. Option B (longer/easier streams) is reserved if Phase 4 shows stream length is the bottleneck; option C (coarse typing) if the 4-way stays unidentifiable downstream. Branch `phase-4-streaming-fl`.

### 2026-10-27 - Phase 4: streaming FL engine and adapter - branch `phase-4-streaming-fl`
Step log (in order):
1. Decision D15 recorded (option A). Design: deployed model per client = E3b personalised model (FedAvg 60 rounds x 2 epochs + local fine-tune, checkpoints chosen on the CLEAN calibration pool so no arrival window influences selection); stream = Phase 2 manifest, 60 rounds, ~3 labelled arrival windows per client per round; per-round evaluation of every client on its held-out TEST pool at that round's drift strength (primary endpoint = post-change macro-F1, analysis plan section 2).
2. `src/fl/adapter.py` AdaptedLSTM: input recalibration (9x9 + 9) -> frozen LSTM -> residual bottleneck (r = 8) -> head copy. **1,966 trainable parameters = 7,864 B FP32 = 3.7% of the full model (53,260 params, 213,040 B)**; identity/zero initialisation reproduces the base model exactly. 7 tests.
3. `src/fl/streaming.py` engine: policies = trigger (none / oracle onset+2 / PH-on-error detector (tuned D2: delta 0.05, lambda 8) / periodic) x response (full fine-tune, adapter, input, head, align) x federated or local; byte accounting from the serialised payload; adaptation episode of R rounds on the recent labelled buffer; federated responses average the participants of the round only and send the average back to them. `src/eval/streaming.py` metrics (post-change F1, pre-drift F1, max drop, recovery rounds, off-phase F1, clean-client F1, bytes). 9 engine tests on synthetic clients (frozen invariance, oracle timing, unaffected clients untouched, byte accounting for full / adapter / periodic, determinism, metric definitions).
3b. Caveat for Phase 3: its frozen local models were selected on the full `val` pool (cal + arrival), so arrival windows influenced checkpoint selection (slightly optimistic stream error rate). Phase 4 fixed this (selection on cal only). Phase 3 detector numbers are therefore tuning-side only, as already stated.
4. **Findings from smoke runs (seed 1001, S1, med) that changed the design:**
   - Fine-tuning on the recent arrival buffer COLLAPSES the model (0.50-0.57 vs 0.89 frozen): the stream arrives in activity blocks, so the buffer holds 2-4 classes. => replay of clean train windows added (ratio 1), and for real drift replay of only the classes absent from the buffer.
   - Gradient training of the input layer fails (the model is already right on the seen classes, so cross-entropy carries almost no signal about the sensor transform; replay pulls the affine map halfway back to the identity).
   - => `src/fl/align.py`: **closed-form input recalibration**. Virtual drift is a class-independent transform of the inputs, so per IMU a ridge-regularised least-squares fit of out = m'A + t between the drifted buffer's class-conditional mean sensor vectors and the clean train-pool class means (bias unregularised, A pulled to the identity) recovers offset, gain and rotation from a few labelled windows, needs no gradient steps and no communication. 10 unit tests with known transforms.
   - Oracle-triggered results (seed 1001): gain 0.754 frozen -> 0.886 align; rotation 0.643 -> 0.798; offset 0.894 -> 0.869 (frozen is already close to pre-drift 0.942, the fit adds noise). Real drift stays hard: continual head adaptation with unseen-class replay 0.681 vs 0.630 frozen (only classes already seen after the change can be relabelled).
   - With the PH detector as trigger most of the gain is lost (alarms 1 of 3 drifted clients on gain/offset), consistent with Phase 3 (recall 0.42).
5. First-look experiment launched (3 tuning seeds; S1 virtual: offset partial / gain / rotation / offset full; real partial) -> `results/phase4/s1_*_first_look.json`.
6. **S1 first look (tuning seeds 1001-1003, med severity; `results/phase4/s1_*_first_look.json`, tables in `results/phase4/summary.md`, curves `curves_s1_*.png`).** Post-change macro-F1 on drifted clients, frozen E3b personalised model as the passive baseline:

| variant | frozen E3b | periodic E3c (full, fed) | oracle full fed | oracle adapter fed (CE) | **oracle align** | PH + full fed (E5) | **PH + align** |
|---|---|---|---|---|---|---|---|
| gain, partial | 0.762 | 0.720 | 0.775 | 0.795 | **0.884** | 0.744 | **0.780** |
| rotation, partial | 0.682 | 0.688 | 0.688 | 0.692 | **0.845** | 0.672 | **0.752** |
| offset, partial | 0.892 | 0.844 | 0.840 | 0.885 | **0.909** | 0.869 | 0.879 |
| offset, full | 0.873 | 0.843 | 0.820 | 0.871 | **0.880** | 0.850 | 0.867 |
| real, partial | 0.622 | 0.610 | 0.590 | - | head_long **0.675** | 0.624 | ph_head_long 0.666 |

   Cost (uplink, 3 clients unless noted): full-model fed retraining 1.3-5.1 MB per run, periodic E3c 8.5 MB (all clients, every 10 rounds), adapter (CE) 24-189 kB, align and head_long 0 B (local only).
   Reading (descriptive, 3 seeds, no test):
   - The closed-form input adapter ("align") is the only response that clearly beats the frozen personalised model on virtual drift, with an oracle trigger by +12 (gain) and +16 (rotation) points; with the PH detector as trigger the gain is +2 and +7 (detection recall 0.42 is the bottleneck, as predicted by Phase 3). On offset drift the frozen personalised model is already near its pre-drift level, so nothing helps much.
   - Full-model federated retraining after a trigger (the standard "drift-aware FL" response) is NOT better than doing nothing here (0.59-0.85, often below frozen), and costs 18-70x the bytes of the adapter; gradient-trained adapters ("adapter fed") are about equal to frozen. Periodic retraining of all clients hurts clean clients (0.88 vs 0.95).
   - Real drift stays hard (best +5 points: continual head adaptation with replay of unseen classes, local).
   - Hence the H4 (adapter vs full retraining) claim holds in a stronger form than planned: the adapter is not just cheaper, it is better; but note the adapter that works is the closed-form recalibration, not federated gradient training, and it is local: its communication saving is 100% because it needs no federation at all. A fair bandwidth framing must say so.
   - Caveat for the paper: the closed-form fit assumes immediate labels (class-conditional means) and a clean per-class reference on the client (train pool); both are stated assumptions.
7. **Typed (oracle) vs fixed responses, S2-S4 (`typed_virtual`, `typed_real`, `mode_typed` JSONs; `summary.md`).** Typed = ground-truth type from the manifest drives the response: real -> continual head adaptation; virtual sudden -> one-shot align; gradual -> tracking align (sliding buffer); recurring -> cache the fit, revert to the identity map when the drift ends, restore it on recurrence; feature-level -> align restricted to the affected sensor. Post-change macro-F1, 3 tuning seeds, med severity:
   - vs the best fixed policy chosen PER VARIANT (favours the fixed side): typed - best fixed = +0.011 on average over 9 variants (range -0.004 to +0.047). Single fixed policies fail across modes: align-only collapses on real drift (S3 real 0.552 vs frozen 0.784), head-only/full-fed do not fix virtual drift. So "one fixed response for all drifts" loses clearly; the oracle gap to a per-variant best fixed is small because the per-variant best often equals the typed choice.
   - **What the 4-way type adds beyond the 2-way mode (virtual/real), both oracle:** +0.033 post-change F1 on average over 9 variants (S2 gradual tracking +0.04, S3 recurring revert/restore +0.04 / +0.095 on rotation, S4 sensor restriction +0.015..0.02). Typing therefore has a real ceiling value; the open question is how much survives when types come from a classifier.
   - S3 recurring: fixed one-shot align keeps the stale map after the drift ends (off-phase F1 0.62-0.85 vs 0.945 clean), typed revert keeps 0.90-0.93: forgetting/retention is where typing pays.
8. **Mode (virtual vs real) can be identified from a detected alarm** (`experiments/phase3_mode_typing.py`, `results/phase3/mode_typing.md`; 927 episodes, leave-one-seed-out, label-free features): gradient boosting accuracy 0.845 (majority 0.613, rule on covariate shift 0.655, logistic regression 0.731); recall virtual 0.92 / real 0.72; accuracy rises with severity (0.72 / 0.82 / 0.89). The 4-way typing stays weak (item 11 of Phase 3), so the deployable system types the MODE only, with the typing delay of 5 rounds after the alarm.
9. Deployable typed policy `typed_learned` (PH detector -> episode features after 5 rounds -> mode classifier trained on Phase 3 episodes of seeds 1004-1010, disjoint from the stream seeds 1001-1003 -> mode-typed response) and `ph_mode_oracle` (true mode, to price the classifier) were added to the engine (`Policy.typer`, `typing_delay`); runs in progress (`results/phase4/learned_*.json`). Caveat: the classifier was trained on episodes of Phase 3's local models, applied to signals of the Phase 4 personalised models; label-free covariate features dominate, so the shift should be small but is not tested.
10. **Deployable systems (detector-triggered, immediate labels; 3 tuning seeds, 14 variants S1-S4 at med severity; `results/phase4/summary.md`, `learned_*.json`).** Mean post-change macro-F1 over the 13 variants common to all columns:

| frozen E3b | PH + full fed (E5-like) | PH + align | PH + head_long | **typed_learned (mode classifier)** | typed_learned conf>=0.8 | PH + oracle mode |
|---|---|---|---|---|---|---|
| 0.801 | 0.794 | 0.789 | 0.805 | **0.807** | 0.806 | 0.812 |

   Reading (descriptive; 3 seeds; nothing is a test): in deployable form every system is within about 1 point of doing nothing. The full system gains +3 to +6 points on rotation / gain / sensor variants (S1 rotation 0.682 -> 0.735, S2 rotation 0.734 -> 0.783, S4 chest rotation 0.733 -> 0.795, S4 ankle gain 0.724 -> 0.759) and loses 1 to 7 points elsewhere (S3 real 0.784 -> 0.714, S3 rotation 0.814 -> 0.793). Type-agnostic align is harmful on real drift (S2 real 0.541, S3 real 0.570 vs 0.656 / 0.784 frozen); type-agnostic head adaptation is harmless but fixes nothing on virtual drift. A confidence gate on the mode classifier changed nothing (0.806), and even the oracle mode only reaches 0.812: the remaining loss is not the classifier.
11. **Why the deployable gain is small: detection.** With an oracle trigger (onset + 2 rounds) the same align response gives +12 (gain) and +16 (rotation) points on S1; with the PH detector (recall 0.42, mean delay 9 rounds) plus 5 rounds of typing delay, the gain shrinks to +2 / +7, and a response that starts at round ~35 of 60 only has ~25 rounds to pay off. Also in recurring scenarios (S3) a persistent head adaptation damages the off-phase. => H1 (E8 > E3b) is NOT supported in deployable form on these streams; the ceiling shown by oracle triggers is large. This is the main result of Phase 4 and must be reported as it is, with the oracle results labelled as upper bounds.
12. **Phase 4 exit gate:** detector wired into FL: yes; first recovery curves on S1: yes (`curves_s1_virtual.png`, `curves_s1_real.png`); adapter vs full retraining preliminary: yes (adapter-style response better and 18-70x cheaper in bytes on virtual drift, full-model federated retraining no better than frozen). 82 + 30 = 112 tests (engine 13, adapter 7, align 10, plus earlier).
13. **Recommendation / decision needed from the author (D16):** the confirmatory question is currently limited by detection and stream length, not by the response. Options: (A) revise the benchmark before any evaluation-seed run: longer streams (e.g. 120 rounds, onset at round 30) so that detection delay and typing delay no longer eat the post-drift period, optionally also a higher-recall detector or sparse-label supervision; register as Amendment A4 (evaluation seeds untouched); (B) go straight to Phase 6 with the current benchmark and report the small deployable gain honestly; (C) Phase 5 (clustering extension) first. Recommendation: A, then Phase 6. Phase 5 only if worst-client results show a need.

### 2026-10-28 - Decision D16 (author): option A; benchmark v2 - branch `phase-4b-benchmark-v2`
Author chose option A (revise the benchmark: longer streams, then Phase 6). Registered as Amendment A4 in `docs/analysis_plan.md` BEFORE any v2 code or run. Also asked by the author: has ADWIN been tried? Answer logged: Phase 3 yes as a detector (D1, recall 0.20), Phase 4 no (E4 missing) -> E4 (ADWIN-triggered full-model retraining) is added to the v2 runs.
Plan: (1) code: `passes` (two arrival passes), `configs/scenarios_v2/`, bench path helper, ADWIN trigger in the engine, tests; (2) Phase 3 v2: signals, detector tuning/validation, mode classifier; (3) Phase 4 v2 runs on tuning seeds, report; (4) report to the author.
1. Code for benchmark v2 (all behind `DRIFT_BENCH=v2`, v1 untouched): `ClientStream(passes=2)` replays the arrival pool in a new random block order per pass (about 3 windows/round, 360 arrival windows over 120 rounds; calibration pool unaffected); `configs/scenarios_v2/` (S1/S4 onset 30; S2 async onsets 30-40, ramp 10; S3 period 30, duration 15 -> onsets 30/60/90, offsets 45/75/105; severities unchanged); `src/utils/bench.py` keeps caches/results apart (`results/raw/phase3_v2`, `results/phase3_v2`, `results/phase4_v2`); streaming engine got an ADWIN trigger (E4) next to PH (E5) and policies take the TUNED detector settings from the benchmark's detector report instead of constants. Tests: 43 (+2 generator, +2 engine) pass.
2. Pipeline launched in the background (`results/phase4_v2/run_v2_pipeline.sh`): signals for the 10 tuning-side seeds -> detector tuning/validation (same grids, bound and matching rules as A3) -> mode classifier -> 3 parallel batches of streaming runs (10 variants, 10 policies incl. E3c, E4, E5, typed, oracle upper bounds; seeds 1001-1003).

### 2026-10-28 - Session paused by the author (resume tomorrow)
State at pause: branch `phase-4b-benchmark-v2` (code + Amendment A4 committed). The v2 pipeline (`results/phase4_v2/run_v2_pipeline.sh`) was started in the background: v2 signals done (10 seeds) -> detector tuning/validation (`results/phase3_v2/`) -> mode classifier -> 3 parallel streaming batches `v2_A/B/C` (JSON written only when a batch finishes; completion marker `results/phase4_v2/V2_DONE`).
To resume: check whether `V2_DONE` exists. If yes: `set DRIFT_BENCH=v2` and run `experiments/phase4_report.py` (add the v2 tags `v2_A v2_B v2_C` to FILES), compare with v1 (`results/phase4/summary.md`), log the result, then report to the author. If the batches were interrupted: re-run `results/phase4_v2/run_v2_pipeline.sh` (signals are cached; detector tuning is minutes; each batch takes about 1.5 h).
Open: after v2 -> decide Phase 5 (clustering, optional) vs Phase 6 (baselines/ablations/10 evaluation seeds); venue/deadline; E7 method list.
Pause update: at the author's request the v2 streaming batches (`v2_A/B/C`) were stopped before finishing; no v2 streaming result file exists (they are written at the end of a batch). Already done and kept: v2 signals (10 seeds), v2 detector tuning/validation (`results/phase3_v2/detector_report.md`) and v2 mode-typing (`results/phase3_v2/mode_typing.md`) - not yet read or logged. To resume: re-run `results/phase4_v2/run_v2_pipeline.sh` (it re-does the quick detector steps, then the 3 batches, about 1.5 h in parallel), or run the three `phase4_streaming.py` commands from it directly.
Note: a `V2_DONE` marker file appeared when the batches were killed (the script's `wait` returned) although no batch finished; it was deleted. Trust only the existence of `results/phase4_v2/v2_A.json`, `v2_B.json` and `v2_C.json`.

### 2026-10-28 (resumed) - benchmark v2 results so far
1. **v2 detector validation (`results/phase3_v2/detector_report.md`; same grids/bound/matching rules as A3; tuned on 1001-1003, validated on 1004-1010):** D2 PH-on-error recall 0.42 (placebo 0.05), delay 8.6 rounds, FA 0.48/1000 - practically identical to v1 (0.42 / 9.2 / 0.39); **D1 ADWIN recall 0.28 (v1 0.20)**, delay 14.5 rounds, FA 0.20/1000 (still weaker and slower than PH); D5 two-level 0.32 at 454 labels/1000 windows vs D2s (sparse-label PH) 0.34 at 200 labels: the two-level design is again not better than a plain sparse-label detector. Label-free D3/D4 remain barely above chance (+0.12/+0.09) and violate the bound. Tuned D2 = (delta 0.05, lambda 8), D1 delta 0.3 (same as v1). Recall within the fixed 20-round horizon does not improve with longer streams, as expected; the benefit of v2 can only appear in the post-change F1 averaged over ~90 rounds.
2. **v2 mode classifier** (`results/phase3_v2/mode_typing.md`, 1099 episodes, leave-one-seed-out): gradient boosting label-free 0.839 accuracy (all features 0.857) vs majority 0.602 - unchanged from v1 (0.845).
3. The three v2 streaming batches were restarted (`results/phase4_v2/run_v2_batches.sh`); completion marker `V2_DONE` is now only written if all three result files exist.

## 2026-10-28 - benchmark v2 streaming results (seeds 1001-1003, 10 variants, post-change macro-F1, mean over variants)
(The wrapper script was stopped by a memory-pressure reaper; v2_A/B/C JSONs were all written regardless. Report: `results/phase4_v2/summary.md`.)
frozen E3b 0.735 | E3c periodic full-fed 0.737 (18.7 MB uplink) | E4 ADWIN full-fed 0.735 (2.7 MB) | E5 PH full-fed 0.730 (4.8 MB) | ph_align 0.725 | ph_head_long 0.777 | typed_learned (deployable, mode classifier) 0.783 | ph_mode_oracle 0.808 | oracle_align 0.728 | typed_oracle 0.827. All typed/adapter responses cost 0 uplink.
Reading: deployable typed system +4.8 pts over frozen E3b (v1: ~+0.6); full-model federated retraining (E3c/E4/E5) gives no gain over frozen at 4-25x the bytes; fixed ph_align alone is NOT better than frozen (hurts real drift: 0.42-0.54), the gain comes from choosing alignment for virtual and head retraining for real drift. Not confirmatory (3 tuning seeds); per A4 no further benchmark change before Phase 6.

## 2026-10-28 - Phase 6 start (branch phase-6-confirmatory)
1. Amendment A5 registered (analysis_plan.md): systems, fixed comparators per hypothesis, E7 = our FedProx-style re-implementation (E7a) + E5 as FedDrift-style analogue (not the official code), E8 = typed_learned, deployable typing is 2-way mode; 4-way is an ablation.
2. E7a proximal strength tuned on seeds 1001-1003 (S1 rotation + real, post-change F1, mean of the two): mu 0.01 -> 0.666, 0.1 -> 0.671, 1.0 -> 0.695 (E5 0.659, frozen E3b 0.652). mu = 1.0 fixed (best, but at the edge of the registered grid; the grid is not extended). E7a gain over frozen is small.
3. Not yet run: confirmatory batches on the 10 evaluation seeds (about 5 h, 2 parallel processes) - waiting for the user's go-ahead on launch.

## 2026-10-28 - Phase 6 confirmatory results (10 evaluation seeds, benchmark v2; `results/phase6/confirmatory.md`, script `experiments/phase6_stats.py`)
Run exactly as registered in A5 (S0 streaming not run: runner has no S0 support; false alarms from Phase 3 detector validation). Holm over 12 tests.
- H1 (E8 vs E3b): S1 +0.079 (Holm p 0.023, significant); S2 +0.080 (CI excludes 0, Holm p 0.078, NOT significant); S3 -0.050 (significantly WORSE).
- H2 (E8 vs E3c): +0.046 / +0.057 / -0.033; none significant after Holm.
- H3 (E8 vs E7a FedProx): +0.018 / +0.023 / -0.052; only S3 significant and in E7a's favour. H3 not supported.
- H5 (E8 vs best fixed ph_head_long): +0.004 / -0.016 / -0.055; typed deployable does NOT beat the best fixed response (S3 significantly worse). H5 not supported for the deployable system.
- H4: federated adapter vs full retraining non-inferior (CI upper < +0.010) in S1-S3 with 96% uplink reduction (127 vs 3387 kB in S1).
- Typing ablations: oracle 4-way beats oracle 2-way in S2 +0.045, S3 +0.066, S4 +0.052 (S1 0); deployable E8 loses 2.3-5.3 pts vs E8 with true mode (S1, S2 significant uncorrected).
- Reading: gains over passive FL hold for sudden/feature drift (S1, S4), not for recurring drift (S3), where the 2-way typed system lacks memory/revert and aligns where it should not; E8's seed variance is high (std 0.05-0.08). The cheap-adaptation claim (H4) holds; the typed-beats-fixed claim (H5) does not.

## 2026-10-29 - Decision D17 (author): option B after Phase 6 - stop experiments, write the paper
No further experiments (no recurring-drift memory fix, no severity sweep, no clustering, no S0 streaming run). Paper draft: `docs/paper/main.tex` (IEEEtran, journal) + `refs.bib`; figures from `experiments/phase6_figures.py` (`docs/paper/figs/`). Every number comes from `results/phase6/confirmatory.md`, `results/phase3_v2/*`, `results/phase1d/comparison.md`. Positive claims: H4 (adapter non-inferior, 96% less uplink), H1 on S1 (and S2 CI only), oracle 4-way typing value. Negative results reported: H2, H3, H5 not supported, S3 loss, weak 4-way typing from detected alarms, two-level detector no better than sparse-label PH, label-free detectors ~chance. Caveats written into the paper: E7a/E5 are our re-implementations (not official FedProx/FedDrift), seeds 1004-1010 not pristine, comparators of H3/H5 chosen from tuning results, S0 streaming not run. NOT done: LaTeX not compiled (no TeX installed here); bibliography unverified (entries from memory, flagged in refs.bib and in the paper text); author block, venue and page limit open.

## 2026-10-30 - Paper: bibliography verified, Related Work expanded
1. The 16 references of the plan document (`../Federated Drift Research Paper Plan.docx`, citation table) plus the 7 already cited were checked against Crossref / PMLR / NeurIPS / MLSys / JMLR / arXiv; `docs/paper/refs.bib` replaced by the verified version (source URL above every entry). Plan errors corrected: FedStream DOI is 10.1109/TSMC.2023.3293462 (not ...3292910), FLACC DOI is 10.1109/TII.2023.3252599 (not ...3252536), Chen et al. "Virtual concept drift ..." is IJDSA 21(1) art. 46, 2026 (online Dec 2025). Spot-checked the three corrected DOIs on Crossref.
2. Related Work rewritten into four paragraphs (drift; FL under heterogeneity incl. FedNova, CFL, FLACC; FL under concept drift incl. FDAL survey, FedDrift, Flash, FedConD, CDA-FedAvg, FedStream, Chen et al. virtual drift; federated HAR incl. FedHAR, Aouedi survey). The "unverified bibliography" disclaimer was removed. All 21 keys cited, none missing. LaTeX still not compiled here.
3. The 5 extra 2024-2026 references of the plan were verified and added (autocfl2024 = Gong et al. TBD 2024 "Adaptive Client Clustering ..."; jiang2026cm = X. Jiang, MDM 2026 3-page overview, single author, not "Jiang & Borcea"; wang2026falcon = malware-detection FAL, TMC 2026; land2026 = TCE 2026 early access, abstract not retrieved, cited by title only; driftguard2026 = Alnaim & Alwakeel, Electronics 2026, encrypted traffic, simulation only, ADWIN role unconfirmed). Cited in Related Work with claims limited to what was confirmed. 26 keys, all cited.
4. Paper figures from cached results only (no training, no streaming reruns; seconds): `experiments/paper_figs/` (`figures.py`, `run_all.py`, `test_figures.py`, 5 tests pass). Run `python experiments/paper_figs/run_all.py` -> `docs/paper/figs/`: forest_confirmatory, paired_seeds, heatmap_policy_variant, s3_offphase, collateral_worst_client, alarm_delay, detectors_tradeoff, mode_classifier, typing_ladder, cost_recovery + summary_stats.md. summary_stats.md reproduces Tables I/II of main.tex exactly. New descriptive number: E8 alarms reach 262/300 drifted clients at ANY time after onset (median delay 23 rounds) vs recall 0.42 within the 20-round matching horizon of A3. The figures are not yet referenced in main.tex.
5. Four of the new figures added to main.tex with captions and in-text references: detectors_tradeoff (Detection and typing), forest_confirmatory and s3_offphase (Confirmatory results / H1), typing_ladder (What typing is worth). All figure paths and \ref labels checked; LaTeX still not compiled.
