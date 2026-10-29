# Research Plan: Drift-Aware Federated Human Activity Recognition on Wearable Streams

| | |
|---|---|
| **Status** | v0.2, author decisions incorporated (Section 7); awaiting go-ahead to start Phase 0 |
| **Target venues** | IEEE TMC / IEEE IoT-J / IEEE TSMC:Systems (journal); PerCom / ISWC (conference fallback) |
| **Dataset** | PAMAP2 (9 subjects, 100 Hz IMU), Protocol + Optional |
| **Compute** | Single workstation, RTX 5050 GPU |
| **Basis** | `Federated Drift Research Paper Plan.docx` plus an audit of `fl-drift-try_2` (Section 2) |

---

## 1. Research Framing

### 1.1 Problem
Wearable HAR models are trained on past data, but user behaviour, sensor placement and sensor calibration change over time. In federated learning (FL) this happens independently on each client, at different times. Two naive options exist:
- **Passive FL:** retrain and upload the full model every round. This wastes bandwidth and energy, and recovers slowly after a shift.
- **Isolated local adaptation:** this forgets old activities and does not share what other clients have learned.

### 1.2 Central hypothesis
> Detecting drift locally and then adapting only a small part of the model (adapter plus head), with clustering restricted to clients that share a drift concept, restores post-drift accuracy comparably to full retraining. It does so at a fraction of the communication and compute cost, and without catastrophic forgetting.

### 1.3 Research questions
| ID | Question | Answered by |
|---|---|---|
| RQ1 | Which local detection signal (supervised error, label-free entropy/MMD, or two-level) gives the best delay / false-alarm trade-off under realistic label availability? | Phase 2, 6 |
| RQ2 | Does drift-triggered adaptation beat passive FL and published federated-drift methods on accuracy recovery and bytes sent? | Phase 5 |
| RQ3 | Does adapter-only adaptation match full retraining (within 1 pt macro-F1) with >70% fewer uplink bytes and less forgetting? | Phase 5, 6 |
| RQ4 | Does drift-*type* classification plus type-specific adaptation outperform a single type-agnostic adaptation policy? | Phase 3, 4, 7 |
| RQ4b | *(Extension)* Does drift-gated, step-normalised clustering avoid degenerate clusters and improve worst-client performance? | Phase 5, 7 |
| RQ5 | What is the runtime and memory overhead of monitoring on client hardware? | Phase 6 |

### 1.4 Contributions we can defend (claim only what the ablations support)
1. **Drift-type-aware federated adaptation (primary novelty).** Existing federated drift methods treat drift as a single phenomenon. We detect, *classify* (sudden / gradual / recurring / feature-level) and apply a type-specific response, and we show the classification step pays for itself against a type-agnostic response.
2. A two-level, label-efficient drift detection pipeline with an explicit supervision model (immediate vs delayed labels).
3. Event-triggered, adapter-only federated adaptation with a measured bandwidth and forgetting analysis.
4. *(Extension, kept only if results justify it)* Drift-gated clustering with step-normalised updates, plus a diagnosis of singleton-cluster failure.
5. A reproducible FDAL-taxonomy benchmark for streaming federated HAR (scenarios S0–S3, a ground-truth change-point manifest, and a metric suite).

**Privacy wording.** Unless we implement secure aggregation or DP-SGD (an optional Phase 6b), we claim only *data locality*, not formal privacy.

---

## 2. Current-State Audit (what exists today)

The audit found the pipeline is **not yet a valid experimental platform**. Several findings change the scope of the work, and they should be agreed before anything else.

### 2.1 Blocking issues (must fix before any result is reportable)

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| A1 | **The "dynamic drift FL" simulator uses mock training.** Loss is `np.var(features)` and accuracy is a label-overlap heuristic plus random noise. | `src/federated/drift_aware_fl.py:357-371` | Any number from this module is not a real result. It must be replaced. |
| A2 | **No clustering code exists.** `grep` finds no clustering, FedNova or FedProx implementation. `aggregator.py` has `fedprox (future)`. | `src/federated/aggregator.py:31` | The plan's "fix singleton clusters" step is really "build clustering from scratch", which is a larger scope than the plan assumes. |
| A3 | **No Page-Hinkley detector** and no unsupervised (entropy/MMD) detector. Only ADWIN on supervised binary error. | grep over `src/` | E5 and the two-level detector (RQ1) are unimplemented. |
| A4 | **Injected drift is label-preserving covariate shift only.** It adds a constant offset to standardised accelerometer values. | `drift_injection.py:96-250` | This is *virtual* drift. There is no *real* drift (P(y\|x) changing). The plan's S1 real/virtual split requires a new injector. |
| A5 | **Train/test split is temporal, per subject.** PAMAP2 protocol activities are performed sequentially, so the final 20% of a subject's stream contains activities absent from the first 80%. | `run_random_client_drift_fl.py:147-155` | The reported ~96% accuracy is likely inflated or deflated depending on the split, and is not comparable to the literature. |
| A6 | **Normalisation uses a running EMA over the whole recording.** It is not fit on pre-drift data only. | `preprocessing.py:142-192` | Temporal leakage, and it silently masks injected offset drift. |
| A7 | **Window labels are the majority label** and windows can straddle activity transitions. Windows are not filtered for label purity. | `windowing.py:153-161` | Transition windows create spurious error spikes, which inflate false alarms. |
| A8 | **Drift is re-sampled each round** from a fresh RNG, and each round trains on a freshly drifted copy of the same static training set. There is no stream, no persistent change point and no ground-truth manifest. | `run_random_client_drift_fl.py:352-390` | Detection delay and recovery time cannot be measured. This is not streaming evaluation. |
| A9 | **The detector is not wired into the FL experiment.** The experiment script uses `FLClient.train_local` only. ADWIN, the classifier and the adapter are not used in the FL loop. | `run_random_client_drift_fl.py` imports | The central hypothesis has never been tested end to end. |

### 2.2 Documentation and repo-hygiene issues
- The LSTM has **53,650 parameters** (~210 KB FP32), computed from the architecture. The plan document says ~77K / 308 KB. Reported figures must match the code.
- `CONTEXT_HANDOFF.md`, `README.md` and `CLAUDE.md` describe tests (`test_phase1..6.py`, `test_flower_fl.py`) that are **deleted** in the working tree. The remaining tests are `test_01_*` to `test_04_*` and read as scripts, not assertions.
- `README.md` references a nonexistent `src/models/`, `src/evaluation/` and `test_preprocessing.py`.
- The repo has only 2 commits, 11 test files are deleted and uncommitted, and `__pycache__` is tracked.
- `adapter.py` and `improved_adapter.py` are near-duplicates (~700 lines each).
- The adapter is **type-specific** (full retrain / fine-tune / model library / sensor compensation), not a bottleneck adapter with a frozen backbone. The plan's "adapter-only" claim needs a new module.
- The 4-way rule-based drift classifier has 69% accuracy, and 0% on SUDDEN in `drift_detection_results.json`. The perfect detection scores (P/R/F1 = 1.0) come from synthetic error streams, not from real model error.
- The `.tex` results chapter reports per-subject independent training (~96.5%). That is not a federated result.
- Dataset: `Protocol/` and `Optional/` are both present. Subjects 102, 103, 104 and 107 have no `Optional` file.

### 2.3 Assets worth keeping
PAMAP2 loading and interpolation, the sliding-window generator, the LSTM (`lstm_model.py`), the ADWIN integration via `river`, the drift-injection primitives (sudden/gradual/recurring/feature-level offsets), and the FL client/server scaffolding. The dataset overview and figure scripts are reusable for the dataset section.

---

## 3. Experimental Design

### 3.1 Data protocol (fixes A5–A7)
- 9 subjects = 9 clients (primary). A secondary run partitions by subject groups to vary the client count.
- Drop Activity 0. Use the 12 protocol activities as the primary label set, with the Optional activities as a recorded extension.
- Window: 256 samples (2.56 s) with a 128 stride. **Keep only windows with ≥90% single-label purity.** Log how many are dropped.
- Per-client stream is built in **randomised-block activity order**, so an activity is not confounded with time position. Each client is split into a pre-drift training stream, a pre-drift validation set and a post-drift evaluation stream. The split is stratified so every class appears in train and test.
- Normalisation statistics are **fit on the client's pre-drift training partition only** and then frozen.
- Channels: start with the 9 accelerometer channels (current). Add the gyroscope as an ablation if reviewers ask, since it is a standard HAR input.

### 3.2 Drift scenarios (fixes A4, A8)
Each scenario is a **deterministic, versioned schedule** (YAML) that produces a per-client stream and a **ground-truth change-point manifest**.

| ID | Regime | Coverage | Variants |
|---|---|---|---|
| S0 | None (control) | – | False-alarm baseline, idle overhead |
| S1 | Sudden | partial (3/9) and full (9/9) | **Virtual** (axis scaling / rotation / offset) and **Real** (activity-to-label regime change) |
| S2 | Gradual | asynchronous across clients | Linear ramp over W rounds |
| S3 | Recurring | partial | A→B→A, tests forgetting and concept reuse |

Severity sweep: 3 magnitudes per scenario. Timing: fixed round T and a randomised T per client.

### 3.3 Baselines (8 tiers, identical splits, seeds and budgets)

| ID | Method | Purpose |
|---|---|---|
| E1 | Local-only LSTM | Lower bound, value of federation |
| E2 | Centralised LSTM | Upper bound |
| E3 | FedAvg (passive) | Drift-unaware FL |
| E3b | FedAvg + **periodic** retraining | Answers "why not just retrain on a schedule?" |
| E4 | FedAvg + ADWIN | Window-cut detector |
| E5 | FedAvg + Page-Hinkley | Sequential detector |
| E6 | FedProx and FedNova | Static heterogeneity baselines |
| E7 | CDA-FedAvg / FedConD / FedDrift (and FLASH if time permits) | Published drift-aware FL |
| E8 | **Proposed**: two-level detector + adapter-only + drift-gated clustering | Our system |

Additional fine-tuning baseline: FedAvg followed by per-client fine-tuning (FedAvg+FT), which is the standard personalisation comparison.
Re-implementing E7 methods is the highest-effort item. Where there is no official code, we state clearly that the comparison is against **our re-implementation**.

### 3.4 Ablation matrix (Phase 6)
0. **Drift typing:** oracle drift type, learned/rule-based classifier, type-agnostic policy (always full retrain, always fine-tune, always adapter). This is the key ablation for contribution 1.
1. Detector: ADWIN, PH, entropy/MMD only, two-level.
2. Supervision: immediate labels, delayed labels (k windows), label budget (1%, 5%, 20%).
3. Adaptation: full retrain, last-layers fine-tune, adapter-only, adapter plus rehearsal buffer.
4. Aggregation: global single model, static clusters, drift-gated clusters (raw vs step-normalised).
5. Sensitivity: ADWIN δ, PH λ, cooldown, adapter bottleneck size, clustering threshold.

### 3.5 Metrics
| Group | Metrics |
|---|---|
| Predictive | Accuracy, **macro-F1**, per-class P/R, confusion matrix (global and per-client) |
| Detection | TP / FP / FN, **mean detection delay** (windows and rounds), false-alarm rate on S0 |
| Adaptation | Recovery time, max accuracy drop, area under the accuracy curve, **backward retention** |
| Fairness | Mean, worst-client and variance of client macro-F1 |
| Clustering | Cluster-size histogram per round, ARI vs ground-truth drift groups, cluster oscillation rate |
| Communication | Uplink and downlink bytes per round and cumulative (payload bytes) |
| Compute | Local train time, detector latency, peak memory, and ideally an energy proxy (FLOPs) |

### 3.6 Statistical protocol
- **5 seeds** (42, 123, 999, 2024, 7), not 3. With only 9 clients, variance is high.
- Report mean ± std and **95% bootstrap confidence intervals**. Use paired tests (Wilcoxon signed-rank) across seeds and clients with Holm correction.
- Hyperparameters are tuned on S0 plus a held-out validation drift schedule, **never on the test schedule**.
- One configuration file per run, with git commit hash recorded in every result JSON.

---

## 4. Method Work Packages

| WP | Deliverable | Notes |
|---|---|---|
| WP1 | Detector module: supervised ADWIN, PH, entropy-PH, MMD, and the two-level combiner with cooldown | Formal definition of the scalar entering each detector |
| WP1b | **Drift-type classifier** rebuilt on real model-error streams, with extended features and a learned variant compared against the rule-based one | Core novelty; evaluated on classification accuracy and downstream adaptation benefit |
| WP2 | Adapter module: bottleneck adapters inserted after each LSTM layer plus a head, with a frozen backbone | Report exact trainable-parameter count and bytes |
| WP3 | Clustering module: FedNova-normalised updates, cosine similarity on adapter/head, drift-gated split, agglomerative merge | Diagnose singleton failure first with a synthetic test |
| WP4 | Streaming FL engine on **real LSTM training** (GPU), integrating WP1–WP3, with Flower used for message accounting or retained as the transport | Decide between native simulator and Flower in Phase 0 |
| WP5 | Baseline wrappers (E1–E7) on the identical backbone | |
| WP6 | Experiment runner, result schema, analysis and plotting scripts | |

---

## 5. Phased Roadmap

Effort is indicative for one researcher working with Claude Code. Each phase has an exit gate, and we do not proceed until it passes.

| Phase | Scope | Exit gate | Est. |
|---|---|---|---|
| **0. Foundations** | Commit current state; restore or rewrite tests as real asserts; remove `__pycache__`; consolidate the duplicate adapters; add `pyproject`, pinned `requirements.txt`, CUDA check for the RTX 5050; results schema; config system | CI-style `pytest` green; one deterministic command reproduces a baseline | 2–3 d |
| **1. Data and baseline lock** | Fix A5–A7 (purity filter, stratified split, frozen normalisation); data audit table; real E1, E2, E3 on stationary data; parameter and bytes table | Verified config table; E1–E3 curves reproduce across 3 seeds | 4–5 d |
| **2. Drift generator** | New injector with real and virtual drift; S0–S3 schedules; change-point manifest; plots of every stream | Manifest-driven unit tests pass; visual check of each scenario | 3–4 d |
| **3. Detectors** | WP1; measure delay and false alarms on S0–S3 with real model error | Detector report table; FPR on S0 within agreed bound | 4–5 d |
| **4. Streaming FL engine and adapter** | WP2 and WP4; E3, E4, E5 end to end; adapter vs full retraining preliminary | Detector wired into FL; first recovery curves on S1 | 6–8 d |
| **5. Clustering** | WP3; synthetic test (4 of 9 drifted clients form 2 clean clusters) then real streams | Cluster-size histogram shows no singleton degeneracy | 4–6 d |
| **6. Baselines and SOTA** | WP5; E3b, E6, E7; full E1–E8 × S0–S3 × 5 seeds | Complete results JSON; sanity checks against published trends | 8–10 d |
| **7. Ablations and sensitivity** | §3.4; Pareto plots (delay vs FPR vs bytes) | Ablation tables; claims in §1.4 confirmed or dropped | 5–6 d |
| **8. Edge-cost validation (optional)** | Runtime and memory on a throttled CPU (and a Raspberry Pi if available). Second dataset deferred (Section 7) | Overhead table | 2–3 d |
| **9. Paper** | IEEE LaTeX: Intro, Related Work (FDAL taxonomy), System Model, Method, Setup, Results, Ablations, Limitations, Conclusion; reproducibility package | Internal review round, then submission | 10–14 d |

**Total:** roughly 8–10 weeks. The critical path is Phases 1–4. Phase 6 (E7 re-implementation) is the largest risk to the schedule.

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Proposed system does not clearly beat FedAvg+FT or FedDrift | Medium | High | Frame it as a Pareto result (accuracy vs bytes vs delay). Decide the framing after Phase 4 and not at the end. |
| Only 9 subjects, so low statistical power | High | Medium | 5 seeds, bootstrap CIs, second dataset (Phase 8), and per-client reporting. |
| Synthetic drift criticised as unrealistic | High | High | Add *natural* drift evidence, for example cross-subject sensor-placement shift or session splits from PAMAP2 Optional, and justify the injector parameters. |
| Re-implemented baselines viewed as unfair | Medium | Medium | Use official code where available, tune baselines with equal budgets, and state differences openly. |
| Clustering has no signal at 9 clients | Medium | Medium | Report it honestly as a negative or neutral ablation, and move it from core contribution to a variant. |
| The 4-way classifier is the novelty but currently scores 69% overall and 0% on sudden, on synthetic error streams | High | High | Rebuild it on real model-error streams from the new generator (Phase 3). Report classification accuracy and its *downstream* benefit. If typing does not beat type-agnostic adaptation, we report that honestly and pivot the framing. |
| Scope creep (quantisation, DP, extra datasets) | Medium | Medium | Deferred items are fixed in Section 7. |
| Windows/CUDA compatibility on the new GPU (Blackwell) | Low–Med | Low | Verify the PyTorch build in Phase 0. |

---

## 7. Decision Log (author-approved)

| # | Decision | Rationale / consequence |
|---|---|---|
| 1 | **Keep the 4-way drift classifier as a core contribution.** | Prior work addresses a single drift notion, so typing is the differentiator. It must be rebuilt and validated on real model-error streams (WP1b) and justified by the type-agnostic ablation. |
| 2 | **Clustering is an optional extension**, decided after Phase 5 results. | Reduces risk. It is dropped from the paper if it gives no measurable benefit. |
| 3 | **PAMAP2 only** for now. | A second dataset is deferred to future work or revision, and the limitation is stated in the paper. |
| 4 | **Baselines are chosen from current published methods and compared against their reported findings.** | We re-implement E7 methods on our backbone for a controlled comparison, and also place our results beside the numbers reported in the original papers, flagging that the protocols differ. Official code is used where it exists. |
| 5 | **Differential privacy / secure aggregation deferred** to the end (Phase 6b). | Until then we claim data locality only. |
| 6 | **Transport: native PyTorch simulator** on the GPU for all experiments. | Fastest and most controllable for 9 clients × 5 seeds × many scenarios. Flower stays in the repo as an optional integration, and byte accounting follows the real serialised payload so the cost numbers are Flower-comparable. |
| 7 | **Repo restructure and commits approved** (layout in Section 8). | Executed in Phase 0, as small reviewable commits with the old modules preserved in git history. |

### Open items (non-blocking)
- Target venue and deadline are still unset. They change the depth of E7, not the plan structure.
- The exact list of E7 methods (CDA-FedAvg, FedConD, FedDrift, FLASH) is fixed at the start of Phase 6, depending on code availability.

---

## 8. Proposed Repository Layout

```
configs/            # scenario, model, detector, FL configs (one YAML per experiment)
src/
  data/             # loading, windowing, splits, normalisation
  drift/            # injection, schedules, ground-truth manifests
  detect/           # ADWIN, PH, entropy, MMD, two-level
  models/           # LSTM, adapters
  fl/               # engine, aggregation, clustering, baselines
  eval/             # metrics, statistics, plotting
experiments/        # runnable entry points (E1..E8, ablations)
results/            # JSON logs (git-ignored raw, committed summaries)
tests/              # asserting unit and integration tests
docs/               # this plan, protocol, paper draft
```
