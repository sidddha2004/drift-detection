# Review: "Drift-Aware Federated Learning for Wearable Activity Recognition: Typed Local Adaptation, Recurring-Drift Failure, and Privacy Limits" (8-page IEEE conference draft)

Checked on 2026-11-23 against the repository results and against how the papers we cite are written (FedDrift, FedStream, FedConD, FedHAR, Flash).

**Overall.** The paper is honest, well controlled and has an unusually strong evaluation for a student paper: a pre-specified plan, held-out seeds, Holm correction and negative results reported. The main tables (II, III and the H4 numbers) match the result files exactly. What keeps it from reading like an IEEE paper:
1. Internal project vocabulary (E-codes, policy names, "Phase 6", "A6 M1", file paths) instead of named methods.
2. No formal problem statement, no system figure, no algorithm box.
3. The DP and statistics sections are missing their standard citations.
4. The same factual slips as in the BTP report.

Priorities: **A** must fix before submission (errors), **B** expected by reviewers (structure), **C** polish.

---

## A. Must fix: factual errors and inconsistencies

| # | Where | Now | Change to | Evidence |
|---|---|---|---|---|
| A1 | §III-A, last para | "An untrimmed v1 protocol raises every method by 4–9 points" | "lowers every method by 4–6 points (trimming raises them)" | `results/phase1d/comparison.md`: v2 − v1 = +4.2 to +5.9 |
| A2 | §III-B | "about 3 labelled arrival windows per client and round" | "about 2 (median 2, range 1–3)" | per-client arrivals; §VII-B already says 2 |
| A3 | §VI-A | "Label-free detectors are barely above chance" | "are weak (recall 0.21 and 0.14 against placebo 0.09 and 0.05), exceed the false-alarm bound, and are nearly blind to real drift (recall ≤ 0.09)" | `results/phase3_v2/detector_report.md` |
| A4 | §VIII, para 2 | "barely above chance and far below the frozen model" | "far below the frozen model" | chance macro-F1 with 12 classes ≈ 0.08 |
| A5 | §VIII, para 3 | "adapter (d ≈ 1,971)" | "(d = 1,966)" | §IV-C itself says 1,966 |
| A6 | Table IV, row 5 | "align / head retrain (45 rounds) 288 / 810" | "align (16 rounds) / head retrain (45 rounds) 288 / 810" | `results/phase7/memory_budget.md` |
| A7 | Table V, last rows | E8 "0.729 ± 0.050 / 0.951"; E3b "± 0.024 / 0.940" | E8 "0.729 ± 0.054 / 0.943"; E3b "± 0.023 / 0.944" | recomputed like the other rows from `conf_*.json` |
| A8 | Table III caption, Fig. 8 caption, §IV-C | "Oracle … are upper bounds" | "use ground-truth alarm times and types" | oracle 2-way scores **below** "E8 with true mode" in every column (S1 0.779 < 0.783), so it is not an upper bound |
| A9 | §VI-A | "4-way typing … accuracy 0.46 against 0.51" | add "on the first (shorter) benchmark" | from the Phase 3 v1 benchmark, not v2 |
| A10 | §IX | "With an oracle trigger, input alignment recovers 12–16 points on virtual drift (tuning seeds)" | add "on the first benchmark" | `docs/decisions.md` items 197/215 (Phase 4 v1) |
| A11 | §VII-B and §VII-C headings | "(A6, M1)" and "Edge Memory Budget (A6, M2)" | drop the codes (see B1); if kept, M2 is the **DP** study, not the memory budget | `docs/analysis_plan.md` A6 |
| A12 | §V | "amendments A1–A5" | "A1–A6" (A6 is used later in the same section) | |
| A13 | §VII-A, item 1 | "the transform fitted for regime B" ("regime B" is never defined) | "keeps applying the transform fitted to the drifted signal to clean signals" | |
| A14 | §II, DriftGuard [20] | "pairs representation-level and error-level detectors with adapter-only adaptation" | "studies federated continual learning for encrypted-traffic classification under asynchronous drift (simulation)" | the method details were not confirmed; the cited title does not say "DriftGuard" |
| A15 | §II, [25] | "system-level work covers continual learning and sparse aggregation" | "federated learning systems for mobile data are surveyed in [25]" | [25] is a 3-page overview |
| A16 | §I | real-drift example "changed gait or fatigue" | "an activity is relabelled (e.g. two activities swap meaning)" | gait/fatigue changes P(X), which is virtual drift; the benchmark's real drift is label remapping |
| A17 | §IX | "Federated full-model retraining did not help" | "helped little: +0.02–0.03 on S1 at 2.3–18.7 MB of uplink" | the sentence after it already gives these numbers |
| A18 | Author block | "Abhinav Agrawal", "Richa Priyadarshani" | spell as in the BTP report ("Agarwal", "Priyadarshini"); check which is correct | the two documents disagree |
| A19 | Units | "kB" | state once: uplink in kB = 1,000 B, memory in kB = 1,024 B | the two tables use different kB |

## B. Structure that reviewers expect (compared with the cited papers)

**B1. Replace internal codes with method names.** Cited papers name systems by what they do. Ours uses E3b, E5, E7a, E8, `typed_learned`, `ph_head_long`, `ph_fedprox_E7a`, `frozen_E3b`, "Phase 6", "Phase 7", "A6, M1", "family H6", and the file path `docs/analysis_plan.md`.
- Give the proposed system a name. For example **TLA (Typed Local Adaptation)**; then E8 becomes TLA.
- Rename the baselines: Frozen-Global, Frozen-Personal, Periodic-FedAvg, ADWIN-FedAvg, PH-FedAvg, PH-FedProx, PH-Adapter, Fixed-Align, Fixed-Head.
- Add a footnote on the first mention of "pre-registered": "A time-stamped analysis plan was committed to version control before evaluation runs (commit `<hash>`)." Use "pre-specified" unless the plan is in a public registry such as OSF; reviewers may challenge "pre-registered".
- Apply the same names in Table I, the figure legends (Figs. 1–10 use D1…D5ns, `rule dF_delta>0.05`, `gboost`) and the abstract (which uses "E8" without defining it).

**B2. Add a formal problem statement (new §III-A).** FedDrift and FedStream both formalize the setting. Define:
- clients k = 1…K
- rounds t, and the stream of labelled windows (x, y) per client
- the drift indicator α_k(t)
- virtual drift: P_t(X) changes, P_t(Y|X) fixed; real drift: P_t(Y|X) changes
- the metric: post-change macro-F1 = mean over t ≥ t_k* of macro-F1_k(t), averaged over drifted clients
- the uplink cost U = Σ bytes uploaded
- the objective: maximise post-change F1 subject to low U

**B3. Add a system-overview figure (Fig. 1).** The BTP report already has one (Report Fig. 3.1: stream → PH detector → mode typing → align / head retrain; server only for the federated baselines). Put it right after the introduction.

**B4. Add an algorithm box.** For example, "Algorithm 1: Typed local adaptation on client k": per round, update PH on the error bit; on an alarm, wait τ = 5 rounds; predict the mode; apply closed-form alignment (Eq. 1) or head retraining. Also add the Page–Hinkley statistic as an equation; the report has it as Eq. 3.1.

**B5. Position against the closest work in a comparison table** at the end of §II, with columns: detection on client / drift-type aware / local response / uplink per episode / HAR / evaluation protocol. Rows: FedDrift [14], Flash [15], FedConD [16], CDA-FedAvg [17], FedStream [18], Chen et al. [19], FedHAR [23], ours. Chen et al. [19] also separate virtual from real drift, so one sentence must say what is new here: client-side typing that selects a zero-uplink response, and a byte-accounted evaluation.

**B6. Reproducibility paragraph (end of §V).** Most cited papers give their hyperparameters and hardware.
- Optimiser, learning rate, batch size, local epochs, head-retraining epochs, buffer sizes.
- GPU (RTX 5050 laptop), PyTorch version, runtime per stream.
- A code link (anonymised for review if the venue is double-blind).
- Contribution (i) says "reproducible"; that claim needs the code link.

**B7. Add the missing standard citations.** Verify the details before adding.
- LSTM: Hochreiter & Schmidhuber, *Neural Computation* 9(8), 1997.
- Gradient boosting: Friedman, *Annals of Statistics* 29(5), 2001.
- Wilcoxon signed-rank test: Wilcoxon, *Biometrics Bulletin* 1(6), 1945.
- Holm correction: Holm, *Scandinavian J. Statistics* 6(2), 1979.
- Bootstrap CI: Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993.
- Differential privacy: Dwork & Roth, *Found. Trends TCS* 9(3–4), 2014.
- Rényi DP accountant: Mironov, IEEE CSF 2017.
- Client-level DP in FL: McMahan *et al.*, "Learning differentially private recurrent language models", ICLR 2018.
- DP-SGD: Abadi *et al.*, ACM CCS 2016.
- Deep HAR on PAMAP2: Ordóñez & Roggen, *Sensors* 16(1), 2016 (cited in the BTP report).

Without the DP citations, §VIII reads as unsupported to a privacy reviewer.

**B8. Make RQ5 explicit.** The intro says "We add a practical question" after RQ1–RQ4. Number it RQ5 (memory and DP), as the BTP report does, and answer each RQ in one line in the conclusion.

**B9. Missing baseline.** Reviewers in this area will ask why no published drift-aware FL method was run with its own code (FedDrift, Flash). Either run one, or keep the existing disclaimer and add one sentence in Limitations: "No official implementation of a drift-aware FL method was evaluated; E5/E7a approximate their response step."

**B10. Report cost on the device.** The paper claims cheap local adaptation but gives no compute numbers. Add the wall-clock time of the closed-form alignment and of head retraining per episode on the GPU, and state that device energy and latency were not measured.

## C. Polish

- **Abstract:** about 260 words; IEEE asks for 150–250. Cut the memory-budget clause and the two Holm p-values (keep the effect sizes).
- **Spelling:** the text uses American spelling ("personalized") but figure legends use British ("personalised"). Pick one; IEEE uses American.
- **Author block:** "Jaipur,India" → "Jaipur, India". "Mentor" is not an affiliation; use "Department of Electronics and Communication Engineering, LNMIIT". If the mentor is a co-author, drop the acknowledgment to her and thank the co-supervisor (Dr. Ankit Jain, named in the report) instead.
- **Figures:** 10 figures in 8 pages is heavy.
  - Merge Figs. 7 and 9 (bar charts of a single metric), or move Fig. 6 (the heatmap) to supplementary material.
  - Every figure must be cited in the text in order; Fig. 4 is cited before Fig. 3.
  - Fig. 1 x-axis: the label must be readable at column width.
- **Title:** 21 words is long for IEEE. For example: "Typed Local Adaptation for Drift in Federated Wearable Activity Recognition: Gains, Failures and Privacy Limits".
- **Conclusion:** add one sentence stating what practitioners should do: "use error-triggered local head retraining or a federated adapter; avoid full-model retraining and client-level DP at this cohort size".
- **Tables:** Table III has 14 rows and should be one double-column table (`table*`); mark the best deployable value per column in bold.
- **Numbers:** "Holm p = 0.078" against "p = 0.078"; be consistent. Use "macro-F1" with the hyphen everywhere.

## What is already good (keep)

- The disclosure paragraph (§V) and the reporting of negative results. They are unusually honest and reviewers will respect them.
- H4 framed as a non-inferiority test with a stated margin.
- The DP section explains the mechanism (noise norm z·C·√d against √K averaging) instead of just showing a curve.
- The S3 failure diagnosis (on/off-phase F1, 18 of 36 late responses) turns a negative result into a finding.
