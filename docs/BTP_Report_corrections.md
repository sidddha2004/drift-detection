# BTP report: corrections (checked against the result files, 2026-11-23)

Apply these in the report's LaTeX source (Overleaf). Each item gives the place, the current text, and the replacement. Sources are the files in `results/`.

## A. Clear errors

1. **Section 3.3, last paragraph (page 18)**
   - Now: "An untrimmed protocol raises every method by 4–9 points without changing any ranking."
   - Replace with: "An untrimmed protocol lowers every method by 4–6 points (trimming raises them) without changing any ranking."
   - Source: `results/phase1d/comparison.md` (v2 − v1: E1 +5.6, E2 +5.9, E3 +4.2, E3b +4.8 points).

2. **Section 3.5, after Table 3.2 intro (page 18)**
   - Now: "with about 3 labelled arrival windows per client and round"
   - Replace with: "with about 2 labelled arrival windows per client and round (median 2, range 1–3)"
   - Source: per-client arrivals [3, 2, 1, 2, 2, 2, 2, 2]; Sections 3.12.2 and 4.8.2 already say 2.

3. **Section 4.10, first paragraph (page 30)**
   - Now: "Stronger noise holds F1 at about 0.26–0.28, barely above chance and far below the frozen model without any upload (0.682)."
   - Replace with: "Stronger noise holds F1 at about 0.26–0.28, far below the frozen model without any upload (0.682)."
   - Reason: chance macro-F1 with 12 classes is about 0.08, so 0.27 is not "barely above chance".

## B. Smaller mismatches

4. **Table 4.7, row 5 (page 32)**
   - Now: "Labelled arrival buffer: align / head retraining (45 rounds) | 288 / 810"
   - Replace with: "Labelled arrival buffer: align (16 rounds) / head retraining (45 rounds) | 288 / 810"
   - Source: `results/phase7/memory_budget.md`.

5. **Table 4.8, last two rows (page 32)**
   - Now: "E8 (no upload) – 0 0.729 ± 0.050 0.951" and "E3b frozen – 0 0.682 ± 0.024 0.940"
   - Replace with: "E8 (no upload) – 0 0.729 ± 0.054 0.943" and "E3b frozen – 0 0.682 ± 0.023 0.944"
   - Computed the same way as the other rows (seed-level mean over S1–S3, std over seeds; clean F1 = mean over runs), from `results/phase4_v2/conf_*.json`.

6. **Section 4.10, second paragraph (page 31)**
   - Now: "For the adapter (d ≈ 1,971)"
   - Replace with: "For the adapter (d = 1,966)"
   - The noise factor stays about 22C.

7. **Table 4.1, rows 2 and 3 (page 23)**
   - ADWIN row, last column: "–" → "0.20".
   - Label-free row: "Label-free (entropy, MMD) | chance level | – | –" → "Label-free (entropy / MMD) | 0.21 / 0.14 | 9.3 / 11.3 | 1.32 / 1.40".
   - **Section 4.2, sentence 3:** "Label-free detectors are barely above chance, as expected for real drift, which changes no input statistics." → "Label-free detectors are weak (recall 0.21 and 0.14 against placebo 0.09 and 0.05), break the false-alarm bound, and are nearly blind to real drift (recall ≤ 0.09), which changes no input statistics."
   - Source: `results/phase3_v2/detector_report.md`.

8. **Section 4.3 (page 24)**
   - Now: "4-way typing from detected alarms is weak: accuracy 0.46 …"
   - Replace with: "On the first (shorter) benchmark, 4-way typing from detected alarms was weak: accuracy 0.46 …"
   - These numbers come from the Phase 3 v1 benchmark, not the final v2 benchmark.

9. **Section 3.7, sentence 3 (page 20)**
   - Now: "It is trained on episodes of seeds 1004–1010 with leave-one-seed-out validation."
   - Replace with: "It is trained on episodes of seeds 1004–1010. Its leave-one-seed-out accuracy over seeds 1001–1010 is reported in Section 4.3."
   - Source: `experiments/phase4_streaming.py` trains on 1004–1010; `results/phase3_v2/mode_typing.md` validates over 1001–1010.

10. **Table 4.3 caption (page 26) and List of Tables**
    - Now: "Oracle rows are upper bounds."
    - Replace with: "Oracle rows use ground-truth alarm times and types. Oracle 2-way uses one-shot alignment and can score below E8 with the true mode."
    - Also in the Fig. 4.8 caption: "Oracle bars are upper bounds." → "Oracle bars use ground-truth alarms and types."

11. **Units (Section 3.12.2 or the start of Chapter 4)**
    - Add: "Uplink is reported in kB = 1,000 bytes; memory sizes in kB = 1,024 bytes."

12. **Fig. 2.1 caption (page 15) and List of Figures**
    - Now: "Types of concept drift following Gama et al. [4]: sudden, gradual, recurring, and feature-level drift."
    - Replace with: "Drift profiles used in this work: sudden, gradual and recurring drift following Gama et al. [4], plus feature-level drift on a single sensor."

13. **Section 4.8.1, item 1 (page 28)**
    - Now: "After round 45 it applies the transform fitted for regime B to clean signals."
    - Replace with: "After the episode ends at round 45, it keeps applying the transform fitted to the drifted signal to clean signals."

## C. References to verify before submission

These were not in the 26 references we checked against publisher records:
- [1] Ramadan et al.
- [2] Villegas-Ch et al.
- [3] Ordóñez & Roggen
- [5] Agrahari & Singh
- [13] FLARE (also check the "factor of 5" and "factor of 16" claims in Section 2.4)
- [19] FedPLC
- [25] Rahimli et al.

Claims about other references to soften or confirm:
- **[27] DriftGuard (Section 2.4):** "pairs a representation-level and an error-level detector with adapter-only adaptation" was not confirmed. Safer wording: "studies federated continual learning for encrypted-traffic classification under asynchronous drift (simulation only)".
- **[32] Jiang:** it is a 3-page overview; keep the claim about it general.

## D. Check by eye

- **Title:** it says "Privacy Preserving", while the DP result is negative and Section 5.2 claims data locality only. Be ready to explain this, or add "(data locality)" in the abstract.
- **Certificate:** says ECE department; the degree is CSE. This is fine if the supervisor is in ECE.
- **Missing spaces:** the extracted text shows words run together (Acknowledgements, Section 1 paragraph 1, RQ5, the Table 4.8 caption, the Section 4.6 heading). It may only be the text extraction; check the PDF by eye.
