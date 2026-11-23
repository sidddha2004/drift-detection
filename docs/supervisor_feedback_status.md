# Supervisor feedback (23 Nov 2026): status

Paper: `docs/paper_ieee/main.tex` (IJCNN 2027, 6 pages, double-blind). Figures: `docs/paper_ieee/figs/` (`experiments/paper_figs/ijcnn_figs.py`).

| # | Item | Status | Note |
|---|---|---|---|
| 1 | New title and abstract (a study, not a better system) | done | Title as suggested. The abstract states the four findings, H5 failing and the label requirement. |
| 2 | Reorder contributions (adapter, clean users, recurring failure, wrong type) | done | In that order. **Changed number:** clean users drop **about 5 points** (0.944 → 0.895 under S1; 4.4–5.4 across variants) under *periodic* retraining. The 6.7 in the feedback is not supported by the results; alarm-triggered full retraining costs only 0.1–0.4. |
| 3 | 96% follows from the adapter size | done | One sentence in the contributions and one in §IV-A (3.7% of the parameters). |
| 4 | Remove "better than baselines" language | done | H5 failure stated in the abstract, contributions, §IV-C and the conclusion. |
| 5 | Labels needed on device, in the Introduction | done | A "Labels are required" paragraph in §I. |
| 6 | Only the type classifier is label-free | done | Abstract, §I, §III and the Fig. 1 caption. |
| 7 | Minimum Holm p = 0.0234; p-values primary | done | §III Protocol: 2/2^10 = 0.00195 × 12 = 0.0234. |
| 8 | ConceptMemory on S3, 10 evaluation seeds | done | No gain: post-change 0.709 vs 0.711 (p = 0.38), between episodes 0.785 vs 0.784; K=0 and K=4 identical (8 switches in 20 streams). Reported in Table I, Fig. 4 and §IV-E as a negative result. |
| 9 | S0 no-drift run | done | 0.9–1.5 false alarms per stream; F1 change −0.1 to −0.5 points for alarm-based systems; periodic retraining −4.1 points and 18.7 MB uplink with no drift. §IV-F. |
| 10 | LSTM layers | done | Code: `num_layers: 2`. 19,200 + 33,280 + 780 = 53,260 parameters. The text says two-layer and shows the sum. |
| 11 | Oracle 0.827 vs 0.797 | done | 0.827 does not appear in any result file; every oracle number now comes from Table I (oracle 4-way S3 = 0.797). |
| 12 | What the adapter sends | done | The code averages the whole adapter: input layer 90 + bottleneck 1,096 + head 780 = 1,966 parameters (7,864 B per upload). Stated exactly in §III. The feedback's 1,096 / 4.4 kB would only hold if the input layer and head stayed local, which is not what was run. |
| 13 | 1,400 runs | done | 14 systems × 10 variants (S1 4, S2–S4 2 each) × 10 seeds. |
| 14 | Related Work in three groups, ~10 more references | done | FL under drift; recurring-concept memory (Katakis 2010, RCD 2013, Gama & Kosina 2014); sensor displacement (Kunze & Lukowicz 2014, Baños 2014, Chavarriaga 2013). Each group ends with a "Difference:" line. All 6 new references checked online. |
| 15 | Plain baseline names | done | In the text, tables and figures (e.g. "Personalized (no adaptation)", "Head retraining (fixed)"). |
| 16 | 6 pages; bigger figure text | partly | New column-width figures at 8 pt. **Not compiled here** (no LaTeX); check the page count in Overleaf. |
| 17 | Blind review | done | IJCNN 2027 is double-blind (reviewer guidelines). `\blindtrue` hides the authors and the GitHub link; switch to `\blindfalse` for the camera-ready version. **The official site lists both 15 Jan and 31 Jan 2027 as deadlines; confirm on CMT.** |
| 18 | Author block, no Acknowledgment | done | Supervisor's block used exactly (department: Communication and Computer Engineering); Acknowledgment removed. |
| 19 | Department name | open | The block uses "Computer Science and Engineering"; confirm the official name. |

**Name spelling:** the supervisor's block spells "Agrawal" and "Priyadarshani" and is used exactly; the BTP report uses "Agarwal" and "Priyadarshini". Settle one spelling for both documents.
