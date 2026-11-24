"""Full experiments-and-results report, in the order the experiments were run (Phase 0 -> supervisor follow-ups).

Every number is copied from the result files named next to it (results/..., docs/decisions.md). No training is run.
    PYTHONPATH=<dir with reportlab> python experiments/report/build_results_pdf.py [out.pdf]
"""
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "Concept_Drift_Experiments_and_Results_Full.pdf"

FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-B", str(FONT_DIR / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Body-I", str(FONT_DIR / "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Body-BI", str(FONT_DIR / "arialbi.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402
registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-BI")

INK, ACC, MUTED, BAND = colors.HexColor("#14213D"), colors.HexColor("#C4561B"), colors.HexColor("#4A5568"), colors.HexColor("#EEF2F7")
ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Body-B", fontSize=22, leading=27, textColor=INK, alignment=TA_CENTER),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontName="Body", fontSize=11.5, leading=16, textColor=MUTED, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Body-B", fontSize=16, leading=20, textColor=INK, spaceBefore=6, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Body-B", fontSize=12.5, leading=16, textColor=ACC, spaceBefore=10, spaceAfter=4, keepWithNext=1),
    "p": ParagraphStyle("p", parent=ss["Normal"], fontName="Body", fontSize=9.6, leading=13.4, spaceAfter=5),
    "b": ParagraphStyle("b", parent=ss["Normal"], fontName="Body", fontSize=9.6, leading=13.2, leftIndent=12, bulletIndent=2, spaceAfter=2),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontName="Body-I", fontSize=8.4, leading=11, textColor=MUTED, spaceAfter=8),
    "box": ParagraphStyle("x", parent=ss["Normal"], fontName="Body", fontSize=9.6, leading=13.4, textColor=INK),
    "cell": ParagraphStyle("cell", parent=ss["Normal"], fontName="Body", fontSize=8.2, leading=10.2),
    "cellb": ParagraphStyle("cellb", parent=ss["Normal"], fontName="Body-B", fontSize=8.2, leading=10.2, textColor=colors.white),
}
W = A4[0] - 3.6 * cm
story = []


PENDING = []


def add(*flows):
    """Append flowables; a pending heading is kept on the same page as the first of them."""
    global PENDING
    if PENDING:
        story.append(KeepTogether(PENDING + [flows[0]])); PENDING = []; flows = flows[1:]
    story.extend(flows)


def H1(t): PENDING.append(Paragraph(t, S["h1"]))
def H2(t): PENDING.append(Paragraph(t, S["h2"]))
def P(t): add(Paragraph(t, S["p"]))
def SRC(t): add(Paragraph("Source: " + t, S["cap"]))


def BUL(items):
    for t in items:
        add(Paragraph(t, S["b"], bulletText="•"))


def BOX(t, color=BAND):
    tb = Table([[Paragraph(t, S["box"])]], colWidths=[W])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LINEBEFORE", (0, 0), (0, -1), 3, ACC)]))
    add(tb, Spacer(1, 6))


def TAB(rows, widths=None, hl=None, cap=None):
    data = [[Paragraph(str(c), S["cellb"] if i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    if widths:
        tot = sum(widths); widths = [W * w / tot for w in widths]
    tb = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D1DC")),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FB")]),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    for r in (hl or []):
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#FDEBDD")))
    tb.setStyle(TableStyle(st))
    add(KeepTogether([tb] + ([Paragraph(cap, S["cap"])] if cap else [Spacer(1, 6)])))


def FIG(path, cap, width=0.78):
    p = ROOT / path
    if not p.exists():
        P(f"<i>[figure missing: {path}]</i>"); return
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(p)).getSize()
    w = W * width
    add(KeepTogether([Image(str(p), width=w, height=w * ih / iw), Paragraph(cap, S["cap"])]))


def footer(c, d):
    c.saveState(); c.setFont("Body", 8); c.setFillColor(MUTED)
    c.drawString(1.8 * cm, 1.1 * cm, "Drift-aware federated HAR on PAMAP2: experiments and results (full record)")
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {d.page}")
    c.restoreState()


# =====================================================================================================  cover
story += [Spacer(1, 3.2 * cm), Paragraph("Federated Learning for Privacy-Preserving Concept Drift Detection", S["title"]),
          Spacer(1, 6), Paragraph("Complete record of experiments and results, in the order they were run", S["sub"]),
          Spacer(1, 4), Paragraph("PAMAP2 wearable HAR · 8 federated clients · Phases 0–7 and supervisor follow-ups · status 24 Nov 2026", S["sub"]),
          Spacer(1, 1.2 * cm)]
BOX("<b>What this document is.</b> Every experiment of the project, from the audit of the first prototype to the last "
    "supervisor-requested runs, in chronological order. For each step: <b>why</b> it was done, <b>what</b> was run, the "
    "<b>numbers</b> (copied from the result files named under each table) and <b>what we concluded</b>. It replaces the earlier "
    "8-page summary, which stopped at Phase 6 and contained a few errors (listed in Section 14).")
BOX("<b>The project in five sentences.</b> We built a leakage-free PAMAP2 benchmark with ground-truth drift events and a federated "
    "streaming simulator that counts every byte. We tested a client-side pipeline that detects drift from its own errors, types it as "
    "virtual (sensor) or real (label meaning) and repairs it locally. Under a pre-registered test on 10 held-out seeds it beats no "
    "adaptation on sudden drift, but not a simple fixed head-retraining rule, and it fails on recurring drift. Sending only a small "
    "adapter instead of the whole model keeps accuracy at 4% of the upload, and local repair protects clients that did not drift. "
    "A memory for recurring drift and client-level differential privacy were both tested and did not help at this scale.", colors.HexColor("#FDF1E8"))
TAB([["Order", "Phase / step", "Dates (2026)", "Section"],
     ["1", "Phase 0: audit of the first prototype, foundations", "26 Oct", "2"],
     ["2", "Phase 1: clean data pipeline and baselines (E1, E2, E3, E3b); natural-drift check; edge trim (v2)", "26–27 Oct", "3"],
     ["3", "Phase 2: drift generator and severity calibration", "27 Oct", "4"],
     ["4", "Phase 3: drift detectors; drift-type classifiers", "27 Oct", "5"],
     ["5", "Phase 4: streaming FL engine, adapter, closed-form alignment; first streaming results (benchmark v1)", "27 Oct", "6"],
     ["6", "Phase 4b: longer benchmark v2 (tuning seeds)", "28 Oct", "7"],
     ["7", "Phase 6: pre-registered confirmatory test on 10 evaluation seeds", "28 Oct", "8"],
     ["8", "Phase 7: recurring-drift trace, memory, device memory budget, differential privacy", "22 Nov", "9–10"],
     ["9", "Supervisor follow-ups: S0 no-drift control; memory on evaluation seeds", "24 Nov", "11"],
     ["10", "Overall conclusions; what was and was not evaluated; corrections", "—", "12–14"]], [0.65, 5.1, 1.1, 0.75])
story.append(PageBreak())

# =====================================================================================================  1 glossary
H1("1. How to read this report")
P("Systems carry a short code from the project (E3b, E8, ...). The paper now uses plain names; both are given here. "
  "<b>Post-change macro-F1</b> is the main metric: the mean macro-F1 of the drifted clients from the moment the drift starts to the "
  "end of the stream, measured on held-out windows. Macro-F1 averages the F1 of every class, so rare activities count as much as common ones.")
TAB([["Code", "Plain name", "What it does after drift"],
     ["E1", "Local-only", "Each client trains alone (Phase 1 reference)"],
     ["E2", "Centralised", "All data pooled (Phase 1 reference, not private)"],
     ["E3", "Global (no adaptation)", "FedAvg model, never updated"],
     ["E3b", "Personalized (no adaptation)", "FedAvg + local fine-tuning; the start model of every system"],
     ["E3c", "Periodic full retraining", "Full-model federated retraining every 10 rounds, all clients"],
     ["E4", "ADWIN + full retraining", "ADWIN alarm, then full-model federated retraining"],
     ["E5", "PH + full retraining", "Page–Hinkley alarm, then full-model federated retraining"],
     ["E7a", "PH + FedProx retraining", "As E5 with a FedProx proximal term (μ = 1); our re-implementation"],
     ["ph_adapter_fed", "PH + federated adapter", "Alarmed clients train and average a 1,966-parameter adapter"],
     ["ph_full_local", "PH + local full retraining", "Retrains the full model locally, no upload"],
     ["ph_align", "Input alignment (fixed)", "Always the closed-form sensor correction"],
     ["ph_head_long", "Head retraining (fixed)", "Always retrains the classifier head with replay"],
     ["E8 / typed_learned", "TLA (typed local adaptation)", "Types the drift (virtual/real), then align or head; no upload"],
     ["ph_mode_oracle", "TLA with true type", "Same, but given the true type (prices the typing errors)"],
     ["mode_typed_oracle / typed_oracle", "Oracle 2-way / 4-way", "True alarm time and type; reference, not deployable"]], [1.5, 2.0, 4.2])
P("<b>Drift kinds.</b> <i>Virtual</i>: the sensor signal changes (offset 1 g, gain ×1.3, rotation 20°), the meaning of activities does not. "
  "<i>Real</i>: two pairs of activity labels swap meaning. <b>Scenarios.</b> S0 none · S1 sudden at round 30 · S2 gradual 10-round ramp "
  "(onsets 30–40) · S3 recurring (on 30–44, 60–74, 90–104) · S4 one sensor only. <b>Seeds.</b> Tuning 1001–1003, validation 1004–1010, "
  "evaluation 7, 11, 13, 17, 19, 23, 42, 123, 999, 2024 (untouched until Phase 6).")
story.append(PageBreak())

# =====================================================================================================  2 phase 0
H1("2. Phase 0 — Audit of the first prototype (26 Oct)")
P("<b>Why.</b> The project started from a prototype that reported 96.5% accuracy, 100% ADWIN detection and 61% drift-type accuracy. "
  "Before building on it we audited every part.")
TAB([["Finding in the prototype", "Consequence", "Fix"],
     ["Federated 'training' was a mock (loss = a variance, accuracy = a label-overlap heuristic)", "No FL number was real", "Native PyTorch FL engine with real training and byte counting (Phase 4)"],
     ["Wrong sensor columns: IMU temperature instead of accelerometer axes", "Temperature leaks the label (sensor warm-up): temperature alone scored 0.836", "Correct ±16 g accelerometer columns 4–6, 21–23, 38–40"],
     ["Running-average normalisation over the whole stream", "Test data leaks into training; offset drift is masked", "Train-split statistics, frozen"],
     ["Windows straddle activity changes; temporal split loses classes", "Optimistic or broken evaluation", "Bout-based windowing; blocked split with purge"],
     ["Only virtual offset drift, re-sampled every round, no ground truth", "Detection could not be scored", "Persistent drift with a change-point manifest; real drift added (Phase 2)"],
     ["No Page–Hinkley or label-free detectors; detector never wired into FL", "No end-to-end system", "Detector suite (Phase 3) and streaming engine (Phase 4)"],
     ["Drift classifier 0% on sudden drift (synthetic error streams only)", "Typing claim unsupported", "Classifier rebuilt on real model-error streams (Phase 3)"],
     ["Model size quoted as 77K parameters / 308 KB", "Wrong numbers in the plan", "Measured: 53,260 parameters, 213,040 B (12 classes, 2-layer LSTM)"]], [2.6, 2.2, 2.6])
SRC("docs/decisions.md §2–3; legacy/README.md.")
BOX("<b>Conclusion.</b> None of the prototype's numbers could be trusted, so all were discarded. Everything below comes from the rebuilt "
    "pipeline. Foundations: snapshot commit, retired mock code to <i>legacy/</i>, pinned environment (PyTorch cu128 on an RTX 5050), test suite.")

# =====================================================================================================  3 phase 1
H1("3. Phase 1 — Clean data pipeline and baselines (26–27 Oct)")
H2("3.1 Baselines on stationary data (protocol v1)")
P("<b>Why.</b> Before studying drift we needed honest numbers for a model with <i>no</i> drift, and to know which baseline the "
  "proposed system must beat. <b>What.</b> PAMAP2 subjects 101–108 (109 excluded: a single 64 s bout of one class) as 8 clients, "
  "9 accelerometer channels, 256-sample windows (stride 128), two splits: <i>blocked</i> (random 16-window blocks: stationary control) and "
  "<i>temporal</i> (last 25% of each bout: a natural-drift probe). Locked budget: FedAvg 60 rounds × 2 local epochs; E1/E2 120 epochs; 5 seeds.")
TAB([["Method", "Blocked split (control)", "Temporal split (probe)"],
     ["E1 Local-only", "0.883 ± 0.003", "0.815 ± 0.005"],
     ["E2 Centralised", "0.871 ± 0.008", "0.810 ± 0.004"],
     ["E3 FedAvg", "0.851 ± 0.015", "0.774 ± 0.013"],
     ["E3b FedAvg + local fine-tuning", "0.888 ± 0.009", "0.815"]], [2.5, 2, 2],
    cap="Test macro-F1, mean ± std over 5 seeds. Source: results/phase1/summary.md, results/phase1/fedavg_ft_*.json.")
BUL(["The prototype's '95% accuracy' does not survive a leak-free protocol; realistic stationary macro-F1 is about 0.85–0.89.",
     "Personal models beat pooled ones on within-person data; plain FedAvg is the weakest (worst client 0.68, client 103 with only 8 classes).",
     "<b>FedAvg + local fine-tuning (E3b) closes the whole gap</b> (E3b − E1 = +0.005, CI [−0.002, +0.012]). So every later claim must be made against E3b, not against plain FedAvg."])
H2("3.2 Is the temporal drop natural drift? (Phase 1b)")
P("Every method loses 6–8 points on the temporal split with no injected drift. The diagnosis showed this is <b>mostly not drift</b>: the first and "
  "last quarter of every activity bout are intrinsically hard (error 0.17/0.14 vs 0.04–0.06 mid-bout), and the temporal test set is exactly the "
  "last quarter. Genuine covariate shift is small; class frequency and sensor warm-up are not causes. <b>Conclusion:</b> the temporal split is not "
  "presented as evidence of natural drift; injected drift with ground truth is the main evidence.")
H2("3.3 Removing bout edges: protocol v2 (Phase 1d)")
P("<b>Why.</b> Bout-edge windows are hard for every method and would cause false drift alarms. The rule was fixed before looking at results: trim "
  "5 s at both ends of every bout, because PAMAP2 labels are only approximately aligned at transitions. Validation: 93–95% of windows kept per "
  "client, no client × class pair lost, covariate content preserved (MMD² 0.038, below the 0.071 train/test reference).")
TAB([["Method", "Blocked v1 → v2", "Temporal v1 → v2"],
     ["E1 Local-only", "0.883 → 0.939", "0.815 → 0.895"],
     ["E2 Centralised", "0.871 → 0.930", "0.810 → 0.884"],
     ["E3 FedAvg", "0.851 → 0.893", "0.774 → 0.862"],
     ["E3b FedAvg + FT", "0.888 → 0.935", "0.815 → 0.903"]], [2.5, 2, 2],
    cap="Test macro-F1 under the untrimmed (v1) and trimmed (v2) protocol, 5 seeds. Source: results/phase1d/comparison.md.")
BOX("<b>Conclusion.</b> Trimming raises every method by 4–6 points on the blocked split (and 8–9 on the temporal split) and changes <b>no ranking</b> "
    "(0 ranking changes under a rule fixed in advance). v2 became the primary protocol; v1 is kept as a sensitivity analysis. v1 and v2 numbers are never mixed in one table.")
story.append(PageBreak())

# =====================================================================================================  4 phase 2
H1("4. Phase 2 — Drift generator and severity calibration (27 Oct)")
P("<b>Why.</b> To score detection and adaptation we need drift whose start, end and type are known exactly. <b>What.</b> A generator that "
  "produces persistent drift events = (profile, mode, transform, sensors, coverage, timing) and writes a ground-truth change-point manifest. "
  "Virtual transforms act on the raw signal in g; real drift relabels disjoint class pairs. 24 unit tests (e.g. rotation preserves vector length, "
  "real drift never touches inputs, determinism).")
TAB([["Transform", "Scope", "Low", "Medium", "High"],
     ["Offset (0.5 / 1 / 2 g)", "all sensors", "0.913", "0.797", "0.462"],
     ["Offset", "hand only", "0.932", "0.912", "0.783"],
     ["Gain (×1.15 / 1.3 / 1.6)", "all sensors", "0.911", "0.813", "0.656"],
     ["Gain", "ankle only", "0.923", "0.825", "0.612"],
     ["Rotation (10 / 20 / 30°)", "all sensors", "0.880", "0.678", "0.421"],
     ["Rotation", "chest only", "0.892", "0.746", "0.563"],
     ["Label swap (2 / 4 / 6 classes)", "real drift", "0.761", "0.584", "0.420"]], [2.6, 1.4, 0.9, 0.9, 0.9],
    cap="Macro-F1 of a frozen local model at full drift strength (clean 0.940), tuning seed 1001. Source: results/phase2/severity_check.md.")
BUL(["All transforms degrade monotonically low > medium > high; <b>medium</b> severity is used from here on.",
     "Rotation was first set to 15/30/60°; 60° collapsed the model (0.12), so it was reduced to 10/20/30° — on a tuning seed, recorded as an amendment.",
     "Hand-only offset is nearly invisible (0.912 at medium): the subtle end for detectors.",
     "A bug found later (Phase 3) was fixed: the arrival stream had shuffled single windows (unrealistically easy); it now arrives in activity blocks, about 2–3 labelled windows per client and round."])
FIG("results/phase2/stream_s3_recurring.png", "Example drift schedule from the generator (S3 recurring): which clients drift, when, and how strongly. Source: results/phase2/.", 0.72)
story.append(PageBreak())

# =====================================================================================================  5 phase 3
H1("5. Phase 3 — Drift detectors and drift-type classifiers (27 Oct)")
H2("5.1 Detectors")
P("<b>Why.</b> Research question 1: how well can a client notice drift, and how many labels does that need? <b>What.</b> Eight detectors, tuned on "
  "seeds 1001–1003 under a fixed bound of at most 1 false alarm per 1,000 clean windows and validated on seeds 1004–1010. A <i>placebo recall</i> "
  "(the same onsets applied to a clean stream) measures how often a detector 'hits' by chance.")
TAB([["Detector", "Labels / 1,000 windows", "Recall (20 rounds)", "Chance (placebo)", "Mean delay (rounds)", "False alarms / 1,000"],
     ["D2 Page–Hinkley on error bit", "1000", "<b>0.42</b>", "0.05", "8.6", "0.48"],
     ["D2c PH, class-conditional", "1000", "0.42", "0.05", "8.4", "0.47"],
     ["D2s PH with 20% labels", "200", "0.34", "0.03", "10.0", "0.56"],
     ["D1 ADWIN on error bit", "1000", "0.28", "0.00", "14.5", "0.20"],
     ["D5 two-level (label-free trigger + check)", "454", "0.32", "0.04", "11.7", "0.88"],
     ["D5ns two-level, no sentinel", "289", "0.26", "0.04", "11.7", "0.86"],
     ["D3 entropy (label-free)", "0", "0.21", "0.09", "9.3", "1.32 (over bound)"],
     ["D4 class-matched MMD (label-free)", "0", "0.14", "0.05", "11.3", "1.40 (over bound)"]], [3.0, 1.1, 1.0, 1.0, 1.0, 1.2], hl=[1],
    cap="Validation, benchmark v2, medium severity. Source: results/phase3_v2/detector_report.md (v1 numbers were nearly identical: D2 0.42 / 9.2 rounds / 0.39).")
FIG("docs/paper/figs/detectors_tradeoff.png", "Recall against false alarms; red = label-free, marker size = labels used; dotted line = the registered false-alarm bound.", 0.62)
BUL(["<b>Page–Hinkley on the model's own errors is the best detector</b>, but finds only 42% of onsets within 20 rounds.",
     "Label-free detectors are weak and break the false-alarm bound; they are almost blind to real drift (recall ≤ 0.09), as expected: swapping labels changes no input statistic.",
     "The two-level design is <b>not</b> better than plain Page–Hinkley with 20% of the labels (0.32 at 454 labels vs 0.34 at 200): this label-efficiency claim was dropped.",
     "Disclosed: after a first weak run, the design was changed once (class-conditional baseline, wider grids, placebo recall), so seeds 1004–1010 are no longer a pristine validation set. The evaluation seeds were untouched."])
H2("5.2 Can a client tell what kind of drift it is?")
P("<b>4-way type</b> (sudden / gradual / recurring / one-sensor) from a detected alarm, first benchmark: gradient boosting accuracy 0.46 and "
  "macro-F1 0.32, <b>below the majority class</b> (0.51); recall sudden 0.72, gradual 0.22, recurring 0.19, one-sensor 0.13. With oracle alarm "
  "times it reaches 0.63 / 0.59, so late alarms are a main cause. The first onset of a recurring drift is by construction identical to a sudden one.")
TAB([["Classifier (virtual vs real)", "Accuracy", "Recall virtual", "Recall real"],
     ["Majority class", "0.602", "1.00", "0.00"],
     ["Threshold rule", "0.690", "0.65", "0.75"],
     ["Logistic regression (label-free)", "0.710", "0.73", "0.68"],
     ["<b>Gradient boosting (label-free)</b>", "<b>0.839</b>", "0.92", "0.72"],
     ["Gradient boosting (all features)", "0.857", "0.92", "0.76"]], [3, 1, 1, 1], hl=[4],
    cap="2-way mode from detected alarms, 1,099 episodes, leave-one-seed-out over seeds 1001–1010 (benchmark v2). Source: results/phase3_v2/mode_typing.md (v1: 0.845).")
BOX("<b>Conclusion.</b> The 4-way type is not identifiable from real alarms, but virtual vs real <b>is</b> (84% accuracy) using label-free features. "
    "Its weak spot: 28% of real drifts are called virtual. The deployable system therefore types 2-way only; the 4-way type stays an oracle ablation. "
    "Author decision D15: go on and let the downstream test (does typing beat a fixed response? = H5) decide.")
story.append(PageBreak())

# =====================================================================================================  6 phase 4
H1("6. Phase 4 — Streaming FL engine, adapter and local repairs (27 Oct, benchmark v1)")
P("<b>Why.</b> Detection and typing only matter if the response works end to end. <b>What.</b> A streaming federated simulator: each client "
  "starts from the personalised model E3b, receives about 3 labelled windows per round for 60 rounds, may raise alarms, run a response and upload; "
  "every byte is counted from the real serialised payload. An adapter module (input map 90 + bottleneck 1,096 + head 780 = <b>1,966 parameters, "
  "7,864 B, 3.7% of the model</b>) reproduces the base model exactly at initialisation.")
H2("6.1 Design findings from the first runs")
BUL(["<b>Fine-tuning on recent windows collapses the model</b> (0.50–0.57 vs 0.89): the stream arrives in activity blocks, so the buffer holds only 2–4 classes. Fix: replay clean training windows; for real drift, replay only classes absent from the buffer.",
     "<b>Gradient training of an input layer fails</b>: the model is already right on the seen classes, so the loss carries almost no signal about the sensor transform.",
     "<b>New: closed-form input alignment.</b> A virtual drift is a class-independent transform, so per sensor a ridge-regularised least-squares fit maps the drifted class means onto the stored clean class means. It recovers offset, gain and rotation from a few labelled windows, with no gradient step and no communication (10 unit tests with known transforms)."])
H2("6.2 First look: S1 sudden drift (tuning seeds 1001–1003)")
TAB([["Variant", "Frozen E3b", "Periodic E3c", "Oracle align", "PH + full fed (E5)", "PH + align", "PH + head"],
     ["gain", "0.762", "0.720", "<b>0.884</b>", "0.744", "0.780", "—"],
     ["rotation", "0.682", "0.688", "<b>0.845</b>", "0.672", "0.752", "—"],
     ["offset (partial)", "0.892", "0.844", "<b>0.909</b>", "0.869", "0.879", "—"],
     ["offset (all clients)", "0.873", "0.843", "<b>0.880</b>", "0.850", "0.867", "—"],
     ["real (label swap)", "0.622", "0.610", "head 0.675", "0.624", "—", "0.666"]], [1.5, 1, 1, 1, 1.2, 1, 0.9],
    cap="Post-change macro-F1, 3 tuning seeds, descriptive. Source: results/phase4/summary.md. Uplink: full federated retraining 1.3–5.1 MB, periodic 8.5 MB, align and head 0 B.")
BUL(["With a perfect (oracle) trigger, alignment recovers <b>+12 points</b> (gain) and <b>+16 points</b> (rotation).",
     "With the real Page–Hinkley trigger the gain shrinks to +2 / +7: <b>detection is the bottleneck</b>.",
     "Full-model federated retraining is no better than doing nothing and costs 18–70× the bytes; periodic retraining hurts clean clients (0.88 vs 0.95).",
     "Real drift is hard: the best response (head retraining with replay) gains about +5 points."])
H2("6.3 Typed vs fixed responses, and the deployable system (benchmark v1)")
BUL(["With oracle types, a single fixed response fails somewhere: alignment collapses on real drift (S3 real 0.552 vs 0.784 frozen), head retraining does not fix virtual drift.",
     "4-way oracle typing adds +0.033 over 2-way on average (gradual tracking, recurring revert/restore, one-sensor restriction): typing has a real ceiling value.",
     "Deployable systems, mean over 13 variants (3 tuning seeds): frozen E3b 0.801 · PH + full fed 0.794 · PH + align 0.789 · PH + head 0.805 · <b>TLA 0.807</b> · PH + oracle mode 0.812. <b>Every deployable system was within about 1 point of doing nothing.</b>"])
BOX("<b>Conclusion and decision D16.</b> The response works (oracle trigger: +12 to +16 points), but with 60-round streams the detection delay "
    "(~9 rounds) plus the typing delay (5 rounds) leaves too little time for any repair to pay off. Author chose to lengthen the benchmark "
    "<b>before</b> touching the evaluation seeds (Amendment A4): 120 rounds, onset at round 30, and an ADWIN-triggered baseline (E4) added.")
story.append(PageBreak())

# =====================================================================================================  7 phase 4b
H1("7. Phase 4b — Benchmark v2 on tuning seeds (28 Oct)")
P("<b>What changed.</b> Streams of 120 rounds (arrival pool replayed twice in a new random block order), onset at round 30; S2 asynchronous "
  "onsets 30–40 with a 10-round ramp; S3 on at 30/60/90 and off at 45/75/105. Detectors and the type classifier were re-tuned on v2 (results above).")
TAB([["System", "Post-change macro-F1", "Uplink"],
     ["Frozen E3b (Personalized)", "0.735", "0"],
     ["E3c Periodic full retraining", "0.737", "18.7 MB"],
     ["E4 ADWIN + full retraining", "0.735", "2.7 MB"],
     ["E5 PH + full retraining", "0.730", "4.8 MB"],
     ["Input alignment (fixed)", "0.725", "0"],
     ["Head retraining (fixed)", "0.777", "0"],
     ["<b>TLA (typed, deployable)</b>", "<b>0.783</b>", "0"],
     ["TLA with true type", "0.808", "0"],
     ["Oracle 4-way typing", "0.827", "0"]], [3, 1.5, 1.2], hl=[7],
    cap="Mean over 10 variants, 3 tuning seeds (descriptive). Source: results/phase4_v2/summary.md. Note: the oracle value 0.827 asked about in the supervisor feedback is this tuning-seed mean, not an evaluation-seed number.")
BUL(["The deployable typed system now gains <b>+4.8 points</b> over frozen (v1: about +0.6).",
     "Full-model federated retraining (E3c, E4, E5) gives no gain over frozen at 4–25× the bytes.",
     "Alignment alone is not better than frozen (it hurts real drift: 0.42–0.54); the gain comes from choosing alignment for virtual and head retraining for real drift.",
     "FedProx strength for E7a was tuned here: μ = 0.01 → 0.666, 0.1 → 0.671, <b>1.0 → 0.695</b> (fixed at 1.0; E5 0.659, frozen 0.652)."])
BOX("<b>Then the protocol was frozen</b> (Amendment A5): 5 hypotheses, fixed comparators, Wilcoxon + Holm over 12 tests, 10 evaluation seeds. No "
    "further benchmark change was allowed.")
story.append(PageBreak())

# =====================================================================================================  8 phase 6
H1("8. Phase 6 — Pre-registered confirmatory test (28 Oct, 10 evaluation seeds)")
P("<b>Design.</b> 14 systems × 10 drift variants (S1: 4, S2–S4: 2 each) × 10 evaluation seeds = <b>1,400 streams</b>. Unit of analysis = seed. "
  "Paired two-sided Wilcoxon signed-rank tests with a 10,000-resample bootstrap 95% CI; Holm correction over the 12 tests of H1, H2, H3, H5 on S1–S3; "
  "H4 is a non-inferiority test with a 1-point margin. With 10 seeds the smallest possible Holm-corrected p is 0.0234, so p-values are the primary evidence.")
H2("8.1 All systems")
TAB([["System", "S1 sudden", "S2 gradual", "S3 recurring", "S4 one sensor", "Uplink S1 (kB)"],
     ["Global (no adaptation) E3", "0.658 ± 0.023", "0.590 ± 0.029", "0.737 ± 0.016", "0.635 ± 0.030", "0"],
     ["Personalized (no adaptation) E3b", "0.677 ± 0.022", "0.612 ± 0.031", "0.761 ± 0.019", "0.681 ± 0.034", "0"],
     ["Periodic full retraining E3c", "0.710 ± 0.026", "0.635 ± 0.021", "0.744 ± 0.020", "0.726 ± 0.024", "18,748"],
     ["ADWIN + full retraining E4", "0.699 ± 0.021", "0.642 ± 0.022", "0.733 ± 0.021", "0.726 ± 0.039", "2,269"],
     ["PH + full retraining E5", "0.704 ± 0.015", "0.647 ± 0.027", "0.735 ± 0.014", "0.721 ± 0.027", "3,387"],
     ["PH + FedProx E7a", "0.738 ± 0.014", "0.669 ± 0.019", "0.763 ± 0.010", "0.761 ± 0.024", "3,174"],
     ["PH + federated adapter", "0.710 ± 0.021", "0.649 ± 0.023", "0.746 ± 0.016", "0.726 ± 0.022", "127"],
     ["PH + local full retraining", "0.704 ± 0.022", "0.647 ± 0.029", "0.725 ± 0.015", "0.722 ± 0.035", "0"],
     ["Input alignment (fixed)", "0.707 ± 0.027", "0.616 ± 0.051", "0.637 ± 0.021", "<b>0.820</b> ± 0.043", "0"],
     ["Head retraining (fixed)", "0.753 ± 0.022", "<b>0.707</b> ± 0.027", "<b>0.766</b> ± 0.014", "0.753 ± 0.019", "0"],
     ["<b>TLA (typed) E8</b>", "<b>0.756</b> ± 0.050", "0.691 ± 0.077", "0.711 ± 0.051", "0.810 ± 0.044", "0"],
     ["TLA with true type", "0.783 ± 0.028", "0.744 ± 0.029", "0.734 ± 0.019", "0.819 ± 0.041", "0"],
     ["Oracle 2-way typing", "0.779 ± 0.049", "0.717 ± 0.035", "0.731 ± 0.020", "0.811 ± 0.061", "0"],
     ["Oracle 4-way typing", "0.779 ± 0.049", "0.762 ± 0.031", "0.797 ± 0.018", "0.863 ± 0.043", "0"]], [2.6, 1.15, 1.15, 1.15, 1.15, 1.0], hl=[11],
    cap="Post-change macro-F1 of drifted clients, mean ± std over 10 seeds; bold = best deployable value per column. Source: results/phase6/confirmatory.md. Oracle rows use true alarm times and types; oracle 2-way uses one-shot alignment, so it can fall below 'TLA with true type'.")
FIG("docs/paper/figs/recovery_curves.png", "Macro-F1 of drifted clients per round (5-round moving average; dotted line = onset). Left: sudden rotation; middle: sudden label swap; right: recurring rotation.", 0.98)
H2("8.2 The 12 confirmatory tests")
TAB([["Hypothesis (TLA vs …)", "Scenario", "Difference", "95% CI", "Holm p", "Verdict"],
     ["H1 Personalized (no adaptation)", "S1", "+0.079", "[+0.055, +0.105]", "0.023", "<b>TLA better</b>"],
     ["", "S2", "+0.080", "[+0.040, +0.120]", "0.078", "not significant"],
     ["", "S3", "−0.050", "[−0.074, −0.029]", "0.023", "<b>TLA worse</b>"],
     ["H2 Periodic full retraining", "S1", "+0.046", "[+0.008, +0.085]", "0.633", "not significant"],
     ["", "S2", "+0.057", "[+0.003, +0.109]", "0.588", "not significant"],
     ["", "S3", "−0.033", "[−0.070, +0.002]", "0.967", "not significant"],
     ["H3 PH + FedProx", "S1", "+0.018", "[−0.008, +0.047]", "1.000", "not significant"],
     ["", "S2", "+0.023", "[−0.021, +0.063]", "1.000", "not significant"],
     ["", "S3", "−0.052", "[−0.084, −0.025]", "0.035", "<b>TLA worse</b>"],
     ["H5 Head retraining (fixed)", "S1", "+0.004", "[−0.019, +0.026]", "1.000", "not significant"],
     ["", "S2", "−0.016", "[−0.054, +0.019]", "1.000", "not significant"],
     ["", "S3", "−0.055", "[−0.082, −0.031]", "0.023", "<b>TLA worse</b>"]], [2.6, 0.8, 0.9, 1.5, 0.8, 1.3],
    cap="Paired Wilcoxon, Holm over 12 tests. Source: results/phase6/confirmatory.md.")
FIG("docs/paper/figs/forest_confirmatory.png", "The 12 tests with bootstrap 95% CIs. Green = significant gain, red = significant loss, grey = not significant.", 0.62)
H2("8.3 H4: is a small adapter enough?")
TAB([["Scenario", "PH + full retraining F1", "Adapter F1", "Full − adapter [95% CI]", "Uplink full / adapter (kB)"],
     ["S1", "0.704", "0.710", "−0.005 [−0.015, +0.005]", "3,387 / 127"],
     ["S2", "0.647", "0.649", "−0.002 [−0.013, +0.008]", "3,888 / 142"],
     ["S3", "0.735", "0.746", "−0.011 [−0.022, −0.000]", "2,929 / 112"]], [0.8, 1.5, 1, 2, 1.8],
    cap="Non-inferior in all three scenarios (every CI upper bound below the +0.010 margin); 96% less uplink. Source: results/phase6/confirmatory.md.")
FIG("docs/paper_ieee/figs/cost_vs_uplink.png", "Accuracy against uplink (mean over S1–S3). Local systems (no upload) are drawn at 1 kB.", 0.62)
H2("8.4 Detection in the full streams, wrong types, clean users, typing value")
BUL(["<b>Detection:</b> TLA alarmed on 262 of 300 drifted clients at some point, with a median delay of <b>23 rounds</b> from onset (validation delay was 8.6; real onsets are often gradual or partial).",
     "<b>Wrong type is costly:</b> alignment applied to the real-drift variants scores 0.40 / 0.42 / 0.55 on S1 / S2 / S3 vs 0.58 / 0.60 / 0.76 with no adaptation (−0.18 to −0.21). Head retraining on a virtual drift is harmless but recovers less.",
     "<b>Clean users:</b> periodic federated retraining lowers the F1 of the 5 non-drifted clients from 0.944 to 0.895 under S1 (≈5 points; 4.4–5.4 across variants). Alarm-triggered retraining costs 0.1–0.4 points; TLA 0.1.",
     "<b>Price of typing errors:</b> TLA vs TLA with the true type: −0.027 (S1), −0.053 (S2), −0.023 (S3). Oracle 4-way beats oracle 2-way by +0.045 (S2), +0.066 (S3), +0.052 (S4), tie on S1 — upper references only."])
FIG("docs/paper_ieee/figs/mistype_cost.png", "Real-drift variants: alignment (the virtual-drift repair) is worse than no adaptation; head retraining recovers accuracy.", 0.6)
FIG("docs/paper/figs/heatmap_policy_variant.png", "Post-change macro-F1 per system and drift variant (mean over 10 seeds).", 0.95)
BOX("<b>Phase 6 conclusion.</b> Supported: H1 on sudden drift (+0.079) and H4 (adapter non-inferior, 96% less uplink). Not supported: H2, H3 and "
    "<b>H5 — typing does not beat a fixed head-retraining rule</b>. On recurring drift TLA is significantly worse than no adaptation, FedProx and "
    "head retraining. Author decision D17: stop experiments and write the paper (later reopened for two follow-ups, D18).")
story.append(PageBreak())

# =====================================================================================================  9 phase 7 memory
H1("9. Phase 7 — Why TLA fails on recurring drift, and a memory (22 Nov)")
P("<b>Why.</b> A collaborator review asked for work on recurring drift and on edge-device memory. Decision D18 reopened experiments on a separate "
  "branch; Amendment A6 was registered before any code.")
H2("9.1 Tracing the S3 failure (from the cached Phase 6 runs, no rerun)")
TAB([["System", "Rotation: drift on", "Rotation: drift off", "Real: drift on", "Real: drift off"],
     ["TLA (E8)", "0.692", "0.780", "0.584", "0.788"],
     ["Personalized (no adaptation) E3b", "0.598", "0.932", "0.581", "0.932"]], [2.6, 1.2, 1.2, 1.2, 1.2],
    cap="S3 macro-F1 of drifted clients split by phase (mean over seeds). Source: docs/decisions.md D18.")
BUL(["While the drift is on, TLA beats no adaptation; <b>between episodes it is far worse</b> (0.780 vs 0.932).",
     "<b>No revert:</b> the fitted map or head stays in place after the drift ends and is applied to clean signals.",
     "<b>Late responses:</b> 18 of 36 (rotation) and 20 of 38 (real) responses start after the 15-round episode has already ended."])
FIG("docs/paper/figs/s3_offphase.png", "S3: macro-F1 between drift episodes. Every response that keeps its adaptation falls below no adaptation.", 0.62)
H2("9.2 A deployable memory (tuning seeds)")
P("Each client keeps up to K cached adapter states. At an alarm, after the typing delay, the candidates (original model = undo, current state, cached "
  "states) are scored on recent labelled windows; the client switches if the best is within a tolerance of its normal error, otherwise TLA runs and "
  "caches its result. Grid: K in {0, 1, 2, 4} × 3 tolerances × 2 scoring windows (since the alarm / last 3 rounds) = 24 configurations, S3 rotation and real, seeds 1001–1003.")
BUL(["Best configuration 0.736 vs TLA 0.735: <b>no gain</b>. All K gave identical results (a cached state was never chosen); the registered rule selected K = 0, tolerance 0.02 (≈ 'undo only').",
     "Scoring since the alarm: 13 responses fell in the drift-off phase but <b>0 undos</b> (the window lies inside the episode).",
     "Scoring on the last 3 rounds: 25 switches instead of 2, <b>7 correct undos but 6 wrong ones</b> while the drift was still on; worse overall (0.721).",
     "<b>Root cause:</b> about 2 labelled windows per round, arriving in activity blocks, cannot separate drift-on from drift-off within a few rounds — the same limit that slows detection."])
H2("9.3 On-device memory budget (computed, no training)")
TAB([["Component", "Size (FP32)"],
     ["Deployed LSTM (53,260 parameters)", "208.0 KiB"],
     ["Adapter (1,966 parameters) / input map only", "7.7 KiB / 0.4 KiB"],
     ["Page–Hinkley detector state", "64 B"],
     ["Labelled buffer: align (16 rounds) / head (45 rounds) / federated (15 rounds)", "288 / 810 / 270 KiB"],
     ["Replay pool (median client, 966 windows)", "8,694 KiB"],
     ["Memory cache K = 1 / 2 / 4", "7.7 / 15.4 / 30.7 KiB"],
     ["Peak: Personalized (frozen)", "208 KiB"],
     ["Peak: PH + full retraining / FedProx", "9,380 KiB"],
     ["Peak: PH + federated adapter", "9,180 KiB"],
     ["Peak: TLA / TLA + memory (K = 4)", "9,720 / 9,751 KiB"]], [4.5, 1.6],
    cap="Source: results/phase7/memory_budget.md (experiments/phase7_memory_budget.py).")
BOX("<b>Conclusion.</b> Memory itself is cheap (under 2% of the device budget even at K = 4); the replay pool dominates RAM. Devices with less than "
    "1 MB would need synthetic exemplars or frozen features instead of raw replay. Memory brings no accuracy gain with sparse labels.")
story.append(PageBreak())

# =====================================================================================================  10 phase 7 DP
H1("10. Phase 7 — Differential privacy on federated uploads (22 Nov)")
P("<b>Why.</b> Local adaptation uploads nothing, but the federated adapter and full retraining do. Can those uploads be made differentially private? "
  "<b>What (Amendment A6 M2).</b> Client-level Gaussian mechanism: each update (relative to the start of its episode) is clipped to L2 norm C and "
  "noise N(0, (zC)²) is added before upload, z in {0, 0.5, 1, 2, 4}. C fixed on tuning seeds as the median unclipped update norm, with no accuracy-based "
  "tuning: adapter 6.10, full model 2.59. Privacy reported as (ε, δ = 10⁻⁵) per drifted client over the whole stream, Rényi-DP accountant, no "
  "subsampling amplification. 640 runs (2 systems × 4 noise levels × 8 variants × 10 evaluation seeds), descriptive.")
TAB([["System", "z", "ε per client", "Uploads / drifted client", "Drifted-client F1", "Clean-client F1"],
     ["PH + federated adapter", "0", "∞", "~6 (tuning)", "0.704 ± 0.018", "0.940"],
     ["", "0.5", "55.2", "3.0", "0.276 ± 0.043", "0.888"],
     ["", "1", "21.8", "3.0", "0.274 ± 0.044", "0.887"],
     ["", "2", "9.4", "3.0", "0.274 ± 0.045", "0.887"],
     ["", "4", "4.3", "3.0", "0.275 ± 0.044", "0.887"],
     ["PH + full retraining", "0", "∞", "~7 (tuning)", "0.698 ± 0.015", "0.940"],
     ["", "0.5", "52.5", "2.8", "0.292 ± 0.039", "0.890"],
     ["", "1 – 4", "21.4 – 4.2", "3.0", "0.261 ± 0.043", "0.887"],
     ["TLA (no upload)", "—", "0", "0", "0.729 ± 0.054", "0.943"],
     ["Personalized (no adaptation)", "—", "0", "0", "0.682 ± 0.023", "0.944"]], [2.4, 0.6, 1.0, 1.3, 1.4, 1.2],
    cap="S1–S3 pooled, 10 seeds. Source: results/phase7/dp_tradeoff.md, results/phase4_v2/dp_eval_*.json.")
FIG("docs/paper_ieee/figs/dp_cliff.png", "Accuracy against ε. Squares = no noise. Any noise gives a cliff, not a trade-off.", 0.62)
BUL(["Every noisy setting collapses accuracy from about 0.70 to about 0.27, far below doing nothing (0.68); noise also reaches clean clients through the shared model.",
     "<b>Not a bug:</b> the z = 0 logging path reproduces the plain run exactly (unit test).",
     "<b>Why:</b> noise of scale zC on each of d parameters has norm zC√d — 22C for the adapter (d = 1,966) and 115C for the full model at z = 0.5 — while a clipped update has norm at most C. Averaging 8 clients shrinks noise only by √8 ≈ 2.8. Client-level DP of this kind needs hundreds of clients per round."])
BOX("<b>Conclusion.</b> At wearable cohort sizes, client-level DP on uploads has no useful operating point. Keeping updates on the device is the only "
    "setting we tested that keeps both accuracy and an unspent privacy budget (a property of the design, not a formal guarantee for local computation).")
story.append(PageBreak())

# =====================================================================================================  11 supervisor runs
H1("11. Supervisor follow-ups (24 Nov, 10 evaluation seeds)")
H2("11.1 S0: what happens when nothing drifts?")
P("<b>Why.</b> Phase 6 had no end-to-end no-drift run (false alarms came only from detector validation). The runner was extended to support S0, and "
  "7 systems were run on 10 evaluation seeds (70 streams).")
TAB([["System", "Alarms / stream (8 clients)", "Whole-stream F1", "Change vs Personalized [95% CI]", "Uplink (kB)"],
     ["Personalized (no adaptation)", "0", "0.939", "—", "0"],
     ["TLA (typed)", "0.9", "0.938", "−0.0010 [−0.0023, −0.0001]", "0"],
     ["Head retraining (fixed)", "1.0", "0.938", "−0.0009 [−0.0020, −0.0002]", "0"],
     ["Input alignment (fixed)", "1.4", "0.934", "−0.0051 [−0.0107, −0.0011]", "0"],
     ["PH + federated adapter", "1.5", "0.936", "−0.0032 [−0.0071, −0.0002]", "34"],
     ["PH + full retraining", "1.2", "0.936", "−0.0032 [−0.0054, −0.0012]", "724"],
     ["Periodic full retraining", "88 (scheduled)", "0.899", "<b>−0.0407</b> [−0.0475, −0.0345]", "18,748"]], [2.4, 1.4, 1.1, 2.2, 1.0], hl=[7],
    cap="S0, 10 evaluation seeds. Source: results/phase7/s0_mem.md (experiments/phase7_s0_mem.py).")
BOX("<b>Conclusion.</b> Page–Hinkley raises about one false alarm per 8-client, 120-round stream (consistent with ~0.5 per 1,000 windows in validation), "
    "and the repairs triggered by these false alarms cost at most half a point. Periodic federated retraining loses 4.1 points and uploads 18.7 MB although nothing changed.")
H2("11.2 Memory on recurring drift, evaluation seeds")
P("The configuration selected on tuning seeds (K = 0, tolerance 0.02) and a larger memory (K = 4) were run on both S3 variants with all 10 evaluation "
  "seeds (40 streams) and compared, seed by seed, with TLA from Phase 6.")
TAB([["System", "Switches (20 streams)", "Post-change F1 (TLA)", "Difference [95% CI], p", "Between episodes (TLA)", "Difference, p"],
     ["TLA + memory, K = 0", "8", "0.709 (0.711)", "−0.002 [−0.011, +0.003], p = 0.38", "0.785 (0.784)", "+0.001, p = 0.38"],
     ["TLA + memory, K = 4", "8", "0.709 (0.711)", "identical to K = 0", "0.785 (0.784)", "identical"]], [1.7, 1.1, 1.2, 2.0, 1.2, 1.1],
    cap="S3 rotation + real, 10 seeds, paired Wilcoxon (descriptive). Source: results/phase7/s0_mem.md.")
FIG("docs/paper_ieee/figs/recurring.png", "S3 recurring: F1 between episodes and over the whole post-change period, including TLA + memory.", 0.7)
BOX("<b>Conclusion.</b> The negative tuning-seed result is confirmed on the evaluation seeds: the memory almost never switches and changes nothing.")
story.append(PageBreak())

# =====================================================================================================  12 conclusions
H1("12. What we can conclude")
TAB([["#", "Finding", "Evidence", "Strength"],
     ["1", "Sending a small adapter (1,966 params) instead of the full model loses no accuracy and cuts uplink by 96% (mostly because the adapter is 3.7% of the model).", "H4 non-inferior in S1–S3", "Confirmatory"],
     ["2", "Local repair protects users who did not drift; periodic federated retraining lowers their F1 by ~5 points, and by 4.1 points even with no drift.", "Phase 6 clean-client F1; S0", "Descriptive"],
     ["3", "TLA beats doing nothing on sudden drift (+0.079).", "H1 S1, Holm p = 0.023", "Confirmatory"],
     ["4", "Typing does not beat a fixed head-retraining rule.", "H5 fails (S1, S2 ns; S3 worse)", "Confirmatory"],
     ["5", "On recurring drift TLA is worse than doing nothing: no revert and late responses; a memory does not fix it.", "H1 S3 −0.050; trace; memory on 10 seeds", "Confirmatory + descriptive"],
     ["6", "A wrong type is costly: aligning a real drift loses 0.18–0.21.", "Fixed-align on real variants", "Descriptive"],
     ["7", "Detection is the bottleneck: oracle trigger +12–16 points; real detector recall 0.42, median delay 23 rounds.", "Phase 3, 4, 6", "Descriptive"],
     ["8", "Label-free detection does not work for real drift; only the type classifier is label-free.", "Phase 3 detectors", "Validation"],
     ["9", "Client-level DP on uploads collapses accuracy at 8 clients; local adaptation spends no privacy budget.", "640 DP runs", "Descriptive"],
     ["10", "Device memory is dominated by the replay pool (~9 MiB peak); a model cache is cheap.", "Memory budget", "Computed"]], [0.3, 4.4, 2.0, 1.1])
BOX("<b>In one sentence:</b> where adaptation happens (locally, or with a small federated adapter) matters more than how cleverly the drift is typed; "
    "the real limit is that a device sees too few labels, too late, to know when a drift starts and when it ends.", colors.HexColor("#FDF1E8"))

# =====================================================================================================  13 coverage
H1("13. What was and was not evaluated")
TAB([["Question", "Answer"],
     ["Multiple drift types?", "Yes: virtual (offset, gain, rotation) and real (label swaps), sudden / gradual / recurring / one-sensor, plus a no-drift control (S0)."],
     ["Multiple detectors?", "Yes: Page–Hinkley (full and 20% labels, class-conditional), ADWIN, entropy, class-matched MMD, two-level with and without sentinel."],
     ["Multiple responses?", "Yes: no adaptation (global, personal), periodic / ADWIN / PH full federated retraining, FedProx-style, federated adapter, local full retraining, input alignment, head retraining, typed (TLA), memory, DP variants, oracle references."],
     ["Every detector × response × drift combination?", "No. The confirmatory matrix is 14 systems × 10 variants × 10 seeds; detectors were compared separately."],
     ["No-drift control?", "Yes, added after Phase 6 (Section 11.1), descriptive."],
     ["S4 statistics?", "Descriptive only (S4 was not in the registered test family): TLA 0.810 vs Personalized 0.681; alignment best (0.820)."],
     ["Official FedDrift / Flash code?", "No. E5 / E7a re-implement their response step on our backbone."],
     ["Real devices, natural drift, delayed labels, other datasets?", "No: one dataset (PAMAP2), injected drift, immediate labels, simulation only."],
     ["Clustering, secure aggregation?", "No. Only data locality is claimed; the DP study covers 8 clients."]], [2.2, 5.6])

# =====================================================================================================  14 corrections
H1("14. Corrections to the earlier 8-page summary")
TAB([["Earlier text", "Correct statement", "Source"],
     ["Model has approximately 77,000 parameters", "53,260 parameters (2-layer LSTM, 64 hidden, 12-class head), 213,040 B FP32", "decisions.md §3"],
     ["Baseline table (0.883 / 0.871 / 0.851) presented as the protocol", "Those are v1 (untrimmed) numbers; the primary v2 protocol gives 0.939 / 0.930 / 0.893 (E3b 0.935)", "results/phase1d/comparison.md"],
     ["No S0 run exists", "S0 was run on 10 evaluation seeds (Section 11.1)", "results/phase7/s0_mem.md"],
     ["S4: three variants, no statistics", "S4 has 2 variants in the confirmatory run (ankle gain, chest rotation); results are descriptive (Table 8.1)", "results/phase6/confirmatory.md"],
     ["H4 needs an audit", "Audited: margin 0.010, CIs in Table 8.3, all non-inferior", "results/phase6/confirmatory.md"],
     ["Oracle rows are upper bounds", "They are references with true alarms/types; oracle 2-way can fall below TLA with the true type", "Table 8.1"],
     ["(missing)", "Phase 7 memory, memory budget, DP study, S3 trace and supervisor runs added (Sections 9–11)", "results/phase7/"]], [2.4, 3.6, 1.8])
P("<b>Reproduce.</b> Code and raw results: repository branch <i>phase-7-memory-dp</i>. Tables: experiments/phase6_stats.py, "
  "phase7_memory_budget.py, phase7_dp_tradeoff.py, phase7_s0_mem.py; figures: experiments/paper_figs/. This PDF: experiments/report/build_results_pdf.py.")

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                        title="Concept drift experiments and results (full record)", author="Project team")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
