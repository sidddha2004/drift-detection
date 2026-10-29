"""Build every paper figure + summary_stats.md from cached results. No training, no streaming runs (seconds).

    python experiments/paper_figs/run_all.py          # all figures -> docs/paper/figs/
    pytest experiments/paper_figs                      # tests for these scripts
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures as F  # noqa: E402


def main(out=F.OUT):
    runs = F.load_runs()
    conf = json.loads(F.CONF.read_text())
    det = json.loads(F.DET.read_text())
    mode = json.loads(F.MODE.read_text())
    ps = F.per_seed(runs)
    made = [F.fig_forest(conf, out), F.fig_paired_seeds(ps, out), F.fig_heatmap(runs, out), F.fig_s3_offphase(runs, out),
            F.fig_collateral(runs, out), F.fig_alarm_delay(runs, out), F.fig_detectors(det, out), F.fig_mode(mode, out),
            F.fig_typing_ladder(runs, out), F.fig_cost(runs, out)]
    (Path(out) / "summary_stats.md").write_text(F.summary_stats(runs, conf), encoding="utf-8")
    print(f"{len(made)} figures -> {out}: {', '.join(made)} + summary_stats.md")
    return made


if __name__ == "__main__":
    main()
