"""Tests for the cached-results figure scripts: synthetic runs for the aggregation logic, plus a smoke test on the real cache."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures as F  # noqa: E402


def _run(policy, seed, scen="S1", var="v", f1=0.7, alarms=(), onset=30):
    alpha = np.zeros((2, 120)); alpha[0, onset:] = 1.0
    return {"policy": policy, "seed": seed, "scenario": scen, "variant": var, "clients": [101, 102], "drifted": [101],
            "alpha": alpha.tolist(), "alarms": [list(a) for a in alarms],
            "metrics": {"post_change_f1": f1, "off_phase_f1": float("nan")}}


def test_per_seed_averages_variants_and_drops_nan():
    runs = [_run("p", 1, var="a", f1=0.6), _run("p", 1, var="b", f1=0.8), _run("p", 2, f1=0.5)]
    ps = F.per_seed(runs)
    assert ps[("p", "S1")] == {1: pytest.approx(0.7), 2: pytest.approx(0.5)}
    assert F.per_seed(runs, "off_phase_f1") == {}


def test_onset_and_alarm_delay():
    assert F.onset_round([0, 0, 0.5, 1]) == 2 and F.onset_round([0, 0]) is None
    runs = [_run(F.E8, 1, alarms=[(101, 20), (101, 38), (102, 31)]), _run(F.E8, 2), _run("other", 3, alarms=[(101, 31)])]
    assert F.alarm_delays(runs) == [8, None]          # pre-onset alarm and the clean client's alarm are ignored; missed = None


def test_summary_stats_lists_every_test(tmp_path):
    conf = {"family": [{"h": "H1", "scenario": "S1", "diff": 0.1, "ci": [0.0, 0.2], "p_holm": 0.01, "significant": True}]}
    s = F.summary_stats([_run(F.E8, 1, alarms=[(101, 35)])], conf)
    assert "H1 S1: +0.100" in s and "1/1 drifted clients detected" in s


@pytest.mark.skipif(not list(F.RUNS.glob("conf_*.json")), reason="cached confirmatory results not present")
def test_run_all_on_cached_results(tmp_path):
    import run_all
    made = run_all.main(out=tmp_path)
    for n in made:
        assert (tmp_path / f"{n}.pdf").stat().st_size > 0 and (tmp_path / f"{n}.png").stat().st_size > 0
    assert "Confirmatory family" in (tmp_path / "summary_stats.md").read_text(encoding="utf-8")


@pytest.mark.skipif(not F.CONF.exists(), reason="confirmatory.json not present")
def test_cached_stats_match_paper_table():
    conf = json.loads(F.CONF.read_text())
    h1s1 = next(t for t in conf["family"] if t["h"] == "H1" and t["scenario"] == "S1")
    assert round(h1s1["diff"], 3) == 0.079 and h1s1["significant"]   # Table I of main.tex
