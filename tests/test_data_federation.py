"""Leakage and correctness tests for the Phase 1 data pipeline (synthetic .dat files)."""
import numpy as np
import pytest

from src.data import ACC16_COLS, DataConfig, build_federation
from src.data.federation import split_bound
from src.data.pamap2 import load_bouts
from tests.conftest import PAMAP2_SUBJECT_101, requires_pamap2

COLS_N = 54


def _write_dat(path, blocks, nan_at=None, seed=0):
    """blocks: list of (activity_id, n_rows). Timestamps are 100 Hz and contiguous."""
    rng = np.random.default_rng(seed)
    rows = []
    for act, n in blocks:
        r = rng.standard_normal((n, COLS_N)) + act      # activity-dependent mean
        r[:, 1] = act
        rows.append(r)
    arr = np.concatenate(rows)
    arr[:, 0] = np.arange(len(arr)) * 0.01
    if nan_at is not None:
        arr[nan_at, 4:7] = np.nan
    np.savetxt(path, arr, fmt="%.5f")
    return arr


@pytest.fixture
def synth_root(tmp_path):
    d = tmp_path / "Protocol"
    d.mkdir()
    _write_dat(d / "subject101.dat", [(0, 300), (1, 4000), (0, 500), (2, 4000), (4, 3000)], nan_at=[10, 11, 12])
    _write_dat(d / "subject102.dat", [(2, 5000), (0, 200), (4, 4000)], seed=1)
    return tmp_path


MODES = ["blocked", "temporal"]


def _cfg(mode="blocked", **kw):
    return DataConfig(root="Protocol", subjects=(101, 102), activities=(1, 2, 4),
                      split_mode=mode, block_windows=4, **kw)


def test_transient_rows_never_enter_windows_and_do_not_glue_bouts(synth_root):
    path = synth_root / "Protocol" / "subject101.dat"
    bouts = load_bouts(path, 101, ACC16_COLS, (1, 2, 4))
    assert [b.activity for b in bouts] == [1, 2, 4]
    assert [len(b.data) for b in bouts] == [4000, 4000, 3000]


def test_nan_interpolated_inside_bout(synth_root):
    bouts = load_bouts(synth_root / "Protocol" / "subject101.dat", 101, ACC16_COLS, (1, 2, 4))
    assert not any(np.isnan(b.data).any() for b in bouts)


def test_long_nan_run_drops_bout(tmp_path):
    p = tmp_path / "s.dat"
    _write_dat(p, [(1, 3000)], nan_at=np.arange(100, 400))
    assert load_bouts(p, 101, ACC16_COLS, (1,)) == []


@pytest.mark.parametrize("mode", MODES)
def test_windows_are_single_label(synth_root, mode):
    clients, lm, _ = build_federation(_cfg(mode), synth_root)
    c = clients[101]
    for s in ("train", "val", "test"):
        assert set(c.prov[s][:, 2].tolist()) <= {1, 2, 4}
        # label of a window equals its bout's activity by construction
        assert (np.array([lm[a] for a in c.prov[s][:, 2]]) == c.y[s]).all()


@pytest.mark.parametrize("mode", MODES)
def test_no_sample_overlap_between_splits(synth_root, mode):
    cfg = _cfg(mode)
    c = build_federation(cfg, synth_root)[0][101]
    spans = {s: [(int(b), int(st), int(st) + cfg.window) for b, st, _ in c.prov[s]] for s in ("train", "val", "test")}
    for a, b in [("train", "val"), ("val", "test"), ("train", "test")]:
        for bid, s0, e0 in spans[a]:
            for bid2, s1, e1 in spans[b]:
                if bid == bid2:
                    assert e0 <= s1 or e1 <= s0, f"{a}/{b} overlap in bout {bid}"


@pytest.mark.parametrize("mode", MODES)
def test_every_activity_is_in_every_split(synth_root, mode):
    c = build_federation(_cfg(mode), synth_root)[0][101]
    sets = [set(c.y[s].tolist()) for s in ("train", "val", "test")]
    assert sets[0] == sets[1] == sets[2]


def test_temporal_mode_puts_test_after_train_inside_each_bout(synth_root):
    c = build_federation(_cfg("temporal"), synth_root)[0][101]
    for bid in set(c.prov["train"][:, 0].tolist()):
        tr = c.prov["train"][c.prov["train"][:, 0] == bid][:, 1].max()
        te = c.prov["test"][c.prov["test"][:, 0] == bid][:, 1].min()
        assert tr < te


def test_blocked_mode_interleaves_splits_in_time(synth_root):
    """Stationary control: some test block must precede the last train block in at least one bout."""
    c = build_federation(_cfg("blocked"), synth_root)[0][101]
    interleaved = False
    for bid in set(c.prov["train"][:, 0].tolist()):
        tr_max = c.prov["train"][c.prov["train"][:, 0] == bid][:, 1].max()
        te = c.prov["test"][c.prov["test"][:, 0] == bid][:, 1]
        interleaved |= bool((te < tr_max).any())
    assert interleaved


def test_blocked_assignment_is_deterministic_in_split_seed(synth_root):
    a = build_federation(_cfg(split_seed=3), synth_root)[0][101]
    b = build_federation(_cfg(split_seed=3), synth_root)[0][101]
    c = build_federation(_cfg(split_seed=4), synth_root)[0][101]
    np.testing.assert_array_equal(a.prov["test"], b.prov["test"])
    assert not (len(a.prov["test"]) == len(c.prov["test"]) and (a.prov["test"] == c.prov["test"]).all())


@pytest.mark.parametrize("mode", MODES)
def test_normalisation_fit_on_train_only(synth_root, mode):
    c = build_federation(_cfg(mode), synth_root)[0][101]
    # train data is ~zero-mean / unit-std per channel, val/test are generally not exactly so
    tr = c.x["train"].reshape(-1, c.x["train"].shape[-1])
    assert np.abs(tr.mean(0)).max() < 0.15
    assert np.allclose(tr.std(0), 1.0, atol=0.15)
    te = c.x["test"].reshape(-1, c.x["test"].shape[-1])
    assert not np.allclose(te.mean(0), 0.0, atol=1e-3)


def test_normalisation_stats_are_independent_of_test_data(synth_root):
    a = build_federation(_cfg("temporal"), synth_root)[0][101]
    # corrupt the tail of the file (which becomes test data) and rebuild: train stats must not move
    p = synth_root / "Protocol" / "subject101.dat"
    arr = np.loadtxt(p)
    arr[-500:, ACC16_COLS] += 1000.0
    np.savetxt(p, arr, fmt="%.5f")
    b = build_federation(_cfg("temporal"), synth_root)[0][101]
    np.testing.assert_allclose(a.mean, b.mean)
    np.testing.assert_allclose(a.std, b.std)


def test_purge_leaves_a_gap_between_splits():
    cfg = DataConfig()
    tr, va, te = split_bound(100, cfg)
    assert va[0] - tr[1] >= 1 and te[0] - va[1] >= 1


@requires_pamap2
def test_acc16_channels_are_really_accelerometers():
    """Audit A14 regression: temperature sits at 3/20/37 (~30-35 C), +-16g acc has |mean|<=~10 and std>1."""
    import pandas as pd
    df = pd.read_csv(PAMAP2_SUBJECT_101, sep=r"\s+", header=None, na_values=["NaN"], nrows=50000)
    for c in (3, 20, 37):
        assert 25 < df[c].mean() < 40 and df[c].std() < 3          # temperature
    for c in ACC16_COLS:
        assert df[c].std() > 1.0 and abs(df[c].mean()) < 12         # gravity-scale acceleration


def test_trim_removes_edges_only_and_keeps_default_unchanged():
    base = DataConfig(subjects=(101,), activities=(1, 2, 4))
    trimmed = DataConfig(subjects=(101,), activities=(1, 2, 4), trim_samples=500)
    a = build_federation(base)[0][101]
    b = build_federation(trimmed)[0][101]
    assert DataConfig().trim_samples == 0
    assert sum(b.n(s) for s in ("train", "val", "test")) < sum(a.n(s) for s in ("train", "val", "test"))
    assert min(int(b.prov[s][:, 1].min()) for s in ("train", "val", "test")) >= 500   # provenance stays in raw bout coordinates
