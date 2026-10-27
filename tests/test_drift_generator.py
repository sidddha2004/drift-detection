"""Manifest-driven tests of the Phase 2 drift generator (synthetic clients, no dataset needed)."""
from pathlib import Path

import numpy as np
import pytest

from src.data import ClientData
from src.drift import build_schedule, build_streams, events_from_manifest, load_yaml
from src.drift.schedule import DriftEvent

CFG = Path(__file__).resolve().parents[1] / "configs/scenarios"
SEV = load_yaml(CFG / "severity.yaml")
CLIENTS = list(range(101, 109))
CLASSES = {c: list(range(8)) for c in CLIENTS}


def _scn(name):
    return load_yaml(CFG / name)


def _manifest(name, variant, severity="med", seed=1, timing=None):
    return build_schedule(_scn(name), SEV, variant, severity, seed, CLIENTS, CLASSES, timing)


def _client(cid=101, n=(40, 30, 30), W=16, seed=0):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 2, 9).astype(np.float32)
    std = rng.uniform(1, 4, 9).astype(np.float32)
    x = {s: rng.normal(0, 1, (k, W, 9)).astype(np.float32) for s, k in zip(("train", "val", "test"), n)}
    y = {s: rng.integers(0, 8, k) for s, k in zip(("train", "val", "test"), n)}
    return ClientData(client_id=cid, x=x, y=y, mean=mean, std=std)


def _event(**kw):
    d = dict(client=101, event_id=0, profile="sudden", mode="virtual", transform="offset", channels=tuple(range(9)),
             magnitude=1.0, start=10)
    d.update(kw)
    return DriftEvent(**d)


# ---- alpha profiles ---------------------------------------------------------------------------------------------
def test_sudden_alpha_is_a_step():
    a = _event(start=10).alpha(30)
    assert (a[:10] == 0).all() and (a[10:] == 1).all()


def test_gradual_alpha_ramps_monotonically_to_one():
    a = _event(profile="gradual", start=10, ramp=5).alpha(30)
    assert (a[:10] == 0).all() and a[10] == pytest.approx(0.2) and a[14] == 1 and (a[14:] == 1).all()
    assert (np.diff(a) >= 0).all()


def test_recurring_alpha_returns_to_zero():
    a = _event(profile="recurring", start=10, period=20, duration=10).alpha(60)
    assert (a[10:20] == 1).all() and (a[20:30] == 0).all() and (a[30:40] == 1).all() and (a[:10] == 0).all()


def test_explicit_zero_start_is_honoured_A11():
    """Audit A11: the old injector treated an explicit start of 0 as unset."""
    assert (_event(start=0).alpha(10) == 1).all()


def test_change_points():
    cps = _event(profile="recurring", start=10, period=20, duration=10).change_points(60)
    assert [(c["round"], c["kind"]) for c in cps] == [(10, "onset"), (20, "offset"), (30, "onset"), (40, "offset"), (50, "onset")]
    cps = _event(profile="gradual", start=10, ramp=5).change_points(30)
    assert [(c["round"], c["kind"]) for c in cps] == [(10, "onset"), (14, "full")]


# ---- virtual transforms -------------------------------------------------------------------------------------------
def test_pre_drift_stream_is_untouched():
    c = _client()
    m = _manifest("s1_sudden.yaml", "virtual_offset_full")
    s = build_streams({101: c}, m)[101]
    start = m["events"][0]["start"]
    x, y = s.eval_batch(start - 1)
    np.testing.assert_array_equal(x, c.x["test"])
    np.testing.assert_array_equal(y, c.y["test"])


def test_offset_adds_exactly_the_magnitude_in_raw_units():
    c = _client()
    s = build_streams({101: c}, _manifest("s1_sudden.yaml", "virtual_offset_full", "high"))[101]
    x, _ = s.eval_batch(59)
    raw0 = c.x["test"] * c.std + c.mean
    raw1 = x * c.std + c.mean
    np.testing.assert_allclose(raw1 - raw0, 2.0, atol=1e-4)


def test_feature_level_touches_only_target_channels():
    c = _client()
    m = _manifest("s4_feature.yaml", "hand_offset_partial")
    cid = m["drifted_clients"][0]
    c = _client(cid)
    s = build_streams({cid: c}, m)[cid]
    x, _ = s.eval_batch(59)
    changed = np.where(~np.isclose(x, c.x["test"]).all(axis=(0, 1)))[0]
    assert list(changed) == [0, 1, 2]
    assert m["events"][0]["type4"] == "feature_level"


def test_rotation_preserves_acceleration_magnitude_and_other_sensors():
    c = _client()
    ev = _event(transform="rotation", channels=(3, 4, 5), magnitude=45.0)
    from src.drift.transforms import apply_virtual
    x = apply_virtual(c.x["test"], c.mean, c.std, ev, 1.0)
    r0, r1 = c.x["test"] * c.std + c.mean, x * c.std + c.mean
    np.testing.assert_allclose(np.linalg.norm(r0[..., 3:6], axis=-1), np.linalg.norm(r1[..., 3:6], axis=-1), rtol=1e-4)
    np.testing.assert_allclose(x[..., [0, 1, 2, 6, 7, 8]], c.x["test"][..., [0, 1, 2, 6, 7, 8]], atol=1e-6)
    np.testing.assert_allclose(x[..., 3], c.x["test"][..., 3], atol=1e-6)    # rotation axis is x


def test_gradual_virtual_strength_scales_with_alpha():
    c = _client()
    from src.drift.transforms import apply_virtual
    ev = _event(profile="gradual", magnitude=2.0, ramp=10)
    half = (apply_virtual(c.x["test"], c.mean, c.std, ev, 0.5) - c.x["test"]) * c.std
    np.testing.assert_allclose(half, 1.0, atol=1e-4)


# ---- real drift ---------------------------------------------------------------------------------------------------
def test_real_drift_relabels_but_never_touches_inputs():
    m = _manifest("s1_sudden.yaml", "real_full", "med")
    e = m["events"][0]
    cid = e["client"]
    c = _client(cid)
    s = build_streams({cid: c}, m)[cid]
    x, y = s.eval_batch(59)
    np.testing.assert_array_equal(x, c.x["test"])
    swap = {a: b for a, b in e["remap"]} | {b: a for a, b in e["remap"]}
    expect = np.array([swap.get(int(v), int(v)) for v in c.y["test"]])
    np.testing.assert_array_equal(y, expect)
    assert len(e["remap"]) == 2                                   # med = 4 classes = 2 disjoint pairs
    flat = [k for p in e["remap"] for k in p]
    assert len(set(flat)) == len(flat) and set(flat) <= set(CLASSES[cid])


def test_gradual_real_drift_remaps_a_growing_fraction():
    m = _manifest("s2_gradual.yaml", "real_partial", "high", timing="fixed")
    e = m["events"][0]
    cid = e["client"]
    c = _client(cid, n=(40, 30, 3000))
    s = build_streams({cid: c}, m)[cid]
    frac = []
    for r in (e["start"], e["start"] + 4, e["start"] + 9):
        _, y = s.eval_batch(r)
        remapped = np.isin(c.y["test"], [k for p in e["remap"] for k in p])
        frac.append(float((y[remapped] != c.y["test"][remapped]).mean()))
    assert frac[0] < frac[1] < frac[2] and frac[2] == pytest.approx(1.0)


# ---- manifest -----------------------------------------------------------------------------------------------------
def test_manifest_is_deterministic_and_seed_dependent():
    assert _manifest("s1_sudden.yaml", "virtual_offset_partial", seed=5) == _manifest("s1_sudden.yaml", "virtual_offset_partial", seed=5)
    drifted = {tuple(_manifest("s1_sudden.yaml", "virtual_offset_partial", seed=s)["drifted_clients"]) for s in range(8)}
    assert len(drifted) > 1


@pytest.mark.parametrize("name,variant,n", [("s1_sudden.yaml", "virtual_offset_partial", 3), ("s1_sudden.yaml", "virtual_offset_full", 8),
                                            ("s2_gradual.yaml", "real_partial", 3), ("s3_recurring.yaml", "real_partial", 3)])
def test_coverage(name, variant, n):
    m = _manifest(name, variant)
    assert len(m["drifted_clients"]) == n == len(m["events"])


def test_s0_has_no_events():
    m = build_schedule(_scn("s0_none.yaml"), SEV, None, "med", 1, CLIENTS, CLASSES)
    assert m["events"] == [] and m["drifted_clients"] == []


def test_timing_modes():
    fixed = _manifest("s1_sudden.yaml", "virtual_offset_full", timing="fixed")
    rnd = _manifest("s1_sudden.yaml", "virtual_offset_full", timing="random")
    asy = _manifest("s2_gradual.yaml", "virtual_offset_full")
    assert {e["start"] for e in fixed["events"]} == {20}
    assert all(15 <= e["start"] <= 25 for e in rnd["events"]) and len({e["start"] for e in rnd["events"]}) > 1
    assert all(20 <= e["start"] <= 30 for e in asy["events"]) and len({e["start"] for e in asy["events"]}) > 1


def test_type4_labels_cover_all_four_types():
    got = set()
    for name, variant in [("s1_sudden.yaml", "virtual_offset_full"), ("s2_gradual.yaml", "virtual_offset_full"),
                          ("s3_recurring.yaml", "virtual_offset_partial"), ("s4_feature.yaml", "hand_offset_partial")]:
        got |= {e["type4"] for e in _manifest(name, variant)["events"]}
    assert got == {"sudden", "gradual", "recurring", "feature_level"}


def test_manifest_roundtrip_through_json():
    import json
    m = _manifest("s3_recurring.yaml", "real_partial")
    ev = events_from_manifest(json.loads(json.dumps(m)))
    assert sum(len(v) for v in ev.values()) == len(m["events"])


def test_severity_levels_are_ordered():
    for t in ("offset", "gain", "rotation", "label_remap"):
        v = SEV[t]
        assert v["low"] < v["med"] < v["high"]


# ---- stream -------------------------------------------------------------------------------------------------------
def test_stream_does_not_mutate_client_data_and_is_reproducible():
    c = _client()
    before = {k: v.copy() for k, v in c.x.items()}
    m = _manifest("s1_sudden.yaml", "virtual_offset_full")
    a = build_streams({101: c}, m)[101].arrival_batch(40)
    b = build_streams({101: c}, m)[101].arrival_batch(40)
    np.testing.assert_array_equal(a[0], b[0])
    for k in before:
        np.testing.assert_array_equal(c.x[k], before[k])


def test_arrival_stream_covers_every_val_window_once_per_pass():
    c = _client(n=(40, 30, 30))
    s = build_streams({101: c}, _manifest("s1_sudden.yaml", "virtual_offset_full"))[101]
    seen = np.concatenate([s.order[(np.arange(s.per_round) + r * s.per_round) % 30] for r in range(60)])
    assert set(seen.tolist()) == set(range(30))
