"""Drift schedules: scenario YAML -> per-client DriftEvents + ground-truth manifest.

Time is measured in federated ROUNDS. A schedule is a pure function of (scenario file, variant, severity, seed,
client ids, client class sets): same inputs -> identical manifest.

Event model
-----------
profile   how the drift strength alpha(r) in [0, 1] evolves: sudden (step), gradual (linear ramp),
          recurring (on for `duration` rounds every `period` rounds, i.e. A -> B -> A -> B ...).
mode      virtual: the input distribution P(x) changes, labels keep their meaning.
          real:    P(y|x) changes (activity-to-label regime change); inputs are untouched.
transform virtual: offset (additive bias, in g), gain (multiplicative, sensor sensitivity),
          rotation (degrees about the sensor x axis, i.e. a re-mounted sensor);
          real: label_remap (disjoint class swaps).
type4     ground-truth 4-way drift type used to train/score the drift-type classifier:
          feature_level if a virtual transform touches only a subset of channels, else the profile name.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

N_CHANNELS = 9                      # acc16: hand xyz, chest xyz, ankle xyz
PROFILES = ("sudden", "gradual", "recurring")
MODES = ("virtual", "real")
TRANSFORMS = {"virtual": ("offset", "gain", "rotation"), "real": ("label_remap",)}
SEVERITIES = ("low", "med", "high")


@dataclass(frozen=True)
class DriftEvent:
    client: int
    event_id: int
    profile: str
    mode: str
    transform: str
    channels: Tuple[int, ...]            # empty for real drift
    magnitude: float                     # g | gain factor | degrees | number of swapped classes
    start: int                           # first round with alpha > 0
    ramp: int = 0                        # gradual: rounds to reach alpha = 1
    period: int = 0                      # recurring
    duration: int = 0                    # recurring: active rounds per period
    remap: Tuple[Tuple[int, int], ...] = ()   # real: swapped global class indices

    @property
    def type4(self) -> str:
        if self.mode == "virtual" and 0 < len(self.channels) < N_CHANNELS:
            return "feature_level"
        return self.profile

    def alpha(self, n_rounds: int) -> np.ndarray:
        """Drift strength per round, shape (n_rounds,)."""
        r = np.arange(n_rounds)
        a = np.zeros(n_rounds)
        if self.profile == "sudden":
            a[r >= self.start] = 1.0
        elif self.profile == "gradual":
            a = np.clip((r - self.start + 1) / max(1, self.ramp), 0.0, 1.0)
            a[r < self.start] = 0.0
        else:
            on = (r >= self.start) & (((r - self.start) % self.period) < self.duration)
            a[on] = 1.0
        return a

    def change_points(self, n_rounds: int) -> List[Dict]:
        """Ground truth: onset = alpha leaves 0, offset = alpha returns to 0, full = gradual alpha first reaches 1."""
        a = self.alpha(n_rounds)
        cps, prev = [], 0.0
        for r in range(n_rounds):
            if a[r] > 0 and prev == 0:
                cps.append({"round": r, "kind": "onset"})
            if a[r] == 0 and prev > 0:
                cps.append({"round": r, "kind": "offset"})
            if a[r] == 1 and prev < 1 and self.profile == "gradual":
                cps.append({"round": r, "kind": "full"})
            prev = a[r]
        return cps


def _resolve_channels(spec, mode: str) -> Tuple[int, ...]:
    if mode == "real":
        return ()
    if spec in (None, "all"):
        return tuple(range(N_CHANNELS))
    ch = tuple(int(c) for c in spec)
    if not ch or min(ch) < 0 or max(ch) >= N_CHANNELS:
        raise ValueError(f"bad channel list {spec}")
    return ch


def load_yaml(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_schedule(scenario: dict, severity_table: dict, variant: Optional[str], severity: str, seed: int,
                   clients: Sequence[int], client_classes: Dict[int, Sequence[int]],
                   timing: Optional[str] = None) -> Dict:
    """Returns the manifest dict (JSON-serialisable). `variant=None` is only valid for scenarios without variants (S0).
    `timing` (fixed | random | async) overrides the variant/scenario default."""
    n_rounds = int(scenario["n_rounds"])
    manifest = {"scenario": scenario["scenario"], "variant": variant, "severity": severity, "seed": int(seed), "timing": timing,
                "n_rounds": n_rounds, "clients": list(map(int, clients)), "events": [], "drifted_clients": []}
    if not scenario.get("variants"):
        return manifest
    if severity not in SEVERITIES:
        raise ValueError(severity)
    v = scenario["variants"][variant]
    profile, mode, transform = v["profile"], v["mode"], v["transform"]
    if profile not in PROFILES or mode not in MODES or transform not in TRANSFORMS[mode]:
        raise ValueError(f"invalid variant {variant}: {profile}/{mode}/{transform}")
    channels = _resolve_channels(v.get("channels"), mode)
    if transform == "rotation" and len(channels) % 3:
        raise ValueError("rotation needs whole 3-axis sensors")
    mag = float(severity_table[transform][severity])

    rng = np.random.default_rng((seed, 11))
    cov = v["coverage"]
    n_aff = len(clients) if cov == "all" else int(cov)
    affected = sorted(int(c) for c in rng.choice(list(clients), size=n_aff, replace=False))
    base = int(v.get("start", scenario["start"]))
    timing = timing or v.get("timing", scenario.get("timing", "fixed"))
    jitter = int(v.get("jitter", scenario.get("jitter", 0)))
    manifest["timing"], manifest["jitter"] = timing, jitter
    for c in affected:
        crng = np.random.default_rng((seed, 23, int(c)))
        start = base
        if timing == "random":
            start += int(crng.integers(-jitter, jitter + 1))
        elif timing == "async":               # staggered onsets, never earlier than the base round
            start += int(crng.integers(0, jitter + 1))
        remap = ()
        if mode == "real":
            cls = sorted(int(x) for x in client_classes[c])
            k = min(int(mag), len(cls) - len(cls) % 2)
            pick = [int(x) for x in crng.permutation(cls)[:k]]
            remap = tuple((pick[i], pick[i + 1]) for i in range(0, len(pick) - 1, 2))
        ev = DriftEvent(client=int(c), event_id=0, profile=profile, mode=mode, transform=transform, channels=channels,
                        magnitude=mag, start=start, ramp=int(v.get("ramp", 0)), period=int(v.get("period", 0)),
                        duration=int(v.get("duration", 0)), remap=remap)
        if not 0 <= ev.start < n_rounds:
            raise ValueError(f"event start {ev.start} outside the stream")
        manifest["events"].append({**asdict(ev), "type4": ev.type4, "change_points": ev.change_points(n_rounds)})
    manifest["drifted_clients"] = affected
    return manifest


def events_from_manifest(manifest: Dict) -> Dict[int, List[DriftEvent]]:
    out: Dict[int, List[DriftEvent]] = {c: [] for c in manifest["clients"]}
    for e in manifest["events"]:
        d = {k: e[k] for k in DriftEvent.__dataclass_fields__}
        d["channels"] = tuple(d["channels"])
        d["remap"] = tuple(tuple(p) for p in d["remap"])
        out[int(e["client"])].append(DriftEvent(**d))
    return out
