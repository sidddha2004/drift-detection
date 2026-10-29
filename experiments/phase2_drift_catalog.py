"""Phase 2: build the manifest catalogue and stream plots for every scenario/variant (reference seed 42, med severity).

Outputs (results/phase2/): manifest_catalog.json, manifest_catalog.md, stream_<scenario>.png
Visual check = alpha(round, client) heatmap + the arrival stream of one affected client (mean of the first targeted
raw channel, or fraction of relabelled windows for real drift) so every schedule can be inspected by eye.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, build_federation  # noqa: E402
from src.drift import build_schedule, build_streams, load_yaml  # noqa: E402
from src.utils.io import load_config, run_metadata  # noqa: E402

SCENARIOS = ["s0_none", "s1_sudden", "s2_gradual", "s3_recurring", "s4_feature"]
SEED, SEVERITY = 42, "med"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(ROOT / "configs/phase1_v2_trimmed.yaml")
    clients, label_map, _ = build_federation(DataConfig(**cfg["data"]))
    cids = sorted(clients)
    classes = {c: sorted(set(clients[c].y["train"].tolist())) for c in cids}
    sev = load_yaml(ROOT / "configs/scenarios/severity.yaml")
    out = ROOT / "results/phase2"
    out.mkdir(parents=True, exist_ok=True)
    catalog, rows = [], ["# Phase 2 drift scenario catalogue", "",
                         f"Reference seed {SEED}, severity `{SEVERITY}`, protocol v2 clients {cids}.", "",
                         "| scenario | variant | type4 | clients (start round) | change points (first client) |", "|---|---|---|---|---|"]
    for name in SCENARIOS:
        scn = load_yaml(ROOT / f"configs/scenarios/{name}.yaml")
        variants = list(scn.get("variants", {None: None}))
        fig, ax = plt.subplots(2, len(variants), figsize=(3.6 * len(variants), 5.2), squeeze=False)
        for j, v in enumerate(variants):
            m = build_schedule(scn, sev, v, SEVERITY, SEED, cids, classes)
            catalog.append(m)
            n = m["n_rounds"]
            alpha = np.zeros((len(cids), n))
            for e in m["events"]:
                from src.drift.schedule import DriftEvent
                ev = DriftEvent(**{k: (tuple(e[k]) if k == "channels" else tuple(map(tuple, e[k])) if k == "remap" else e[k])
                                   for k in DriftEvent.__dataclass_fields__})
                alpha[cids.index(e["client"])] = ev.alpha(n)
            ax[0, j].imshow(alpha, aspect="auto", cmap="Reds", vmin=0, vmax=1)
            ax[0, j].set_yticks(range(len(cids)))
            ax[0, j].set_yticklabels(cids, fontsize=6)
            ax[0, j].set_title(f"{scn['scenario']} {v or 'none'}", fontsize=8)
            ax[0, j].set_xlabel("round")
            if m["events"]:
                e = m["events"][0]
                streams = build_streams(clients, m)
                cid = e["client"]
                s = streams[cid]
                if e["mode"] == "virtual":
                    ch = e["channels"][0]
                    tr = [float((s.arrival_batch(r)[0][..., ch] * s.c.std[ch] + s.c.mean[ch]).mean()) for r in range(n)]
                    ax[1, j].set_ylabel(f"mean raw ch{ch} (g)", fontsize=7)
                else:
                    tr = [float((s.arrival_batch(r)[1] != s.c.y["val"][s.order[(np.arange(s.per_round) + r * s.per_round) % len(s.order)]]).mean())
                          for r in range(n)]
                    ax[1, j].set_ylabel("frac. relabelled", fontsize=7)
                ax[1, j].plot(tr, lw=1)
                for cp in e["change_points"]:
                    ax[1, j].axvline(cp["round"], ls=":", c="r" if cp["kind"] == "onset" else "g")
                ax[1, j].set_title(f"client {cid} arrival stream", fontsize=8)
                ax[1, j].set_xlabel("round")
                rows.append(f"| {scn['scenario']} | {v} | {e['type4']} | " + ", ".join(f"{x['client']}({x['start']})" for x in m["events"])
                            + " | " + ", ".join(f"{c['kind']}@{c['round']}" for c in e["change_points"]) + " |")
            else:
                ax[1, j].axis("off")
                rows.append(f"| {scn['scenario']} | none | - | - | - |")
        fig.tight_layout()
        fig.savefig(out / f"stream_{name}.png", dpi=130)
        plt.close(fig)
    (out / "manifest_catalog.json").write_text(json.dumps({"meta": run_metadata(), "manifests": catalog}, indent=1), encoding="utf-8")
    (out / "manifest_catalog.md").write_text("\n".join(rows), encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
