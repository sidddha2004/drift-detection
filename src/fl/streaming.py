"""Streaming federated engine (WP4): a drifting stream replayed round by round on the GPU with real LSTM training.

Round r (60 per run):
  1. every client receives its arrival batch (Phase 2 stream; labels are available immediately) and the deployed model
     scores it; the error bits feed the client's detector,
  2. the policy decides which clients start an adaptation episode (trigger) and with which Response,
  3. clients in an episode train locally on their recent labelled arrival buffer; federated responses upload the update
     (full model or adapter only), the server averages the participants, and the participants download the average,
  4. every client is evaluated on its held-out TEST pool transformed with the round's drift strength (primary endpoint).
Bytes are counted from the serialised FP32 payload of what is actually sent.
"""
import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data import ClientData
from src.detect.detectors import PageHinkley
from src.detect.signals import window_stats
from src.drift import ClientStream
from src.drift.schedule import DriftEvent
from src.eval import classification_metrics
from src.fl.adapter import AdaptedLSTM
from src.fl.align import fit_input_alignment, reference_means
from src.fl.engine import DEVICE, TrainConfig, evaluate, fedavg, make_model, predict, state_bytes, to_tensors, train_epochs
from src.utils.seed import seed_everything


# ---------------------------------------------------------------------------------------------------------------------
# pre-deployment federation (E3 global model + E3b personalised models)
def pretrain_federation(clients: Dict[int, ClientData], cal: Dict[int, Tuple[np.ndarray, np.ndarray]], n_classes: int,
                        tc: TrainConfig, seed: int, rounds: int = 60, local_epochs: int = 2, ft_epochs: int = 20):
    """FedAvg on the train pools (E3), then local fine-tuning per client (E3b). Checkpoints are chosen on the CLEAN
    calibration pool only, so no arrival window influences model selection. Returns (global, {cid: personal})."""
    seed_everything(seed)
    T = {c: to_tensors(cl.x["train"], cl.y["train"]) for c, cl in clients.items()}
    V = {c: to_tensors(*cal[c]) for c in clients}
    gen = torch.Generator().manual_seed(seed)
    glob, local = make_model(n_classes, 9, tc), make_model(n_classes, 9, tc)
    best, best_state = -1.0, None
    for _ in range(rounds):
        states, sizes = [], []
        for c in clients:
            local.load_state_dict(glob.state_dict())
            train_epochs(local, *T[c], local_epochs, tc, gen)
            states.append(copy.deepcopy(local.state_dict()))
            sizes.append(len(T[c][1]))
        glob.load_state_dict(fedavg(states, sizes))
        f1 = float(np.mean([evaluate(glob, *V[c], n_classes)["macro_f1"] for c in clients]))
        if f1 > best:
            best, best_state = f1, copy.deepcopy(glob.state_dict())
    glob.load_state_dict(best_state)
    personal = {}
    for c in clients:
        g = torch.Generator().manual_seed(seed * 1000 + c)
        m = make_model(n_classes, 9, tc)
        m.load_state_dict(glob.state_dict())
        opt = torch.optim.Adam(m.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        b, bs = -1.0, None
        for _ in range(ft_epochs):
            train_epochs(m, *T[c], 1, tc, g, opt)
            f1 = evaluate(m, *V[c], n_classes)["macro_f1"]
            if f1 > b:
                b, bs = f1, copy.deepcopy(m.state_dict())
        m.load_state_dict(bs)
        personal[c] = m.eval()
    return glob.eval(), personal


# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Response:
    """How a client adapts once triggered."""
    name: str
    kind: str                       # "full" (all weights) | "adapter" (input+feature+head) | "input" | "head" (gradient training)
                                    # | "align" (closed-form input recalibration, no training, local)
                                    # | "revert" (back to the identity input map, the fitted map is cached)
                                    # | "restore" (reload the cached map of the previous episode, no data needed)
    federated: bool                 # upload / aggregate among the clients adapting this round, or purely local
    rounds: int = 3                 # adaptation episode length in rounds
    epochs: int = 15                # local epochs per round over the buffer
    lr: float = 1e-2                # adapter lr (full fine-tuning uses lr_full)
    lr_full: float = 1e-3
    buffer_rounds: int = 15         # arrival rounds kept in the adaptation buffer
    bs: int = 16
    anchor: float = 1e-2            # L2 pull of the input matrix towards the identity (adapters only)
    track: bool = False             # kind "align": sliding buffer of the last `buffer_rounds` rounds (follows a ramp)
    imus: Optional[Tuple[int, ...]] = None   # kind "align": keep the fit only for these sensors (feature-level drift)
    align_ridge: float = 0.05       # kind "align": pull of the fitted 3x3 sensor map towards the identity
    lookback: int = 8               # kind "align": the buffer starts this many rounds before the alarm
    replay_unseen: bool = False     # replay only classes ABSENT from the buffer (seen classes: the buffer holds the current truth)
    replay: float = 1.0             # clean train-pool windows replayed per buffer window (guards against collapsing onto the
                                    # few activities of a recent block; consistent with virtual drift, conflicts with real drift)


@dataclass
class Policy:
    name: str
    trigger: str                    # "none" | "oracle" | "detector" | "periodic"
    response: Optional[Response] = None
    respond_by_type: Optional[Callable[[Dict], Response]] = None     # typed policies: info -> Response
    oracle_delay: int = 2           # oracle trigger: rounds after the true onset
    oracle_offsets: bool = False    # oracle trigger also fires `oracle_delay` rounds after a drift ENDS (recurring drift)
    period: int = 10                # periodic trigger: all clients every `period` rounds
    detector: Tuple[float, float] = (0.05, 8.0)                      # PH (delta, lambda) on the error bit
    cooldown_windows: int = 30
    typer: Optional[Callable] = None   # deployable typing: typer(cid, record) -> info dict, called `typing_delay` rounds after an alarm
    typing_delay: int = 5
    start_model: str = "personal"   # "personal" (E3b) | "global" (E3)


def _train(model, x, y, epochs, lr, bs, gen, params=None, anchor=0.0):
    params = params if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            if anchor and hasattr(model, "in_w"):
                loss = loss + anchor * ((model.in_w - torch.eye(model.in_w.shape[0], device=DEVICE)) ** 2).sum()
            loss.backward()
            opt.step()
    model.eval()


@dataclass
class _Client:
    cid: int
    train_x: torch.Tensor
    train_y: torch.Tensor
    ref: list
    model: torch.nn.Module
    stream: ClientStream
    ph: PageHinkley
    buf_x: List = field(default_factory=list)
    buf_y: List = field(default_factory=list)
    buf_round: List = field(default_factory=list)
    last_alarm_window: int = -10 ** 9
    window: int = 0
    episode_left: int = 0
    alarm_round: int = 0
    pending: Optional[Tuple[int, int, int]] = None    # (decide_round, alarm_window, alarm_round) while waiting for typing evidence
    sig: Dict[str, List] = field(default_factory=lambda: {"feat": [], "ent": [], "pred": [], "err": []})
    library: List = field(default_factory=list)       # cached adapter states of finished adaptation episodes
    init_state: Optional[Dict] = None                  # adapter state at wrap time (identity input map, original head)
    resp: Optional[Response] = None
    adapted: bool = False


def _aggregate(models: List[torch.nn.Module], sizes: List[int], kind: str) -> Dict[str, torch.Tensor]:
    if kind == "full":
        return fedavg([m.state_dict() for m in models], sizes)
    return fedavg([m.adapter_state() for m in models], sizes)


def run_stream(clients: Dict[int, ClientData], start_models: Dict[int, torch.nn.Module], events: Dict[int, List[DriftEvent]],
               n_rounds: int, seed: int, n_classes: int, policy: Policy, tc: TrainConfig) -> Dict:
    """Run one policy on one drifting stream. `events` come from a Phase 2 manifest (ground truth for oracle triggers)."""
    seed_everything(seed)
    gen = torch.Generator().manual_seed(seed + 7)
    C: Dict[int, _Client] = {}
    for cid, cl in clients.items():
        m = copy.deepcopy(start_models[cid]).eval()
        m.lstm.flatten_parameters()
        tx, ty = to_tensors(cl.x["train"], cl.y["train"])
        C[cid] = _Client(cid, tx, ty, reference_means(cl.x["train"], cl.y["train"], cl.mean, cl.std), m, ClientStream(cl, events[cid], n_rounds, seed), PageHinkley(*policy.detector, warm=25))
    cids = sorted(C)
    f1 = np.zeros((len(cids), n_rounds))
    acc = np.zeros_like(f1)
    up = np.zeros(n_rounds)
    down = np.zeros(n_rounds)
    alarms: List[Tuple[int, int]] = []
    onsets = {c: [cp["round"] for e in events[c] for cp in e.change_points(n_rounds) if cp["kind"] == "onset"] for c in cids}
    infos = {c: [{"type4": e.type4, "mode": e.mode, "transform": e.transform, "channels": e.channels, "start": e.start} for e in events[c]] for c in cids}
    t0, adapt_rounds, labels_used = time.time(), 0, 0

    for r in range(n_rounds):
        triggered: Dict[int, Dict] = {}
        for c in cids:
            st = C[c]
            x, y = st.stream.arrival_batch(r)
            xt, yt = to_tensors(x, y)
            if policy.typer is not None:
                with torch.no_grad():
                    st.model.eval()
                    pr = torch.softmax(st.model(xt), 1)
                pred = pr.argmax(1).cpu().numpy()
                ent = (-(pr * torch.log(pr.clamp_min(1e-12))).sum(1)).cpu().numpy()
                st.sig["feat"].append(window_stats(x)); st.sig["ent"].append(ent); st.sig["pred"].append(pred.astype(np.int16))
                st.sig["err"].append((pred != y).astype(np.float32))
            else:
                pred = predict(st.model, xt)
            err = (pred != y).astype(np.float64)
            st.buf_x.append(xt); st.buf_y.append(yt); st.buf_round.append(r)
            labels_used += len(y)
            fired = False
            for e in err:
                st.window += 1
                if st.ph.update(float(e)) and st.window - st.last_alarm_window >= policy.cooldown_windows:
                    fired = True
                    st.last_alarm_window = st.window
                if st.ph.m - st.ph.mmin > st.ph.lam:      # reset after every crossing (also inside cooldown)
                    st.ph.reset()
            if policy.trigger == "detector" and fired and st.episode_left == 0 and st.pending is None:
                if policy.typer is None:
                    triggered[c] = {"source": "detector"}
                else:
                    st.pending = (r + policy.typing_delay, st.window, r)
            if st.pending is not None and r >= st.pending[0]:
                rec = {"feat": np.concatenate(st.sig["feat"]), "ent": np.concatenate(st.sig["ent"]), "pred": np.concatenate(st.sig["pred"]),
                       "err": np.concatenate(st.sig["err"]), "per_round": st.stream.per_round, "alarm_window": st.pending[1] - 1,
                       "alarm_round": st.pending[2], "round": r}
                info = policy.typer(c, rec)
                st.pending = None
                if info is not None and st.episode_left == 0:
                    info = dict(info); info.setdefault("source", "detector"); triggered[c] = info
            elif policy.trigger == "oracle":
                cps = [(cp["round"], cp["kind"], j) for e in events[c] for j, cp in enumerate([q for q in e.change_points(n_rounds) if q["kind"] == "onset"])]
                if policy.oracle_offsets:
                    cps += [(cp["round"], "offset", j) for e in events[c] for j, cp in enumerate([q for q in e.change_points(n_rounds) if q["kind"] == "offset"])]
                for rr, kind, j in cps:
                    if r == rr + policy.oracle_delay and (st.episode_left == 0 or kind == "offset"):
                        info = dict(infos[c][0]); info.update(source="oracle", event=kind, occurrence=j,
                                                              imus=tuple(sorted({ch // 3 for ch in infos[c][0]["channels"]})))
                        triggered[c] = info
            elif policy.trigger == "periodic" and r > 0 and r % policy.period == 0 and st.episode_left == 0:
                triggered[c] = {"source": "periodic"}
        for c, info in triggered.items():
            st = C[c]
            resp = policy.respond_by_type(info) if policy.respond_by_type else policy.response
            if resp is None:
                continue
            alarms.append((c, r))
            if resp.kind != "full" and not isinstance(st.model, AdaptedLSTM):
                st.model = AdaptedLSTM(st.model)
                st.init_state = st.model.adapter_state()
            if resp.kind == "revert":
                st.episode_left = 0
                if st.init_state is not None:
                    st.model.load_adapter_state(st.init_state)
                continue
            if resp.kind == "restore" and st.library:
                st.episode_left = 0
                st.model.load_adapter_state(st.library[-1])
                continue
            st.resp, st.episode_left, st.alarm_round = resp, resp.rounds, r
            if resp.kind == "restore":
                st.resp = resp = Response(**{**resp.__dict__, "kind": "align"})
            if isinstance(st.model, AdaptedLSTM):
                st.model.set_trainable({"adapter": ("input", "feature", "head"), "input": ("input",), "head": ("head",)}.get(resp.kind, ()))

        # adaptation round: local training of every client in an episode, then optional federation
        active = [c for c in cids if C[c].episode_left > 0]
        if active:
            adapt_rounds += 1
            for c in active:
                st, rs = C[c], C[c].resp
                lo = (r - rs.buffer_rounds + 1 if rs.track else st.alarm_round - rs.lookback) if rs.kind == "align" else r - rs.buffer_rounds + 1
                keep = [i for i, rr in enumerate(st.buf_round) if rr >= lo]
                x = torch.cat([st.buf_x[i] for i in keep]); y = torch.cat([st.buf_y[i] for i in keep])
                if rs.kind == "align":
                    cl = clients[c]
                    W, b = fit_input_alignment(x.cpu().numpy(), y.cpu().numpy(), st.ref, cl.mean, cl.std, ridge=rs.align_ridge)
                    if rs.imus is not None:                         # keep the fit only for the named sensors
                        keep_m = np.zeros(9, dtype=bool)
                        for m_ in rs.imus:
                            keep_m[3 * m_:3 * m_ + 3] = True
                        W = np.where(keep_m[:, None] & keep_m[None, :], W, np.eye(9)); b = np.where(keep_m, b, 0.0)
                    with torch.no_grad():
                        st.model.in_w.copy_(torch.from_numpy(W).float()); st.model.in_b.copy_(torch.from_numpy(b).float())
                    st.adapted = True
                    continue
                if rs.replay > 0:
                    k = int(round(rs.replay * len(y)))
                    pool = torch.arange(len(st.train_y))
                    if rs.replay_unseen:
                        pool = pool[~torch.isin(st.train_y.cpu(), torch.unique(y).cpu())]
                    ridx = pool[torch.randperm(len(pool), generator=gen)[:k]].to(DEVICE) if len(pool) else pool.to(DEVICE)
                    x, y = torch.cat([x, st.train_x[ridx]]), torch.cat([y, st.train_y[ridx]])
                if rs.kind == "full":
                    _train(st.model, x, y, rs.epochs, rs.lr_full, rs.bs, gen)
                else:
                    _train(st.model, x, y, rs.epochs, rs.lr, rs.bs, gen, anchor=rs.anchor if rs.kind in ("adapter", "input") else 0.0)
                st.adapted = True
            fed = [c for c in active if C[c].resp.federated and C[c].resp.kind != "align"]
            if len(fed) >= 1:
                kind = "full" if C[fed[0]].resp.kind == "full" else "adapter"
                payload = state_bytes(C[fed[0]].model) if kind == "full" else C[fed[0]].model.adapter_bytes()
                up[r] += payload * len(fed); down[r] += payload * len(fed)
                if len(fed) > 1:
                    agg = _aggregate([C[c].model for c in fed], [len(C[c].buf_round) for c in fed], kind)
                    for c in fed:
                        (C[c].model.load_state_dict(agg) if kind == "full" else C[c].model.load_adapter_state(agg))
            for c in active:
                C[c].episode_left -= 1
                C[c].model.eval()
                if C[c].episode_left == 0 and C[c].resp.kind != "full":       # cache the finished adaptation for a later recurrence
                    C[c].library.append(C[c].model.adapter_state())

        for k, c in enumerate(cids):                     # evaluation on the held-out test pool (primary endpoint)
            x, y = C[c].stream.eval_batch(r)
            xt, yt = to_tensors(x, y)
            m = classification_metrics(y, predict(C[c].model, xt), n_classes)
            f1[k, r], acc[k, r] = m["macro_f1"], m["accuracy"]

    alpha = np.stack([np.max([e.alpha(n_rounds) for e in events[c]], axis=0) if events[c] else np.zeros(n_rounds) for c in cids])
    return {"policy": policy.name, "clients": cids, "f1": f1, "acc": acc, "alpha": alpha, "up_bytes": up, "down_bytes": down, "alarms": alarms,
            "onsets": onsets, "adapt_rounds": adapt_rounds, "seconds": time.time() - t0, "labels_used": labels_used,
            "in_maps": {c: ((C[c].model.in_w.detach().cpu().numpy(), C[c].model.in_b.detach().cpu().numpy()) if isinstance(C[c].model, AdaptedLSTM) else None) for c in cids},
            "library_sizes": {c: len(C[c].library) for c in cids},
            "adapter_bytes": next((C[c].model.adapter_bytes() for c in cids if isinstance(C[c].model, AdaptedLSTM)), 0),
            "full_bytes": state_bytes(start_models[cids[0]])}
