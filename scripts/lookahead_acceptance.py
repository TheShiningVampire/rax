"""Experiment (exp/add-state-info/lookahead): GPU acceptance check of dexjoco_lookahead.

Runs on the real simulator through the trusted backend (the agent's path, reply filter included),
for each task (default: the five trial tasks), with RAX_LOOKAHEAD on and the privileged level the
shell sets (default full):

  A (reference): start exploration layout L; R1 (40 rows); R2 (60 rows, one dexjoco_step per row).
  B (lookahead): start layout L; R1; observe + object_states; lookahead(Rx: 150 other rows, R2
                 shifted by 5 cm and padded); lookahead(R2, report_every 10); observe +
                 object_states again; R2 one row at a time.

Checks (each one a pass/fail entry of the report):
  control         A and B agree after R1 (the two processes are twins), images included
  continuation    every row of R2: B's state vector and object states equal A's bitwise
                  (max abs diff 0), and the episode step and step budget agree
  prediction      lookahead(R2).final (state and object states) equals A after R2 bitwise
  unchanged       observe and object_states right after the lookaheads equal those before;
                  step_id, operator_state.json and its mtime unchanged; no images in the reply
  frames          the RGB images of B's first real row after the lookaheads equal A's (only
                  judged when the control images agree: rendering must be deterministic)
  episode_end     a fresh episode held to 60 steps before its cap: lookahead(150 rows) stops
                  with stop_reason episode_end at end_row 60 (end_cause step_cap at the full
                  level) while status still shows the episode active; the same rows through
                  dexjoco_step then end it after 60 rows
  budget          lookahead_budget counts the simulated rows and calls; the operator files
                  record the same use
  timing          per-row lookahead cost (recorded; a note above 15 ms per row)

Writes <out>/lookahead_acceptance.json and exits non-zero when any check fails.

  RAX_RUNTIME_ROOT=<runtime> PYTHONPATH=. runtime/envs/host/bin/python scripts/lookahead_acceptance.py \
      --gpu <uuid> --out <dir> [--task water_plant ...] [--no-episode-end]

Not covered here: the success case (end_cause task_condition from a replayed demonstration) and
the hanoi illegal-move stop; the unit tests cover both paths with fakes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402

from services.dexjoco import lookahead as LA  # noqa: E402
from services.dexjoco import rotation_math as RM  # noqa: E402
from services.dexjoco import tasks as TK  # noqa: E402
from services.dexjoco import tools as T  # noqa: E402
from services.dexjoco.backend import DexjocoBackend  # noqa: E402
from services.dexjoco.objects import privileged_level  # noqa: E402

TRIAL_TASKS = ("water_plant", "fold_glasses", "bimanual_microwave_cook", "bimanual_photograph", "bimanual_hanoi")
R1_ROWS, R2_ROWS, RX_ROWS, END_MARGIN = 40, 60, 150, 60
SLOW_ROW_SECONDS = 0.015


def backend_for(task, gpu, root):
    config = SimpleNamespace(root=root, task=task, benchmark="dexjoco", observation_profile="rgbd",
                             exploration_envs=1, subagents=0, eval_seed=0, formal_collection=TK.FORMAL_COLLECTION,
                             sim_gpu=gpu, formal=SimpleNamespace(wall_seconds=3600))
    root.mkdir(parents=True, exist_ok=True)
    return DexjocoBackend(config)


def hold_row(state):
    return RM.rotation_math({"operation": "row_from_state", "state": [float(v) for v in state]})["row"]


def plan(state, dual, rows, drop, shift=0.0, start=0):
    """Rows from the hold row of ``state``: each flange moves down ``drop`` m in total (and along
    +x by 0.3 of it, +y by ``shift``) while the fingers close part way; deterministic."""
    base = np.asarray(hold_row(state))
    out = []
    for i in range(1 + start, rows + 1 + start):
        row, f = base.copy(), min(1.0, i / max(1, rows))
        for s in ((0, 22) if dual else (0,)):
            row[s:s + 3] += [0.3 * drop * f, shift, -drop * f]
            row[s + 7:s + 10] = np.clip(row[s + 7:s + 10] + 0.5 * f, -0.2, 1.5)  # index finger flexion
        out.append([float(v) for v in row])
    return out


def strip(state):
    return {k: v for k, v in state.items() if k not in ("episode_id", "step_id", "episode_ended", "episode_limits")}


def max_diff(a, b):
    """Max abs difference between two JSON values of the same shape (inf when shapes differ)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return math.inf
        return max([max_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return math.inf
        return max([max_diff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if a == b else math.inf
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return 0.0
        return abs(float(a) - float(b))
    return 0.0 if a == b else math.inf


def images(content):
    return [item["data"] for item in content if item.get("type") == "image"]


def step_rows(backend, rows):
    """One dexjoco_step per row: the state, object states and images after each."""
    out = []
    for row in rows:
        packet, content = backend.call(T.STEP, {"actions": [row]})
        states, _ = backend.call(T.OBJECT_STATES, {})
        out.append({"step_id": packet["step_id"], "state": packet["state"], "object_states": strip(states),
                    "images": images(content), "episode_limits": packet.get("episode_limits")})
    return out


def check(report, name, ok, **details):
    report["checks"][name] = {"ok": bool(ok), **details}
    if not ok:
        report["problems"].append(f"{name}: {json.dumps(details, default=str)[:600]}")


def operator_state(backend):
    path = backend.sim.output_dir / "sim" / "operator_state.json"
    return path, json.loads(path.read_text()), path.stat().st_mtime_ns


def twin_check(task, gpu, out):
    info = TK.task_info(task)
    dual = info["dual_arm"]
    report = {"task": task, "privileged_level": privileged_level(), "checks": {}, "problems": [], "timing": {}}
    a, b = backend_for(task, gpu, out / task / "A"), backend_for(task, gpu, out / task / "B")
    try:
        start_a, _ = a.start(formal=False, seed=0)
        start_b, _ = b.start(formal=False, seed=0)
        r1 = plan(start_a["state"], dual, R1_ROWS, 0.10)
        pa, ca = a.call(T.STEP, {"actions": r1})
        pb, cb = b.call(T.STEP, {"actions": r1})
        sa, sb = strip(a.call(T.OBJECT_STATES, {})[0]), strip(b.call(T.OBJECT_STATES, {})[0])
        control_images = images(ca) == images(cb)
        check(report, "control", pa["state"] == pb["state"] and sa == sb, state_diff=max_diff(pa["state"], pb["state"]),
              object_states_diff=max_diff(sa, sb), images_equal=control_images)

        r2 = plan(pa["state"], dual, R2_ROWS, 0.08)
        rx = [[v + (0.05 if (i % 22) == 1 else 0.0) for i, v in enumerate(row)] for row in r2]
        rx += [rx[-1]] * (RX_ROWS - len(rx))
        observe_before, _ = b.call(T.OBSERVE, {})
        path, state_before, mtime_before = operator_state(b)
        step_before = b.sim.step_id

        t0 = time.monotonic()
        other, other_images = b.call(T.LOOKAHEAD, {"actions": rx})
        wall_other = time.monotonic() - t0
        t0 = time.monotonic()
        predicted, predicted_images = b.call(T.LOOKAHEAD, {"actions": r2, "report_every": 10})
        wall_r2 = time.monotonic() - t0
        observe_after, _ = b.call(T.OBSERVE, {})
        states_after = strip(b.call(T.OBJECT_STATES, {})[0])
        _, state_after, mtime_after = operator_state(b)
        check(report, "unchanged",
              observe_after.get("state") == observe_before.get("state") and states_after == sb
              and b.sim.step_id == step_before == R1_ROWS and mtime_after == mtime_before
              and state_after == state_before and other_images == [] and predicted_images == [],
              object_states_diff=max_diff(states_after, sb), step_id=b.sim.step_id,
              operator_state_rewritten=mtime_after != mtime_before)
        check(report, "lookahead_replies",
              other.get("rows_simulated") == RX_ROWS and predicted.get("rows_simulated") == R2_ROWS
              and predicted.get("live_state_restored") is True and len(predicted.get("reports", [])) == R2_ROWS // 10 - 1
              and other.get("stop_reason") is None and predicted.get("stop_reason") is None,
              rows=[other.get("rows_simulated"), predicted.get("rows_simulated")],
              stop=[other.get("stop_reason"), predicted.get("stop_reason")])

        ra, rb = step_rows(a, r2), step_rows(b, r2)
        worst = max(max(max_diff(x["state"], y["state"]), max_diff(x["object_states"], y["object_states"]))
                    for x, y in zip(ra, rb))
        same = all(x["state"] == y["state"] and x["object_states"] == y["object_states"]
                   and x["step_id"] == y["step_id"] and x["episode_limits"] == y["episode_limits"]
                   for x, y in zip(ra, rb))
        check(report, "continuation", same and worst == 0.0, rows=len(rb), max_abs_diff=worst)
        final_a = ra[-1]
        check(report, "prediction",
              predicted["final"]["state"] == final_a["state"] and predicted["final"]["object_states"] == final_a["object_states"],
              state_diff=max_diff(predicted["final"]["state"], final_a["state"]),
              object_states_diff=max_diff(predicted["final"]["object_states"], final_a["object_states"]))
        frames_equal = rb[0]["images"] == ra[0]["images"]
        check(report, "frames", frames_equal or not control_images, images_equal=frames_equal,
              judged=control_images)

        budget = predicted["lookahead_budget"]
        limits = LA.limits(TK.native_step_limit(task))
        _, recorded, _ = operator_state(b)
        check(report, "budget",
              budget["rows_used"] == RX_ROWS + R2_ROWS and budget["calls_used"] == 2
              and budget["rows_left"] == limits["rows_per_episode"] - RX_ROWS - R2_ROWS
              and budget["calls_left"] == limits["calls_per_episode"] - 2
              and recorded.get("lookahead", {}).get("rows") == RX_ROWS + R2_ROWS
              and recorded.get("lookahead", {}).get("calls") == 2,
              budget=budget, recorded=recorded.get("lookahead"))
        per_row = [other["rollout_s_per_row"], predicted["rollout_s_per_row"]]
        report["timing"] = {"rollout_s_per_row": per_row, "elapsed_s": [other["elapsed_s"], predicted["elapsed_s"]],
                            "host_wall_s": [round(wall_other, 3), round(wall_r2, 3)]}
        if max(per_row) > SLOW_ROW_SECONDS:
            report["notes"] = [f"lookahead costs {max(per_row) * 1000:.1f} ms per row (> {SLOW_ROW_SECONDS * 1000:g})"]
    except Exception as exc:  # noqa: BLE001 - reported
        report["problems"].append(f"{type(exc).__name__}: {exc}")
    finally:
        for backend in (a, b):
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001
                report["problems"].append(f"close: {type(exc).__name__}: {exc}")
    return report


def episode_end_check(task, gpu, out, report):
    info = TK.task_info(task)
    limit = TK.native_step_limit(task)
    c = backend_for(task, gpu, out / task / "C")
    try:
        start, _ = c.start(formal=False, seed=1)
        remaining = limit - END_MARGIN
        t0 = time.monotonic()
        while remaining > 0:
            n = min(T.MAX_ACTION_ROWS, remaining)
            c.call(T.HOLD, {"steps": n})
            remaining -= n
        report["timing"]["real_hold_s_per_row"] = round((time.monotonic() - t0) / (limit - END_MARGIN), 4)
        state, _ = c.call(T.OBSERVE, {})
        rows = [hold_row(state["state"])] * T.MAX_ACTION_ROWS
        reply, _ = c.call(T.LOOKAHEAD, {"actions": rows})
        status, _ = c.call(T.STATUS, {})
        expected_cause = "step_cap" if privileged_level() == "full" else None
        ok = (reply.get("stop_reason") == "episode_end" and reply.get("end_row") == END_MARGIN
              and reply.get("rows_simulated") == END_MARGIN and reply.get("end_cause") == expected_cause
              and status.get("active") is True and status.get("step_id") == limit - END_MARGIN)
        packet, _ = c.call(T.STEP, {"actions": rows})
        ended = packet.get("transition", {}).get("episode_ended") is True and c.sim.step_id == limit
        check(report, "episode_end", ok and ended, stop_reason=reply.get("stop_reason"), end_row=reply.get("end_row"),
              end_cause=reply.get("end_cause"), active_after_lookahead=status.get("active"),
              step_after_real_rows=c.sim.step_id, ended=ended, env_step_cap=info["env_step_cap"])
    except Exception as exc:  # noqa: BLE001 - reported
        check(report, "episode_end", False, error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            c.close()
        except Exception as exc:  # noqa: BLE001
            report["problems"].append(f"close: {type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", required=True, help="simulator GPU UUID (nvidia-smi -L)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", action="append", help=f"default: {', '.join(TRIAL_TASKS)}")
    parser.add_argument("--no-episode-end", action="store_true", help="skip the hold-to-the-cap episode-end case")
    args = parser.parse_args()
    if not LA.lookahead_enabled():
        parser.error(f"{LA.LOOKAHEAD_ENV} is off")
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for task in args.task or TRIAL_TASKS:
        report = twin_check(task, args.gpu, args.out)
        if not args.no_episode_end:
            episode_end_check(task, args.gpu, args.out, report)
        report["ok"] = not report["problems"] and all(c["ok"] for c in report["checks"].values())
        reports.append(report)
        print(f"{task}: {'OK' if report['ok'] else 'FAIL'} "
              + " ".join(f"{k}={'ok' if v['ok'] else 'FAIL'}" for k, v in report["checks"].items())
              + f" timing={report['timing']}" + "".join(f"\n  - {p}" for p in report["problems"]), flush=True)
    summary = {"schema": "dexjoco_lookahead_acceptance_v1", "created_unix_s": time.time(),
               "lookahead": os.environ.get(LA.LOOKAHEAD_ENV, LA.DEFAULT_LOOKAHEAD),
               "privileged_level": privileged_level(), "ok": all(r["ok"] for r in reports), "tasks": reports}
    (args.out / "lookahead_acceptance.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"report: {args.out / 'lookahead_acceptance.json'}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
