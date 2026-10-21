"""Experiment (exp/add-state-info/object): GPU smoke check of dexjoco_object_states.

For each task, starts one exploration episode on the real simulator through the trusted backend
(the agent's path, reply filter included), reads the object state, holds 10 steps and reads it
again. Checks: every label read without error, finite values, unit quaternions, positions on
the table's scale, arm joints present, and free-joint objects at their reset poses (the official
state's privileged tail). Writes one JSON report and exits non-zero when any check fails.

  RAX_RUNTIME_ROOT=<runtime> runtime/envs/host/bin/python scripts/object_states_smoke.py \
      --gpu <uuid> --out <dir> --task water_plant --task bimanual_hanoi
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from services.dexjoco import objects as O
from services.dexjoco import tasks as TK
from services.dexjoco import tools as T
from services.dexjoco.backend import DexjocoBackend

TRIAL_TASKS = ("water_plant", "fold_glasses", "bimanual_microwave_cook", "bimanual_photograph", "bimanual_hanoi")


def _finite(values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def check_state(state, env_name, dual):
    problems = []
    objects = state.get("objects", {})
    missing = set(O.state_sources(env_name)) - set(objects)
    if missing:
        problems.append(f"missing labels {sorted(missing)}")
    for label, entry in objects.items():
        if "error" in entry:
            problems.append(f"{label}: {entry['error']}")
            continue
        for key, values in entry.items():
            flat = [values] if key == "angle_rad" else [v for p in values for v in p] if key == "positions_m" else values
            if not _finite(flat):
                problems.append(f"{label}.{key} not finite: {values}")
        if "quaternion_wxyz" in entry and abs(math.hypot(*entry["quaternion_wxyz"]) - 1) > 1e-3:
            problems.append(f"{label} quaternion not unit: {entry['quaternion_wxyz']}")
        if "position_m" in entry and any(abs(v) > 3 for v in entry["position_m"]):
            problems.append(f"{label} position off the table's scale: {entry['position_m']}")
    arms = state.get("arm_joint_positions_rad", {})
    if set(arms) != ({"right", "left"} if dual else {"arm"}):
        problems.append(f"arm keys {sorted(arms)}")
    for arm, values in arms.items():
        if not (isinstance(values, list) and len(values) == 7 and _finite(values)):
            problems.append(f"arm {arm} joints {values}")
    return problems


def check_reset_tail(state, sim_dir, task, env_name):
    """Free-joint objects at step 0 must sit at the reset poses of the official state's tail."""
    initial = json.loads((sim_dir / "private_initial_state.json").read_text())["state"]
    offset, problems, compared = TK.state_dim(task), [], []
    for label, size, source in O.TAIL[env_name]:
        tail = initial[offset:offset + size]
        offset += size
        entry = state["objects"].get(label, {})
        if not (source or "").startswith("joint:") or "position_m" not in entry:
            continue
        error = max(abs(a - b) for a, b in zip(entry["position_m"], tail[:3]))
        compared.append({"label": label, "max_position_error_m": error})
        if error > 1e-4:
            problems.append(f"{label} at step 0 is {error:.2e} m from its reset pose")
    return problems, compared


def smoke(task, gpu, out):
    info = TK.task_info(task)
    config = SimpleNamespace(root=out / task, task=task, benchmark="dexjoco", observation_profile="rgbd",
                             exploration_envs=1, subagents=0, eval_seed=0, formal_collection=TK.FORMAL_COLLECTION,
                             sim_gpu=gpu, formal=SimpleNamespace(wall_seconds=1800))
    config.root.mkdir(parents=True, exist_ok=True)
    backend = DexjocoBackend(config)
    report = {"task": task, "problems": []}
    try:
        backend.start(formal=False, seed=0)
        first, images = backend.call(T.OBJECT_STATES, {})
        report["problems"] += [f"step 0: {p}" for p in check_state(first, info["env_name"], info["dual_arm"])]
        if images:
            report["problems"].append("the tool returned images")
        tail_problems, report["reset_tail"] = check_reset_tail(first, backend.sim.output_dir / "sim", task,
                                                               info["env_name"])
        report["problems"] += tail_problems
        backend.call(T.HOLD, {"steps": 10})
        second, _ = backend.call(T.OBJECT_STATES, {})
        report["problems"] += [f"step 10: {p}" for p in check_state(second, info["env_name"], info["dual_arm"])]
        if second.get("step_id") != 10:
            report["problems"].append(f"step_id after 10 holds is {second.get('step_id')}")
        report["step0"], report["step10"] = first, second
    except Exception as exc:  # noqa: BLE001 - reported
        report["problems"].append(f"{type(exc).__name__}: {exc}")
    finally:
        backend.close()
    report["ok"] = not report["problems"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", required=True, help="simulator GPU UUID (nvidia-smi -L)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", action="append", help=f"default: {', '.join(TRIAL_TASKS)}")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports = [smoke(task, args.gpu, args.out) for task in (args.task or TRIAL_TASKS)]
    (args.out / "object_states_smoke.json").write_text(json.dumps(reports, indent=2) + "\n")
    for r in reports:
        print(f"{r['task']}: {'OK' if r['ok'] else 'FAIL'}"
              + "".join(f"\n  - {p}" for p in r["problems"])
              + "".join(f"\n  reset {c['label']}: {c['max_position_error_m']:.1e} m" for c in r.get("reset_tail", [])))
    return 0 if all(r["ok"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
