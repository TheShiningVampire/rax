"""Experiment (exp/add-state-info/object): GPU smoke check of dexjoco_object_states.

For each task, starts one exploration episode on the real simulator through the trusted backend
(the agent's path, reply filter included), reads the object state, holds 10 steps and reads it
again. Checks: every label read without error, finite values, unit quaternions, positions on
the table's scale, arm joints present, and free-joint objects at their reset poses (the official
state's privileged tail). At the full privileged level (RAX_PRIVILEGED_STATE unset or "full")
also: each free/body object's velocities (finite, about zero at rest at step 0) and size, a
well-formed contact list, and the goal (available with conditions for the trial tasks, every
row shaped, not satisfied at step 0). Writes one JSON report and exits non-zero when any check
fails.

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
from services.dexjoco.goals import GOALS

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
            if key in ("size", "velocity_error"):  # full level: checked by check_full
                continue
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


AT_REST_MPS, AT_REST_RADPS = 0.05, 0.5
CONDITION_KEYS = {"name", "satisfied", "measured", "threshold", "note"}
CONTACT_KEYS = {"a", "b", "normal_force_n", "count", "position_m"}


def _finite_nested(value):
    if isinstance(value, (list, tuple)):
        return all(_finite_nested(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def check_full(state, env_name, *, at_rest):
    """The full level's additions (privileged_level == "full")."""
    problems, level = [], state.get("privileged_level")
    if level != O.privileged_level():
        return [f"privileged_level {level!r}, expected {O.privileged_level()!r}"]
    if level != "full":
        extra = {"contacts", "goal"} & set(state)
        return [f"objects level reply has {sorted(extra)}"] if extra else []
    for label, source in O.state_sources(env_name).items():
        if source.split(":", 1)[0] not in O.BODY_KINDS:
            continue
        entry = state["objects"].get(label, {})
        for key, limit in (("linear_velocity_mps", AT_REST_MPS), ("angular_velocity_radps", AT_REST_RADPS)):
            values = entry.get(key)
            if not (isinstance(values, list) and len(values) == 3 and _finite(values)):
                problems.append(f"{label}.{key}: {values} {entry.get('velocity_error', '')}")
            elif at_rest and math.hypot(*values) > limit:
                problems.append(f"{label}.{key} not at rest at step 0: {values}")
        size = entry.get("size", {})
        if not (len(size.get("half_extents_m", [])) == 3 and len(size.get("center_offset_m", [])) == 3
                and all(v >= 0 for v in size["half_extents_m"])):
            problems.append(f"{label}.size: {size}")
    rows = state.get("contacts")
    if not isinstance(rows, list):
        problems.append(f"contacts: {rows}")
    else:
        if len(rows) > O.MAX_CONTACT_PAIRS:
            problems.append(f"{len(rows)} contact pairs > {O.MAX_CONTACT_PAIRS}")
        forces = [row.get("normal_force_n", 0.0) for row in rows]
        if forces != sorted(forces, reverse=True):
            problems.append("contacts not sorted by force")
        for row in rows:
            if (set(row) != CONTACT_KEYS or not isinstance(row["a"], str) or not isinstance(row["b"], str)
                    or not (isinstance(row["count"], int) and row["count"] >= 1)
                    or not _finite([row["normal_force_n"], *row["position_m"]]) or len(row["position_m"]) != 3):
                problems.append(f"contact row malformed: {row}")
    goal = state.get("goal", {})
    if env_name in GOALS:  # the official env names with a goal specification
        if not goal.get("available") or not goal.get("conditions"):
            problems.append(f"goal unavailable: {goal}")
        else:
            for row in goal["conditions"] + goal.get("failure_conditions", []):
                if set(row) != CONDITION_KEYS or not isinstance(row["satisfied"], bool) or not _finite_nested(row["measured"]):
                    problems.append(f"goal row malformed: {row}")
            if not isinstance(goal.get("satisfied_now"), bool):
                problems.append("goal.satisfied_now missing")
            elif at_rest and goal["satisfied_now"]:
                problems.append("goal satisfied at step 0")
            holding = goal.get("holding")
            if holding is not None and set(holding) != {"steps_satisfied", "steps_required"}:
                problems.append(f"goal.holding malformed: {holding}")
            if not goal.get("targets"):
                problems.append("goal.targets empty")
    elif goal.get("available") is not False:
        problems.append(f"goal for {env_name}: {goal}")
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
        report["problems"] += [f"step 0: {p}" for p in check_full(first, info["env_name"], at_rest=True)]
        if images:
            report["problems"].append("the tool returned images")
        tail_problems, report["reset_tail"] = check_reset_tail(first, backend.sim.output_dir / "sim", task,
                                                               info["env_name"])
        report["problems"] += tail_problems
        backend.call(T.HOLD, {"steps": 10})
        second, _ = backend.call(T.OBJECT_STATES, {})
        report["problems"] += [f"step 10: {p}" for p in check_state(second, info["env_name"], info["dual_arm"])]
        report["problems"] += [f"step 10: {p}" for p in check_full(second, info["env_name"], at_rest=False)]
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
