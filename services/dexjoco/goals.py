"""Experiment (exp/add-state-info/object): each trial task's goal specification for the agent's
privileged state tool at the ``full`` level (services/dexjoco/objects.py object_states).

One function per task, ``fn(env, data)`` with the raw official environment (``tap.unwrapped``)
and its MuJoCo data, returning::

    {"available": True, "description": str,
     "conditions": [{"name", "satisfied", "measured", "threshold", "note"}, ...],
     "failure_conditions": [...same shape...],
     "targets": {label: {"position_m": [...], ...}},
     "holding": {"steps_satisfied": int, "steps_required": int},   # consecutive-step tasks only
     "satisfied_now": bool}

The conditions are the official success check (``_compute_success`` / ``_compute_success_metrics``
of ``DexJoCo/dexjoco/dexjoco/sim/envs/panda_<task>_env.py``), recomputed read-only from the same
MuJoCo data and environment attributes with the official constants. The official checks are
never called where they mutate state: water_plant, fold_glasses and bimanual_hanoi update their
consecutive-step counters inside the check, so those counters are read as attributes and the
conditions recomputed. Constants are read from the environment (attributes) or its module
(module-level constants); the few that are local literals inside the official check are named
below with their source and guarded by a unit test against the official source.

``satisfied_now``: every success condition holds at this step. Tasks with ``holding`` complete
once ``steps_satisfied`` reaches ``steps_required`` (the environment counts the consecutive
steps at the end of each step); the others complete on the step all conditions hold.
No key is named like an evaluation field (services/mcp_reply.py drops those at any depth).
"""
from __future__ import annotations

import sys

import numpy as np

# Local literals of the official checks (no attribute or module constant holds them).
# panda_water_plant_env.py PandaWaterPlantGymEnv._compute_success: R = 0.2, HALF_H = 0.2.
WATER_RADIUS_M = 0.2
WATER_HALF_HEIGHT_M = 0.2
# panda_fold_glasses_env.py PandaFoldGlassesGymEnv._compute_success: j0 > 1.1 and j1 > 1.1,
# x_half = y_half = 0.145, z_min = -0.045, z_max = 0.0475 (the check uses z <= z_max - 0.02).
TEMPLE_FOLDED_RAD = 1.1
CASE_HALF_XY_M = 0.145
CASE_Z_MIN_M = -0.045
CASE_Z_MAX_M = 0.0475
CASE_Z_TOP_ALLOWANCE_M = 0.02
# panda_bimanual_microwave_cook_env.py _compute_success: abs(micro_qpos) < 1e-2,
# hot_dog_inside_microwave(margin=0.01); _geom_in_contact's default dist_threshold is 0.0.
DOOR_CLOSED_RAD = 1e-2
CAVITY_MARGIN_M = 0.01
# panda_bimanual_hanoi_env.py _compute_success_metrics: target_stack; _reconstruct_tower_state:
# pos[2] <= base_top_z + 0.06 is "table", higher off-post disks are "floating".
HANOI_TARGET_STACK = ("hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small")
HANOI_TABLE_MARGIN_M = 0.06
HANOI_LABELS = {"hanoi_disk_large": "large disk", "hanoi_disk_medium": "medium disk",
                "hanoi_disk_small": "small disk"}


def _official(env, name):
    """A module-level constant of the official environment's module (or a base class's)."""
    for cls in type(env).__mro__:
        module = sys.modules.get(cls.__module__)
        if module is not None and hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"official constant {name} not found for {type(env).__name__}")


def _vec(values):
    return [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def _scalar(values):
    return float(np.asarray(values, dtype=np.float64).reshape(-1)[0])


def _condition(name, satisfied, measured, threshold, note):
    return {"name": name, "satisfied": bool(satisfied), "measured": measured, "threshold": threshold, "note": note}


def water_plant(env, data):
    model = env._model
    nozzle = _vec(data.site_xpos[model.site("ref_point").id])
    plant = _vec(data.body("plant").xpos)
    dx, dy, dz = (nozzle[i] - plant[i] for i in range(3))
    radius, half = WATER_RADIUS_M, WATER_HALF_HEIGHT_M
    in_radius = dx * dx + dy * dy <= radius * radius
    in_band = -half <= dz <= half
    inside = in_radius and in_band
    angle = _scalar(data.sensor("spray_joint_0_pos").data)
    pulled = bool(env._trigger_pulled)
    pull, release = float(_official(env, "_TRIGGER_PULL_THRESHOLD")), float(_official(env, "_TRIGGER_RELEASE_THRESHOLD"))
    required = int(_official(env, "_SUCCESS_STEPS_REQUIRED"))
    conditions = [
        _condition("nozzle_within_radius_of_plant", in_radius, (dx * dx + dy * dy) ** 0.5, radius,
                   "horizontal distance (m) from the nozzle point to the plant body origin"),
        _condition("nozzle_within_height_band_of_plant", in_band, dz, [-half, half],
                   "nozzle height (m) above the plant body origin, world z"),
        _condition("trigger_pulled", pulled, angle, [release, pull],
                   f"trigger hinge angle (rad); the latch turns on above {pull} rad and off below {release} rad, "
                   "in between it keeps its last value (satisfied is the latch, updated after each step)"),
    ]
    failures = [_condition(
        "trigger_pulled_outside_region", (not inside) and pulled, [(dx * dx + dy * dy) ** 0.5, dz], [radius, half],
        "the trigger latch on while the nozzle is outside the region ends the episode as a failure; "
        "measured: [horizontal distance, height offset] of the nozzle from the plant origin")]
    return {
        "available": True,
        "description": (f"Hold the watering can's nozzle within {radius} m horizontally and {half} m vertically of "
                        f"the plant's body origin with the trigger pulled (above {pull} rad) for {required} "
                        "consecutive steps. Pulling the trigger with the nozzle outside that region fails the episode."),
        "conditions": conditions, "failure_conditions": failures,
        "targets": {
            "plant": {"position_m": plant, "radius_m": radius, "half_height_m": half, "point": "nozzle",
                      "note": "vertical cylinder around the plant body origin, world axes"},
            "nozzle": {"position_m": nozzle},
            "trigger": {"angle_rad": angle, "pull_above_rad": pull, "release_below_rad": release}},
        "holding": {"steps_satisfied": int(env._success_counter), "steps_required": required},
        "satisfied_now": bool(inside and pulled),
    }


def fold_glasses(env, data):
    model = env._model
    glass = _vec(data.sensor("glass_pos").data)
    case = _vec(model.body_pos[model.body("open_box").id])
    local = [glass[i] - case[i] for i in range(3)]
    j0 = _scalar(data.sensor("glass_joint_0_pos").data)
    j1 = _scalar(data.sensor("glass_joint_1_pos").data)
    z_top = CASE_Z_MAX_M - CASE_Z_TOP_ALLOWANCE_M
    bounds = [[-CASE_HALF_XY_M, CASE_HALF_XY_M], [-CASE_HALF_XY_M, CASE_HALF_XY_M], [CASE_Z_MIN_M, z_top]]
    inside = [lo <= v <= hi for v, (lo, hi) in zip(local, bounds)]
    offset_note = ("glasses centre minus the case origin (m), world axes: the official check does not rotate "
                   "into the case's yaw")
    conditions = [
        _condition("first_temple_folded", j0 > TEMPLE_FOLDED_RAD, j0, TEMPLE_FOLDED_RAD,
                   "first temple hinge angle (rad), must exceed the threshold"),
        _condition("second_temple_folded", j1 > TEMPLE_FOLDED_RAD, j1, TEMPLE_FOLDED_RAD,
                   "second temple hinge angle (rad), must exceed the threshold"),
        *(_condition(f"glasses_inside_case_{axis}", ok, value, limits, offset_note)
          for axis, ok, value, limits in zip("xyz", inside, local, bounds)),
    ]
    required = int(env._success_trigger_target)
    return {
        "available": True,
        "description": (f"Fold both temples past {TEMPLE_FOLDED_RAD} rad and put the glasses centre inside the "
                        f"case (within {CASE_HALF_XY_M} m of its origin in x and y, {CASE_Z_MIN_M}..{z_top:.4g} m "
                        f"in z), all at once for {required} consecutive steps."),
        "conditions": conditions, "failure_conditions": [],
        "targets": {
            "glasses case": {"position_m": case, "accepted_offset_m": bounds, "point": "glasses centre",
                             "note": "the accepted box of the glasses centre around the case origin, world axes"},
            "glasses centre": {"position_m": glass}},
        "holding": {"steps_satisfied": int(env._success_trigger_count), "steps_required": required},
        "satisfied_now": bool(j0 > TEMPLE_FOLDED_RAD and j1 > TEMPLE_FOLDED_RAD and all(inside)),
    }


def bimanual_microwave_cook(env, data):
    model = env._model
    door = _scalar(data.joint("microjoint").qpos)
    closed = abs(door) < DOOR_CLOSED_RAD
    food = np.asarray(data.sensor("hot_dog_pos").data, dtype=np.float64).reshape(3)
    body = data.body("microwave_object")
    rotation = np.asarray(body.xmat, dtype=np.float64).reshape(3, 3)
    origin = np.asarray(body.xpos, dtype=np.float64)
    local = rotation.T @ (food - origin)
    p0, px, py, pz = (np.asarray(model.site(name).pos, dtype=np.float64)
                      for name in ("int_p0", "int_px", "int_py", "int_pz"))
    m = CAVITY_MARGIN_M
    bounds = [[float(p0[0] - m), float(px[0] + m)], [float(p0[1] - m), float(py[1] + m)],
              [float(p0[2] - m), float(pz[2] + m)]]
    inside = bool(env.hot_dog_inside_microwave(margin=m))
    button = int(model.geom("start_button").id)
    touches, touching = 0, set()
    for i in range(int(data.ncon)):
        g1, g2 = int(data.contact.geom1[i]), int(data.contact.geom2[i])
        if button in (g1, g2) and float(data.contact.dist[i]) <= 0.0:
            touches += 1
            other = g2 if g1 == button else g1
            touching.add(model.body(int(model.geom_bodyid[other])).name or f"body{int(model.geom_bodyid[other])}")
    pressed = bool(env._geom_in_contact("start_button"))
    centre_local = np.array([(lo + hi) / 2 for lo, hi in bounds])
    conditions = [
        _condition("door_closed", closed, door, DOOR_CLOSED_RAD, "door hinge angle (rad); |angle| must be below the "
                   "threshold (open is negative)"),
        _condition("food_inside_cavity", inside, [float(v) for v in local], bounds,
                   f"food centre in the microwave body frame (m); the cavity box from its interior sites, "
                   f"widened by {m} m"),
        _condition("start_button_touched", pressed, touches, ">= 1 contact with dist <= 0",
                   "contacts on the start button at penetration; anything may touch it"
                   + (f"; touching: {', '.join(sorted(touching))}" if touching else "")),
    ]
    return {
        "available": True,
        "description": ("Put the food inside the microwave cavity, close the door (|hinge| < "
                        f"{DOOR_CLOSED_RAD} rad) and touch the start button, all at the same step."),
        "conditions": conditions, "failure_conditions": [],
        "targets": {
            "microwave cavity": {"position_m": [float(v) for v in origin + rotation @ centre_local],
                                 "half_extents_m": [float((hi - lo) / 2) for lo, hi in bounds],
                                 "frame": "microwave body axes", "point": "food centre"},
            "start button": {"position_m": _vec(data.geom("start_button").xpos),
                             "half_extents_m": _vec(model.geom("start_button").size)},
            "door handle": {"position_m": _vec(data.geom("door_handle").xpos)},
            "door hinge": {"angle_rad": 0.0, "tolerance_rad": DOOR_CLOSED_RAD}},
        "satisfied_now": bool(closed and inside and pressed),
    }


def bimanual_photograph(env, data):
    # The official metric is side-effect free (it reads MuJoCo data only).
    complete, metrics = env._compute_success_metrics()
    radius = float(_official(env, "_TARGET_REGION_RADIUS"))
    half = float(_official(env, "_TARGET_REGION_HALF_HEIGHT"))
    max_angle = float(_official(env, "_TARGET_ANGLE_DEG"))
    direction = _vec(_official(env, "_WORLD_POS_X"))
    region = env._target_region_geom_id
    conditions = [
        _condition("lens_in_target_region", metrics["region_pass"],
                   [float(metrics["radial_dist"]), float(metrics["region_height_dist"])], [radius, half],
                   "lens point (view_direction_site) in the target cylinder's frame: [radial distance, "
                   "distance along the axis] (m)"),
        _condition("camera_aimed_along_x", metrics["angle_pass"], float(metrics["view_angle_deg"]), max_angle,
                   f"angle (deg) between the camera's optical axis and world {direction}; the logo faces it"),
        _condition("shutter_pressed", metrics["shutter_pressed"], int(metrics["shutter_contact_count"]),
                   ">= 1 contact", "contacts on the shutter button (camera top cap) from any geom outside the camera"),
    ]
    rotation = np.asarray(data.geom_xmat[region], dtype=np.float64).reshape(3, 3)
    return {
        "available": True,
        "description": (f"Put the camera's lens point inside the target cylinder in front of the logo (radius "
                        f"{radius} m, half height {half} m), aim the camera within {max_angle} deg of world "
                        f"{direction}, and press the shutter with something other than the camera, all at the same step."),
        "conditions": conditions, "failure_conditions": [],
        "targets": {
            "target region": {"position_m": _vec(data.geom_xpos[region]), "axis_world": _vec(rotation[:, 2]),
                              "radius_m": radius, "half_height_m": half, "point": "lens point"},
            "lens point": {"position_m": _vec(data.site_xpos[env._view_site_id])},
            "shutter button": {"position_m": _vec(data.geom_xpos[env._shutter_geom_id])},
            "aim direction": {"direction_world": direction, "max_angle_deg": max_angle}},
        "satisfied_now": bool(complete),
    }


def bimanual_hanoi(env, data):
    # Side-effect-free official helpers; _compute_success_metrics itself updates the counters.
    state = env._reconstruct_tower_state()
    towers, positions, posts = state["towers"], state["disk_positions"], state["post_positions"]
    base_top, static, assignment = state["base_top_z"], state["disk_static"], state["disk_assignment"]
    target = list(HANOI_TARGET_STACK)
    xy_tol, height_tol = float(env._post_assignment_tol), float(env._placement_height_tol)
    qvel_tol = float(env._static_qvel_tol)
    label = lambda disk: HANOI_LABELS.get(disk, disk)  # noqa: E731
    order = list(towers["C"]) == target and not towers["A"] and not towers["B"]
    xy_error, height_error = env._tower_geometry_errors(target, "C", positions, posts, base_top)
    norms = {disk: float(np.linalg.norm(np.asarray(data.qvel[env._disk_qvel_adr[disk]:env._disk_qvel_adr[disk] + 6],
                                                   dtype=np.float64))) for disk in env._disk_names}
    all_static = all(bool(v) for v in static.values())
    illegal_count, illegal_required = int(env._illegal_trigger_count), int(env._illegal_trigger_target)
    illegal_reason = str(env._illegal_trigger_reason or "")
    not_illegal = illegal_count < illegal_required
    # The official illegal-state candidate, recomputed (the first matching post wins).
    candidate = ""
    for post, disks in towers.items():
        ranks = [env._disk_size_rank[d] for d in disks]
        if any(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1)) and all(static[d] for d in disks):
            post_xy, post_height = env._tower_geometry_errors(disks, post, positions, posts, base_top)
            if post_xy <= xy_tol and post_height <= height_tol:
                candidate = f"illegal_stack_{post}"
                break
    if not candidate and all_static and any(a == "table" for a in assignment.values()):
        candidate = "disk_on_table"
    tower_text = {post: [label(d) for d in disks] for post, disks in towers.items()}
    conditions = [
        _condition("tower_C_order", order, tower_text["C"], [label(d) for d in target],
                   f"disks on peg C from the bottom; pegs A and B must be empty (A: {tower_text['A']}, "
                   f"B: {tower_text['B']}); a disk is on a peg within {xy_tol} m of it horizontally"),
        _condition("xy_centred_on_peg_C", xy_error <= xy_tol, float(xy_error), xy_tol,
                   "largest horizontal distance (m) of the target stack's disks from peg C (counted only with the order)"),
        _condition("stack_height_on_peg_C", height_error <= height_tol, float(height_error), height_tol,
                   "largest height error (m) of the target stack's disks from their stacked heights on the base"),
        _condition("all_disks_static", all_static, [norms[d] for d in env._disk_names], qvel_tol,
                   "norm of each disk's 6-D free-joint velocity [" + ", ".join(label(d) for d in env._disk_names) + "]"),
        _condition("no_illegal_state", not_illegal, illegal_count, illegal_required,
                   "consecutive steps of the current illegal state; it must stay below the threshold"),
    ]
    stack_count = illegal_count if illegal_reason.startswith("illegal_stack") else 0
    table_count = illegal_count if illegal_reason == "disk_on_table" else 0
    failures = [
        _condition("illegal_stack", candidate.startswith("illegal_stack"), stack_count, illegal_required,
                   "a bigger disk resting on a smaller one, seated and static, ends the episode after the "
                   f"threshold's consecutive steps (current: {candidate or 'none'})"),
        _condition("disk_on_table", candidate == "disk_on_table", table_count, illegal_required,
                   f"all disks static and one off the pegs within {HANOI_TABLE_MARGIN_M} m above the base top ends "
                   "the episode after the threshold's consecutive steps"),
    ]
    targets, height = {}, base_top
    for disk in target:
        half = float(env._disk_half_height[disk])
        targets[label(disk)] = {"position_m": [float(posts["C"][0]), float(posts["C"][1]), float(height + half)]}
        height += 2 * half
    targets["peg C"] = {"position_m": _vec(posts["C"])}
    geometry = order and xy_error <= xy_tol and height_error <= height_tol and all_static
    required = int(env._success_trigger_target)
    return {
        "available": True,
        "description": ("Stack all three disks on peg C (the right peg), large at the bottom, then medium, then "
                        f"small, each centred on the peg and seated, with every disk static, for {required} "
                        "consecutive steps. A bigger disk resting on a smaller one, or a disk left on the table, "
                        f"for {illegal_required} consecutive steps fails the episode."),
        "conditions": conditions, "failure_conditions": failures, "targets": targets,
        "tower_state": tower_text, "disk_assignment": {label(d): a for d, a in assignment.items()},
        "holding": {"steps_satisfied": int(env._success_trigger_count), "steps_required": required},
        "failure_holding": {"reason": illegal_reason or None, "steps_satisfied": illegal_count,
                            "steps_required": illegal_required},
        "satisfied_now": bool(geometry and not_illegal),
    }


GOALS = {"water_plant": water_plant, "fold_glasses": fold_glasses,
         "bimanual_microwave_cook": bimanual_microwave_cook, "bimanual_photograph": bimanual_photograph,
         "bimanual_hanoi": bimanual_hanoi}


def goal(env, data, env_name):
    """The task's goal specification, {"available": False} for tasks without one; an error is
    reported, never raised."""
    fn = GOALS.get(env_name)
    if fn is None:
        return {"available": False}
    if env is None:
        return {"available": False, "error": "no environment"}
    try:
        return fn(env, data)
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
