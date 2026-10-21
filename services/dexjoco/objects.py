"""The task objects of each official DexJoCo environment, for operator-side episode reviews.

Stdlib only: the simulator records the measurements (services/dexjoco/session.py) and the
host reads them (services/dexjoco/analysis.py).

``TAIL``: the privileged tail of the official state vector (after the robot's 23 or 46
values), in the official config's ``proprio_keys`` order
(``DexJoCo/dexjoco/dexjoco/tasks/<task>/config.py``): each object's pose at reset (position in
metres, then a wxyz quaternion when it has 7 values), then the table height offset (1 value).
The environments fix these at reset, so the tail never holds final poses.

Live sources (``TAIL``'s third field and ``POINTS``): what the simulator reads at reset and at
the end of an episode, read-only from the environment's MuJoCo data by the official XML names:
``joint:`` a free joint's pose (its qpos, the same convention as the tail), ``body:`` a body's
world pose, ``site:``/``geom:`` a world position, ``sites:`` several sites' world positions in a
row, ``sensor:`` a sensor's value, ``hinge:`` a hinge joint's angle (rad). Every tail object with
a live source is read under its own label; the other labels are the points and angles a success
condition is defined on.
"""

BUCKET_CORNERS = 8  # bucket_ref_0..7; the even ones are its bottom (the official success check).

# Official env name -> ((label, tail values, live source or None for an untracked object), ...).
TAIL = {
    "hammer_nail": (("hammer", 7, "joint:hammer_joint"), ("nail", 7, "body:nail")),
    "click_mouse": (("mouse", 7, "joint:mouse_root"),),
    "pick_bucket": (("bucket", 7, "joint:bucket_root"), ("boxed food", 7, "joint:boxed_food_0_freejoint")),
    "pinch_tongs": (("tongs", 7, "joint:tongs_root"),),
    "fold_glasses": (("glasses", 7, "joint:glass_root"), ("glasses case", 7, "body:open_box")),
    "water_plant": (("watering can", 7, "joint:spray_root"), ("plant", 7, "body:plant")),
    "bimanual_unlock_ipad": (("stand", 7, "body:ipad_stand"), ("iPad", 7, "joint:ipad_freejoint")),
    "bimanual_hanoi": (("tower base", 3, "body:hanoi_base"),),
    "bimanual_assembly": (("tray", 7, "body:industreal_tray_insert_round_peg_8mm"),
                          ("peg", 7, "body:industreal_round_peg_8mm")),
    "bimanual_microwave_cook": (("food", 7, "joint:hot_dog_free"), ("microwave", 7, "body:microwave_object")),
    # The logo's tail pose is relative to the wall it hangs on; it never moves.
    "bimanual_photograph": (("camera", 7, "joint:camera_root"), ("logo", 7, None)),
}

# Official env name -> {label: live source} beyond the tail objects' own poses.
POINTS = {
    "click_mouse": {"mouse point": "sensor:mouse_pos", "mouse pad centre": "geom:mousepad"},
    "pick_bucket": {"boxed food centre": "sensor:boxed_food_0_pos",
                    "bucket corners": "sites:" + ",".join(f"bucket_ref_{i}" for i in range(BUCKET_CORNERS))},
    "fold_glasses": {"glasses centre": "sensor:glass_pos", "first temple hinge": "sensor:glass_joint_0_pos",
                     "second temple hinge": "sensor:glass_joint_1_pos"},
    "water_plant": {"nozzle": "site:ref_point"},
    "bimanual_assembly": {"peg point": "sensor:assembly_peg_pos", "tray point": "sensor:assembly_socket_pos"},
    "bimanual_microwave_cook": {"food centre": "sensor:hot_dog_pos", "door hinge": "hinge:microjoint"},
}


def live_sources(env_name):
    """Every live measurement of a task: {label: source}."""
    sources = {label: source for label, _, source in TAIL.get(env_name, ()) if source}
    return {**sources, **POINTS.get(env_name, {})}


# Experiment (exp/add-state-info/object): agent-visible object state. Official env name ->
# {label: live source} the episode reviews do not track but the agent's state tool reports.
# Fixtures and movable objects only; no success thresholds or success target points.
STATE_EXTRA = {
    "bimanual_hanoi": {
        **{f"{size} disk": f"joint:hanoi_disk_{size}_joint" for size in ("large", "medium", "small")},
        **{f"peg {post.upper()}": f"site:hanoi_post_{post}_site" for post in ("a", "b", "c")}},
}


def state_sources(env_name):
    """Every measurement the agent's object-state tool reports: {label: source}."""
    return {**live_sources(env_name), **STATE_EXTRA.get(env_name, {})}


def arm_joint_names(dual_arm):
    """The Panda arm joints of the official models: joint1..7, or joint<i>_right/_left."""
    if dual_arm:
        return {side: [f"joint{i}_{side}" for i in range(1, 8)] for side in ("right", "left")}
    return {"arm": [f"joint{i}" for i in range(1, 8)]}


def _state_entry(source, values):
    kind = source.split(":", 1)[0]
    if kind in ("joint", "body"):
        return {"position_m": values[:3], "quaternion_wxyz": values[3:7]}
    if kind == "hinge":
        return {"angle_rad": values[0]}
    if kind == "sites":
        return {"positions_m": [values[i:i + 3] for i in range(0, len(values), 3)]}
    if len(values) == 3:
        return {"position_m": values}
    return {"values": values}


def object_states(data, env_name, dual_arm):
    """Live world-frame state of the task's objects and fixtures and the arm joints, read-only
    from MuJoCo ``data``. A measurement that cannot be read is reported as an error, never raised."""
    objects = {}
    for label, source in state_sources(env_name).items():
        try:
            objects[label] = _state_entry(source, read_live(data, source))
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            objects[label] = {"error": f"{type(exc).__name__}: {exc}"}
    arms = {}
    for arm, names in arm_joint_names(dual_arm).items():
        try:
            arms[arm] = [float(data.joint(name).qpos[0]) for name in names]
        except Exception as exc:  # noqa: BLE001
            arms[arm] = {"error": f"{type(exc).__name__}: {exc}"}
    return {"frame": "world", "objects": objects, "arm_joint_positions_rad": arms}


def tail_size(env_name):
    """Values of the privileged tail: the objects' reset poses and the table height offset."""
    return sum(size for _, size, _ in TAIL[env_name]) + 1


def read_live(data, source):
    """One measurement from MuJoCo ``data`` (read-only), as a list of floats."""
    kind, name = source.split(":", 1)
    if kind == "joint":
        return [float(v) for v in data.joint(name).qpos[:7]]
    if kind == "body":
        body = data.body(name)
        return [float(v) for v in body.xpos] + [float(v) for v in body.xquat]
    if kind == "site":
        return [float(v) for v in data.site(name).xpos]
    if kind == "sites":
        return [float(v) for site in name.split(",") for v in data.site(site).xpos]
    if kind == "geom":
        return [float(v) for v in data.geom(name).xpos]
    if kind == "sensor":
        return [float(v) for v in data.sensor(name).data]
    if kind == "hinge":
        return [float(data.joint(name).qpos[0])]
    raise ValueError(f"Unknown live source kind: {source!r}")


def read_all(data, env_name):
    """Every live measurement of a task: {label: values}, or {label: {"error": ...}} where one
    could not be read (recorded whole, never raised: a review never breaks scoring)."""
    values = {}
    for label, source in live_sources(env_name).items():
        try:
            values[label] = read_live(data, source)
        except Exception as exc:  # noqa: BLE001 - recorded for the operator
            values[label] = {"error": f"{type(exc).__name__}: {exc}", "source": source}
    return values
