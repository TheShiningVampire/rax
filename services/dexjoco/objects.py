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

Experiment (exp/add-state-info/object): ``object_states`` is the agent's privileged state tool,
at the level ``RAX_PRIVILEGED_STATE`` selects (``privileged_level``): ``objects`` (poses, points,
angles, arm joints) or ``full`` (the default), which adds each free-joint/body object's velocity
and collision bounding box, the contacts on task objects and hands, and the task's goal
specification (services/dexjoco/goals.py). The full-level parts read numpy and mujoco lazily.
"""
import os

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
# A label already in live_sources is replaced (e.g. a 1-value sensor read as the hinge angle it is).
STATE_EXTRA = {
    "bimanual_hanoi": {
        **{f"{size} disk": f"joint:hanoi_disk_{size}_joint" for size in ("large", "medium", "small")},
        **{f"peg {post.upper()}": f"site:hanoi_post_{post}_site" for post in ("a", "b", "c")}},
    "water_plant": {"trigger": "hinge:joint_0"},
    "fold_glasses": {"first temple hinge": "hinge:glass_joint_0", "second temple hinge": "hinge:glass_joint_1"},
    # The logo is a geom on the wall, placed per layout (untracked in TAIL).
    "bimanual_photograph": {"logo": "geompose:photograph_logo_disk"},
}


def state_sources(env_name):
    """Every measurement the agent's object-state tool reports: {label: source}."""
    return {**live_sources(env_name), **STATE_EXTRA.get(env_name, {})}


def arm_joint_names(dual_arm):
    """The Panda arm joints of the official models: joint1..7, or joint<i>_right/_left."""
    if dual_arm:
        return {side: [f"joint{i}_{side}" for i in range(1, 8)] for side in ("right", "left")}
    return {"arm": [f"joint{i}" for i in range(1, 8)]}


def _quaternion_wxyz(m):
    """Unit quaternion of a row-major 3x3 rotation matrix (9 values)."""
    r = [[float(m[3 * i + j]) for j in range(3)] for i in range(3)]
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = 2.0 * (trace + 1.0) ** 0.5
        q = [s / 4, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s]
    else:
        i = max(range(3), key=lambda k: r[k][k])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * (1.0 + r[i][i] - r[j][j] - r[k][k]) ** 0.5
        q = [0.0] * 4
        q[0] = (r[k][j] - r[j][k]) / s
        q[1 + i], q[1 + j], q[1 + k] = s / 4, (r[j][i] + r[i][j]) / s, (r[k][i] + r[i][k]) / s
    if q[0] < 0:
        q = [-v for v in q]
    return q


def read_state(data, source):
    """One measurement for the object-state tool: read_live, plus a geom's world pose."""
    kind, name = source.split(":", 1)
    if kind == "geompose":
        geom = data.geom(name)
        return [float(v) for v in geom.xpos] + _quaternion_wxyz(geom.xmat)
    return read_live(data, source)


def _state_entry(source, values):
    kind = source.split(":", 1)[0]
    if kind in ("joint", "body", "geompose"):
        return {"position_m": values[:3], "quaternion_wxyz": values[3:7]}
    if kind == "hinge":
        return {"angle_rad": values[0]}
    if kind == "sites":
        return {"positions_m": [values[i:i + 3] for i in range(0, len(values), 3)]}
    if len(values) == 3:
        return {"position_m": values}
    return {"values": values}


PRIVILEGED_ENV = "RAX_PRIVILEGED_STATE"
PRIVILEGED_LEVELS = ("objects", "full")
# RAX_* settings every process between the launcher and the simulator keeps when it strips the
# other RAX_* variables (the launcher's frontend, the exploration worker).
SESSION_KEPT_ENV = frozenset({PRIVILEGED_ENV})
DEFAULT_PRIVILEGED_LEVEL = "full"
MAX_CONTACT_PAIRS = 64
# The robot models' root bodies (single arm, then the bimanual right and left arms) and hands.
ROBOT_ROOTS = ("base", "base_right", "base_left")
HAND_ROOTS = ("allegro_palm", "allegro_palm_right", "allegro_palm_left")
HAND_BODIES = ("allegro_attachment", "allegro_attachment_right", "allegro_attachment_left")
# Sources whose object is a body with its own pose: they report velocities and a size.
BODY_KINDS = ("joint", "body")


def privileged_level(environ=None):
    """The agent's privileged-state level: RAX_PRIVILEGED_STATE, ``objects`` or ``full`` (default).
    Read the same way by the host (tool text, workspace) and the simulator (the reply)."""
    value = (os.environ if environ is None else environ).get(PRIVILEGED_ENV, DEFAULT_PRIVILEGED_LEVEL)
    if value not in PRIVILEGED_LEVELS:
        raise ValueError(f"{PRIVILEGED_ENV} must be one of {PRIVILEGED_LEVELS}, not {value!r}")
    return value


def _error(exc):
    return {"error": f"{type(exc).__name__}: {exc}"}


def _source_body(model, source):
    """The body a joint:/body:/hinge: source's object is (MuJoCo body id), else None."""
    kind, name = source.split(":", 1)
    if kind in ("joint", "hinge"):
        return int(model.jnt_bodyid[model.joint(name).id])
    if kind == "body":
        return int(model.body(name).id)
    return None


def _ancestors(model, body):
    """body, its parent, ... up to (not including) the world body."""
    while body > 0:
        yield body
        body = int(model.body_parentid[body])


def scene_index(model, data, env_name):
    """Per-episode facts for the full level, computed once at reset: each body's task-object label
    (the nearest labelled ancestor: free joints, bodies and hinges of state_sources; geom poses and
    points label no body), the robot's and the hands' bodies, each body name, and the size of each
    free-joint/body object: the axis-aligned bounding box of its subtree's collision geoms (contype
    or conaffinity set) in the object's own frame, at the reset pose."""
    import numpy as np

    sources = state_sources(env_name)
    labelled, objects = {}, {}
    # Objects first, then hinges: a hinge's body is labelled only when no object owns it directly.
    for label, source in sorted(sources.items(), key=lambda item: item[1].startswith("hinge:")):
        if source.split(":", 1)[0] not in BODY_KINDS + ("hinge",):
            continue
        try:
            body = _source_body(model, source)
        except Exception:  # noqa: BLE001 - an unreadable source labels nothing
            continue
        if body and body > 0:
            labelled.setdefault(body, label)
            if source.split(":", 1)[0] in BODY_KINDS:
                objects[label] = body
    names = [model.body(i).name or f"body{i}" for i in range(model.nbody)]
    owner, robot, hand = {}, set(), set()
    for body in range(1, model.nbody):
        chain = list(_ancestors(model, body))
        owner[body] = next((labelled[b] for b in chain if b in labelled), None)
        chain_names = {names[b] for b in chain}
        if chain_names & set(ROBOT_ROOTS):
            robot.add(body)
            if chain_names & set(HAND_ROOTS) or names[body] in HAND_BODIES:
                hand.add(body)
    sizes = {}
    collide = [g for g in range(model.ngeom) if int(model.geom_contype[g]) | int(model.geom_conaffinity[g])]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    for label, body in objects.items():
        try:
            subtree = {b for b in range(1, model.nbody) if body in _ancestors(model, b)}
            geoms = [g for g in collide if int(model.geom_bodyid[g]) in subtree]
            if not geoms:
                sizes[label] = {"error": "no collision geoms"}
                continue
            rotation = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
            origin = np.asarray(data.xpos[body], dtype=np.float64)
            points = []
            for g in geoms:
                box = np.asarray(model.geom_aabb[g], dtype=np.float64)
                local = box[:3] + signs * box[3:]
                world = np.asarray(data.geom_xpos[g], dtype=np.float64) + local @ np.asarray(
                    data.geom_xmat[g], dtype=np.float64).reshape(3, 3).T
                points.append((world - origin) @ rotation)
            points = np.concatenate(points)
            low, high = points.min(axis=0), points.max(axis=0)
            sizes[label] = {"half_extents_m": [float(v) for v in (high - low) / 2],
                            "center_offset_m": [float(v) for v in (high + low) / 2],
                            "frame": "object", "collision_geoms": len(geoms)}
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            sizes[label] = _error(exc)
    return {"objects": objects, "owner": owner, "robot": robot, "hand": hand, "names": names, "sizes": sizes}


def object_velocity(model, data, source, mj):
    """World-frame (linear m/s, angular rad/s) of a joint:/body: object. A free joint's qvel holds
    the linear velocity in the world frame and the angular one in the body frame (rotated here);
    a body uses mj_objectVelocity (flg_local=0: [angular, linear], at the body's centre of mass)."""
    import numpy as np

    kind, name = source.split(":", 1)
    if kind == "joint":
        joint = model.joint(name).id
        adr, body = int(model.jnt_dofadr[joint]), int(model.jnt_bodyid[joint])
        qvel = np.asarray(data.qvel[adr:adr + 6], dtype=np.float64)
        rotation = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
        return [float(v) for v in qvel[:3]], [float(v) for v in rotation @ qvel[3:6]]
    out = np.zeros(6, dtype=np.float64)
    mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, int(model.body(name).id), out, 0)
    return [float(v) for v in out[3:6]], [float(v) for v in out[0:3]]


def contacts(model, data, scene, mj, limit=MAX_CONTACT_PAIRS):
    """The current contacts that touch a task object or a hand, summed per pair of names (an object
    label, else the MuJoCo body name): total normal force (mj_contactForce), contact count, mean
    position. Robot-robot contacts and contacts inside one object are left out."""
    import numpy as np

    owner, robot, hand, names = scene["owner"], scene["robot"], scene["hand"], scene["names"]
    contact = data.contact
    pairs = {}
    force = np.zeros(6, dtype=np.float64)
    for i in range(int(data.ncon)):
        b1 = int(model.geom_bodyid[int(contact.geom1[i])])
        b2 = int(model.geom_bodyid[int(contact.geom2[i])])
        if b1 in robot and b2 in robot:
            continue
        if not (owner.get(b1) or b1 in hand or owner.get(b2) or b2 in hand):
            continue
        n1, n2 = owner.get(b1) or names[b1], owner.get(b2) or names[b2]
        if n1 == n2:
            continue
        # The task object first, then by name.
        (a, _), (b, _) = sorted(((n1, b1), (n2, b2)), key=lambda item: (owner.get(item[1]) is None, item[0]))
        mj.mj_contactForce(model, data, i, force)
        row = pairs.setdefault((a, b), {"a": a, "b": b, "normal_force_n": 0.0, "count": 0, "_pos": np.zeros(3)})
        row["normal_force_n"] += float(force[0])
        row["count"] += 1
        row["_pos"] += np.asarray(contact.pos[i], dtype=np.float64)
    rows = sorted(pairs.values(), key=lambda row: -row["normal_force_n"])[:limit]
    for row in rows:
        row["position_m"] = [float(v) for v in row.pop("_pos") / row["count"]]
    return rows


def object_states(data, env_name, dual_arm, *, level="objects", model=None, env=None, scene=None, mj=None):
    """Live world-frame state of the task's objects and fixtures and the arm joints, read-only
    from MuJoCo ``data``. A measurement that cannot be read is reported as an error, never raised.
    ``level="full"`` adds velocities, sizes (``scene``: scene_index at reset), contacts and the goal
    (``env``: the raw official environment); ``mj`` is the mujoco module (imported when omitted)."""
    objects = {}
    for label, source in state_sources(env_name).items():
        try:
            objects[label] = _state_entry(source, read_state(data, source))
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            objects[label] = {"error": f"{type(exc).__name__}: {exc}"}
    arms = {}
    for arm, names in arm_joint_names(dual_arm).items():
        try:
            arms[arm] = [float(data.joint(name).qpos[0]) for name in names]
        except Exception as exc:  # noqa: BLE001
            arms[arm] = {"error": f"{type(exc).__name__}: {exc}"}
    state = {"privileged_level": level, "frame": "world", "objects": objects, "arm_joint_positions_rad": arms}
    if level != "full":
        return state
    if mj is None:
        import mujoco as mj
    sources = state_sources(env_name)
    for label, source in sources.items():
        if source.split(":", 1)[0] not in BODY_KINDS or "error" in objects[label]:
            continue
        entry = objects[label]
        try:
            entry["linear_velocity_mps"], entry["angular_velocity_radps"] = object_velocity(model, data, source, mj)
        except Exception as exc:  # noqa: BLE001
            entry["velocity_error"] = _error(exc)["error"]
        entry["size"] = (scene or {}).get("sizes", {}).get(label) or {"error": "no size: the scene index is missing"}
    try:
        if not (scene and "owner" in scene):
            raise RuntimeError((scene or {}).get("error", "the scene index is missing"))
        state["contacts"] = contacts(model, data, scene, mj)
    except Exception as exc:  # noqa: BLE001
        state["contacts"] = _error(exc)
    from services.dexjoco.goals import goal
    state["goal"] = goal(env, data, env_name)
    return state


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
