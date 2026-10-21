"""Experiment (exp/add-state-info/lookahead): exact lookahead on real DexJoCo task models, CPU only.

Runs with the simulator interpreter (runtime/envs/dexjoco: mujoco 3.4.0 and the DexJoCo package)
and is skipped where those are missing (the host interpreter):

    PYTHONPATH=.:<pytest> runtime/envs/dexjoco/bin/python -m pytest -q -p no:cacheprovider \
        local_tests/unit/test_dexjoco_lookahead_physics.py

(the simulator environment has no pytest: <pytest> is a directory holding links to the host
environment's pytest, _pytest, pluggy, iniconfig and pygments packages and their dist-info).
RAX_LOOKAHEAD_CPU_EVIDENCE=<file.json> also writes the per-task comparison evidence.

The official gym environment classes are not constructed (they open an EGL renderer). Each task
runs its real MJCF model under a CPU stand-in for the official step: the same mocap targets, the
official operational-space controller with the official gains, 10 physics substeps per row, the
Allegro position targets, the env_step cap, ``image_obs`` and the ``hz`` pacing flag. It sits under
the official gym wrappers (``DexjocoObsAdapter``, ``Single/DualArmPolicyWrapper``) and the official
``DexJoCoOpenPIEnv`` wrapper, inside a real ``DexjocoSession`` driven through ``dispatch``.

Each test compares a session that looks ahead with a twin that never does: reset -> k real rows
-> lookaheads -> n real rows must be bit-identical (every compared array equal, max abs diff 0).
"""
from __future__ import annotations

import copy
import importlib
import json
import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("dexjoco")
gym = pytest.importorskip("gymnasium")

from services.dexjoco import lookahead as LA  # noqa: E402
from services.dexjoco import tasks as T  # noqa: E402
from services.dexjoco.objects import scene_index  # noqa: E402
from services.dexjoco.session import DexjocoSession, _Tap, hold_row  # noqa: E402
from services.simulator.io import RejectedBeforeAction  # noqa: E402

if not (T.source_root() / "configs").is_dir():
    pytest.skip("pinned DexJoCo checkout not installed", allow_module_level=True)

MODULES = {"water_plant": "panda_water_plant_env", "fold_glasses": "panda_fold_glasses_env",
           "bimanual_microwave_cook": "panda_bimanual_microwave_cook_env",
           "bimanual_hanoi": "panda_bimanual_hanoi_env"}
PRE_ROWS, LOOK_ROWS, REAL_ROWS = 30, 60, 40
EVIDENCE = os.environ.get("RAX_LOOKAHEAD_CPU_EVIDENCE")  # Optional JSON evidence file.


class CpuTaskEnv(gym.Env):
    """The official step's physics on the task's real model (see the module docstring)."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task, cameras, seed=0):
        from dexjoco.sim.controllers import opspace

        self._opspace = opspace
        module = importlib.import_module(f"dexjoco.sim.envs.{MODULES[task]}")
        self._home = np.asarray(module._PANDA_HOME, dtype=np.float64)
        self._model = mujoco.MjModel.from_xml_path(str(module._XML_PATH))
        self._model.opt.timestep = 0.002
        self._data = mujoco.MjData(self._model)
        self._n_substeps = 10
        self._random = np.random.RandomState(seed)
        self.dual = T.task_info(task)["dual_arm"]
        self.cap = T.task_info(task)["env_step_cap"]
        self.cameras = list(cameras)
        self.render_mode, self.image_obs, self.hz = "rgb_array", True, 30
        self.env_step, self.renders, self.paced = 0, 0, 0
        # The task's own step-changed attributes (lookahead.REQUIRED_ATTRS), updated every step
        # below so their restore is exercised too.
        self.required = [n for n in LA.REQUIRED_ATTRS[task] if n != "env_step"]
        for name in self.required:
            setattr(self, name, self._attr_value(name, 0))
        m = self._model
        suffixes = ("_right", "_left") if self.dual else ("",)
        self.arms = []
        for suffix in suffixes:
            dofs = np.array([m.joint(f"joint{i}{suffix}").id for i in range(1, 8)])
            ctrl = np.array([m.actuator(f"actuator{i}{suffix}").id for i in range(1, 8)])
            self.arms.append({"site": m.site(f"attachment_site{suffix}").id, "dofs": dofs, "ctrl": ctrl,
                              "pos": f"franka/flange_pos{suffix}", "quat": f"franka/flange_quat{suffix}"})
        # The hands' position actuators; the left hand's in its official mirrored order.
        fingers = {"": ("ff", "mf", "rf", "th"), "_right": ("ff", "mf", "rf", "th"), "_left": ("rf", "mf", "ff", "th")}
        groups = [[m.actuator(f"{f}a{i}{suffix}").id for f in fingers[suffix] for i in range(4)] for suffix in suffixes]
        for arm, group in zip(self.arms, groups):
            arm["hand_ctrl"] = np.array(group)
            arm["hand_qpos"] = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in group])
        # gym.spaces.Dict sorts its keys: the numbered names keep the official state order.
        self.state_keys = self._state_keys()
        keys = list(self.state_keys.values())
        sizes = {k: 3 if k.endswith("pos") else 4 if k.endswith("quat") else 16 for k in keys}
        box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), np.float64)  # noqa: E731
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({k: box(sizes[k]) for k in keys}),
            "images": gym.spaces.Dict({c: gym.spaces.Box(0, 255, (8, 8, 3), np.uint8) for c in self.cameras})})
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (46 if self.dual else 23,), np.float64)

    def reset(self, *, seed=None, options=None):
        m, d = self._model, self._data
        mujoco.mj_resetData(m, d)
        for arm in self.arms:
            d.qpos[arm["dofs"]] = self._home
        mujoco.mj_forward(m, d)
        for index, arm in enumerate(self.arms):
            d.mocap_pos[index] = d.sensor(arm["pos"]).data
            d.mocap_quat[index] = d.sensor(arm["quat"]).data
        mujoco.mj_forward(m, d)
        self.env_step = 0
        return self._observation(), {}

    def step(self, action):
        m, d = self._model, self._data
        parts = [np.asarray(action["right"]), np.asarray(action["left"])] if self.dual else [np.asarray(action)]
        for index, part in enumerate(parts):
            if not (np.allclose(part[:3], 0.0) and np.allclose(part[3:7], 0.0)):
                d.mocap_pos[index], d.mocap_quat[index] = part[:3], part[3:7]
        for _ in range(self._n_substeps):
            for index, (arm, part) in enumerate(zip(self.arms, parts)):
                d.ctrl[arm["ctrl"]] = self._opspace(
                    model=m, data=d, site_id=arm["site"], dof_ids=arm["dofs"], pos=d.mocap_pos[index],
                    ori=d.mocap_quat[index], joint=self._home, gravity_comp=True, pos_gains=(400.0, 400.0, 400.0),
                    damping_ratio=4)
                d.ctrl[arm["hand_ctrl"]] = part[7:23]
            mujoco.mj_step(m, d)
        obs = self._observation()
        self.env_step += 1
        for name in self.required:
            setattr(self, name, self._attr_value(name, self.env_step))
        terminated = self.env_step >= self.cap
        if 1.0 / self.hz > 0:
            self.paced += 1  # The official env would sleep here.
        return obs, 0.0, terminated, False, {"succeed": False}

    @staticmethod
    def _attr_value(name, step):
        if name == "_pending_illegal_reset":
            return False  # Set, it would stop the lookahead (host unit tests cover that path).
        if name == "_pending_illegal_info":
            return {"step": step, "history": [step, step + 1]}
        if name == "_pending_illegal_deadline":
            return 0.5 * step
        if name == "_illegal_trigger_reason":
            return f"reason {step}"
        if name in ("_trigger_pulled", "_latest_shutter_pressed"):
            return step % 7 == 3
        return step % 11  # The consecutive-step counters.

    def _state_keys(self):
        parts = [(arm, kind) for arm in range(len(self.arms)) for kind in ("pos", "quat")]
        parts += [(arm, "hand") for arm in range(len(self.arms))]
        return {part: f"s{i}_arm{part[0]}_{part[1]}" for i, part in enumerate(parts)}

    def _observation(self):
        d = self._data
        state = {}
        for (index, kind), key in self.state_keys.items():
            arm = self.arms[index]
            state[key] = (d.qpos[arm["hand_qpos"]] if kind == "hand" else d.sensor(arm[kind]).data).copy()
        obs = {"state": state}
        if self.image_obs:
            self.renders += 1
            shade = np.uint8(int(1000 * abs(d.qpos[0])) % 255)
            obs["images"] = {c: np.full((8, 8, 3), shade, np.uint8) for c in self.cameras}
        return obs


def make_session(tmp_path, task, name):
    from dexjoco.tasks.obs_adapters import DexjocoObsAdapter
    from dexjoco.tasks.policy_wrappers import DualArmPolicyWrapper, SingleArmPolicyWrapper
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

    session = DexjocoSession(tmp_path / name, task, 0, include_depth=False)
    info, config = session.info, session.config
    raw = CpuTaskEnv(task, config["camera_mapping"].values())
    policy = (DualArmPolicyWrapper if info["dual_arm"] else SingleArmPolicyWrapper)(raw)
    tap = _Tap(DexjocoObsAdapter(policy))
    env = DexJoCoOpenPIEnv(env_name=info["env_name"], camera_mapping=config["camera_mapping"], seed=0,
                           rand_full=False, randomize_dynamics=False, dual_arm=info["dual_arm"],
                           prompt=config["prompt"], render_mode="rgb_array")
    env.env = tap
    session.env, session.tap = env, tap
    env.reset()
    session.layout_id, session.layout = 0, T.layout(task, 0, 0)
    session.scene = scene_index(raw._model, raw._data, info["env_name"])
    session.episode_id = f"{task}_cpu_{name}"
    session.episode_dir = session.output / session.episode_id
    session.episode_dir.mkdir(parents=True)
    session.obs = session._agent_observation(depth=False)
    return session


def call(session, op, **args):
    return session.dispatch(op, {"episode_id": session.episode_id, "step_id": session.step_id, **args})


def plan(session, rows, seed, drop):
    """Rows that start from the measured hold row, move each flange down by ``drop`` metres in
    total and close the fingers part way, with a seeded jitter."""
    rng = np.random.default_rng(seed)
    base = np.asarray(hold_row(session.env.obs["state"], session.info["dual_arm"]))
    out = []
    for i in range(1, rows + 1):
        row = base.copy()
        for start in ((0, 22) if session.info["dual_arm"] else (0,)):
            row[start:start + 3] += [0.3 * drop * i / rows, 0.0, -drop * i / rows]
            row[start:start + 3] += rng.normal(0.0, 0.002, 3)
            row[start + 6:start + 22] = np.clip(row[start + 6:start + 22] + rng.uniform(0, 0.6, 16) * i / rows,
                                                -0.2, 1.5)
        out.append(row)
    return np.asarray(out)


def live_record(session):
    """Everything compared per row: the MjData state and derived fields, the contacts, the raw
    environment's counters and the session's view."""
    raw = session.tap.unwrapped
    d = raw._data
    fields = ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "mocap_pos", "mocap_quat", "sensordata", "xpos",
              "xquat", "site_xpos", "site_xmat", "qM", "qfrc_bias", "cdof", "subtree_com", "cacc", "cfrc_int")
    record = {f: np.array(getattr(d, f), copy=True) for f in fields}
    record["time"] = np.array([d.time])
    record["ncon"] = np.array([d.ncon])
    record["contact_pos"] = np.array(d.contact.pos, copy=True)
    record["contact_dist"] = np.array(d.contact.dist, copy=True)
    record["contact_geom"] = np.array(d.contact.geom, copy=True)
    record["efc_force"] = np.array(d.efc_force, copy=True)
    record["env_step"] = np.array([raw.env_step])
    record["state"] = np.asarray(session.obs["state"], dtype=np.float64)
    return record


def compare(a, b):
    """(identical, max abs diff) over every record pair."""
    assert len(a) == len(b)
    identical, worst = True, 0.0
    for left, right in zip(a, b):
        assert left.keys() == right.keys()
        for key in left:
            x, y = left[key], right[key]
            same = x.shape == y.shape and np.array_equal(x, y)
            identical &= bool(same)
            if x.shape == y.shape and x.size:
                worst = max(worst, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
            elif x.shape != y.shape:
                worst = float("inf")
    return identical, worst


def run_rows(session, rows):
    records = []
    for row in rows:
        call(session, "chunk_step", actions=row[None, :])
        records.append(live_record(session))
    return records


def strip_timing(value):
    return {k: v for k, v in value.items() if k not in ("elapsed_s", "rollout_s", "rollout_s_per_row")}


def _evidence(entry):
    if not EVIDENCE:
        return
    path = os.path.abspath(EVIDENCE)
    data = json.loads(open(path).read()) if os.path.exists(path) else {}
    data[entry["task"]] = entry
    with open(path, "w") as stream:
        json.dump(data, stream, indent=2)


@pytest.mark.parametrize("task", list(MODULES))
def test_continuation_after_lookahead_is_bit_identical(tmp_path, task, monkeypatch):
    monkeypatch.setenv("RAX_PRIVILEGED_STATE", "full")
    twin, looker = make_session(tmp_path, task, "twin"), make_session(tmp_path, task, "look")
    pre = plan(twin, PRE_ROWS, 1, 0.15)
    run_rows(twin, pre)
    run_rows(looker, pre)
    assert compare([live_record(twin)], [live_record(looker)]) == (True, 0.0)

    real = plan(twin, REAL_ROWS, 2, 0.12)
    other = plan(looker, LOOK_ROWS, 3, 0.45)  # Different rows: deeper, toward contact.
    raw = looker.tap.unwrapped
    before_record = live_record(looker)
    before_states = call(looker, "object_states")
    before_observe = call(looker, "observe")
    files = sorted(p for p in looker.output.rglob("*") if p.is_file())
    mtimes = {p: p.stat().st_mtime_ns for p in files}
    renders, paced = raw.renders, raw.paced

    first = call(looker, "lookahead", actions=other, report_every=10)
    assert first["rows_simulated"] == LOOK_ROWS and first["stop_reason"] is None
    assert len(first["reports"]) == LOOK_ROWS // 10 - 1 and first["live_state_restored"] is True
    ncon_during = [len(r["object_states"]["contacts"]) for r in first["reports"]]
    predicted = call(looker, "lookahead", actions=real)
    assert raw.renders == renders and raw.paced == paced  # No render or pacing during a lookahead.
    assert raw.image_obs is True and raw.hz == 30
    # The live view after the lookaheads equals the view before them.
    assert compare([before_record], [live_record(looker)]) == (True, 0.0)
    assert call(looker, "object_states") == before_states
    assert call(looker, "observe") is before_observe
    assert {p: p.stat().st_mtime_ns for p in files} == mtimes
    assert looker.step_id == PRE_ROWS and looker._video_frames == twin._video_frames

    twin_records, look_records = run_rows(twin, real), run_rows(looker, real)
    identical, worst = compare(twin_records, look_records)
    assert identical and worst == 0.0, (task, worst)
    # The prediction equals the real continuation, state vector and object states.
    assert predicted["final"]["state"] == np.asarray(twin.obs["state"]).tolist()
    twin_states = {k: v for k, v in call(twin, "object_states").items() if k not in ("episode_id", "step_id")}
    assert predicted["final"]["object_states"] == twin_states
    assert looker.lookahead_usage["calls"] == 2 and looker.lookahead_usage["rows"] == LOOK_ROWS + REAL_ROWS
    assert predicted["lookahead_budget"]["rows_left"] == 3 * looker.step_limit - LOOK_ROWS - REAL_ROWS
    _evidence({"task": task, "dual_arm": looker.info["dual_arm"], "pre_rows": PRE_ROWS,
               "lookahead_rows": [LOOK_ROWS, REAL_ROWS], "real_rows_after": REAL_ROWS,
               "compared_rows": len(look_records), "compared_fields": sorted(look_records[0]),
               "bit_identical": identical, "max_abs_diff": worst,
               "prediction_equals_real": True, "contact_pairs_before": len(before_states["contacts"]),
               "contact_pairs_during_lookahead": ncon_during,
               "contacts_after_real_rows": int(look_records[-1]["ncon"][0]),
               "rollout_s_per_row": [first["rollout_s_per_row"], predicted["rollout_s_per_row"]]})


@pytest.mark.parametrize("task", ["water_plant", "bimanual_microwave_cook"])
def test_lookahead_to_the_episode_end_does_not_end_it(tmp_path, task, monkeypatch):
    monkeypatch.setenv("RAX_PRIVILEGED_STATE", "full")
    session = make_session(tmp_path, task, "end")
    run_rows(session, plan(session, 5, 1, 0.05))
    raw = session.tap.unwrapped
    raw.env_step = raw.cap - 15  # As if the episode had run to 15 steps before its cap.
    rows = plan(session, 40, 4, 0.05)
    reply = call(session, "lookahead", actions=rows)
    assert reply["stop_reason"] == "episode_end" and reply["end_row"] == 15 and reply["rows_simulated"] == 15
    assert reply["end_cause"] == "step_cap"
    assert not session.terminated and raw.env_step == raw.cap - 15 and session.step_id == 5
    result = call(session, "chunk_step", actions=rows)
    assert len(result["steps"]) == 15 and session.terminated
    assert result["frames"][-1]["state"] == reply["final"]["state"]
    with pytest.raises(RejectedBeforeAction, match="ended"):
        call(session, "lookahead", actions=rows[:1])


def test_set_state_alone_is_not_exact_on_a_task_model(tmp_path):
    """Negative control (DESIGN.md 7.2): restoring only mj_setState(INTEGRATION) changes the next
    rows on a real model under the official controller, so the twin comparison can see the bug."""
    twin, broken = make_session(tmp_path, "bimanual_microwave_cook", "twin"), make_session(
        tmp_path, "bimanual_microwave_cook", "broken")
    pre = plan(twin, PRE_ROWS, 1, 0.15)
    run_rows(twin, pre)
    run_rows(broken, pre)
    raw = broken.tap.unwrapped
    m, d = raw._model, raw._data
    sig = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(m, sig))
    mujoco.mj_getState(m, d, state, sig)
    env_step = raw.env_step
    for row in plan(broken, 20, 3, 0.45):
        broken.tap._env.step(broken.env._process_action(row))
    mujoco.mj_setState(m, d, state, sig)
    raw.env_step = env_step
    real = plan(twin, 10, 2, 0.12)
    identical, worst = compare(run_rows(twin, real), run_rows(broken, real))
    assert not identical and worst > 1e-9


TOY = """<mujoco><option noslip_iterations='5'/><worldbody><geom type='plane' size='2 2 .1'/>
<body mocap='true' pos='0 0 1'><geom type='sphere' size='.02' contype='0' conaffinity='0'/></body>
<body pos='0 0 .1'><joint name='s' type='slide' axis='1 0 0'/><geom type='box' size='.05 .05 .05'/></body>
""" + "".join(f"<body pos='0.3 0 {0.06 + 0.11 * i}'><freejoint/><geom type='box' size='.05 .05 .05'/></body>"
              for i in range(4)) + "</worldbody><actuator><motor joint='s' gear='1'/></actuator></mujoco>"


def toy_rows(m, d, n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        d.ctrl[:] = rng.uniform(-5, 5, m.nu)
        for _ in range(10):
            mujoco.mj_step(m, d)


def test_in_place_restore_cannot_restore_the_arena_layout():
    """Why the tool runs on a copy of the live MjData (lookahead.py docstring): an in-place restore
    of every field (DESIGN.md 1.1) gives the same next trajectory, but when the contact count
    changed during the lookahead the constraint arrays stay aliased differently, so readers before
    the next step (contact forces) see other values; the exactness fingerprint catches it. The
    copy leaves the live MjData untouched."""
    m = mujoco.MjModel.from_xml_string(TOY)
    reference, d = mujoco.MjData(m), mujoco.MjData(m)
    toy_rows(m, reference, 30, 1)
    toy_rows(m, d, 30, 1)
    before = LA.mjdata_fingerprint(m, d)
    # Twins agree except on the wall-clock timer statistics.
    assert LA.differing(before, LA.mjdata_fingerprint(m, reference)) == ["mj.timer"]
    # The tool's way: the rows run on a copy.
    scratch = LA.copy_mjdata(d)
    rng = np.random.default_rng(5)
    for _ in range(25):
        scratch.xfrc_applied[3:, :3] = rng.uniform(-50, 50, (m.nbody - 3, 3))
        for _ in range(10):
            mujoco.mj_step(m, scratch)
    assert scratch.ncon != d.ncon and LA.mjdata_fingerprint(m, d) == before
    # The in-place way: restore every field into the stepped object.
    snap = copy.copy(d)
    stepped = LA.copy_mjdata(scratch)
    LA.restore_mjdata(m, stepped, snap)
    diff = LA.differing(before, LA.mjdata_fingerprint(m, stepped))
    arena = ("mj.efc_", "mj.iefc_", "mj.i", "mj.dof_island", "mj.tree_island", "mj.map_")
    assert any(k.startswith("mj.efc_") for k in diff) and all(k.startswith(arena) for k in diff), diff
    toy_rows(m, reference, 20, 2)
    toy_rows(m, stepped, 20, 2)
    assert np.array_equal(reference.qpos, stepped.qpos) and np.array_equal(reference.qvel, stepped.qvel)
