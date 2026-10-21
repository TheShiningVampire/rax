"""Experiment (exp/add-state-info/lookahead): dexjoco_lookahead with fakes (no MuJoCo, no EGL).

The snapshot/restore/fingerprint logic of services/dexjoco/lookahead.py and the session op run
against a fake raw environment whose step changes its MjData stand-in, its counters and the
global RNG and counts renders and pacing; the MjData part of the fingerprint is replaced by one
over the fake's arrays. The real-physics counterpart is test_dexjoco_lookahead_physics.py.
"""
import ast
import json
import os
import random
import signal
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from services.benchmarks import BENCHMARKS
from services.dexjoco import lookahead as LA
from services.dexjoco import tasks as TK
from services.dexjoco import tools as T
from services.mcp_contract import Contract
from services.mcp_reply import _TASK_EVALUATION_KEYS, _agent_safe
from services.simulator.io import RejectedBeforeAction

from local_tests.unit.test_dexjoco_backend import FakeProcess, backend_for

needs_source = pytest.mark.skipif(not (TK.source_root() / "configs").is_dir(),
                                  reason="pinned DexJoCo checkout not installed (scripts/bootstrap_sources.sh)")


@pytest.fixture(autouse=True)
def default_settings(monkeypatch):
    for name in (*LA.KEPT_ENV, "RAX_PRIVILEGED_STATE"):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------------ settings
@pytest.mark.parametrize("value, enabled", [(None, True), ("on", True), ("off", False)])
def test_the_switch_defaults_on(value, enabled):
    environ = {} if value is None else {"RAX_LOOKAHEAD": value}
    assert LA.lookahead_enabled(environ) is enabled


@pytest.mark.parametrize("value", ["", "1", "true", "ON"])
def test_an_invalid_switch_fails_fast(value):
    with pytest.raises(ValueError, match="RAX_LOOKAHEAD"):
        LA.lookahead_setting({"RAX_LOOKAHEAD": value})


def test_the_row_budget_is_a_multiple_of_the_step_limit():
    assert LA.rows_budget(1000, {}) == 3000
    assert LA.rows_budget(TK.native_step_limit("bimanual_hanoi"), {}) == 4500
    assert LA.rows_budget(1100, {"RAX_LOOKAHEAD_ROWS_FACTOR": "2"}) == 2200
    with pytest.raises(ValueError):
        LA.rows_factor({"RAX_LOOKAHEAD_ROWS_FACTOR": "0"})
    assert LA.limits(1000, {}) == {"rows_per_episode": 3000, "calls_per_episode": 300, "seconds_per_call": 30.0}


# ------------------------------------------------------- attribute snapshot
class Bomb:
    """A renderer stand-in: copying it would tear down a GL context."""

    def __deepcopy__(self, memo):
        raise AssertionError("a viewer was deep-copied")

    def __copy__(self):
        raise AssertionError("a viewer was copied")


class FakeMjData:
    def __init__(self):
        self.qpos = np.zeros(4)

    def __copy__(self):  # As MjData: copy.copy is a full copy (mj_copyData).
        other = FakeMjData()
        other.qpos = self.qpos.copy()
        return other


def test_attribute_snapshot_restores_values_identity_and_aliases():
    env = SimpleNamespace()
    env.counter, env.latch, env.reason = 3, False, "none"
    env.state = np.arange(4.0)
    env.alias = env.state  # e.g. last_stay_state aliasing obs["state"]
    env.history = {"digits": [1, 2]}
    env.seen = {"a"}
    env.deleted_later = 7
    env._data, env._viewer, env.callback = FakeMjData(), Bomb(), (lambda: 1)
    env.helper = object()
    snap = LA.snapshot_attrs("raw", env)
    before = LA.attrs_fingerprint([snap])
    assert snap.kind("_viewer") == "excluded" and snap.kind("callback") == "excluded"
    assert snap.kind("helper") == "other" and snap.other == {"helper": "object"}
    assert snap.kind("history") == "container" and snap.kind("state") == "array"
    data, viewer, state, history, seen = env._data, env._viewer, env.state, env.history, env.seen

    env.counter += 1
    env.latch = True
    env.state[0] = 99.0
    env.history["digits"].append(3)
    env.history["new"] = []
    env.seen.add("b")
    env.added = "x"
    del env.deleted_later
    env._data = FakeMjData()
    env.state = np.zeros(2)  # rebound: the original array object must come back
    assert LA.attrs_fingerprint([snap]) != before

    LA.restore_attrs(snap)
    assert LA.attrs_fingerprint([snap]) == before
    assert (env.counter, env.latch, env.deleted_later) == (3, False, 7) and not hasattr(env, "added")
    assert env.state is state and env.alias is state and env.state.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert env.history is history and history == {"digits": [1, 2]} and env.seen is seen and seen == {"a"}
    assert env._data is data and env._viewer is viewer


def test_digest_is_canonical_and_sensitive():
    assert LA.digest({"b": [1, np.zeros(2)], "a": {2, 1}}) == LA.digest({"a": {1, 2}, "b": [1, np.zeros(2)]})
    assert LA.digest(np.zeros(2)) != LA.digest(np.zeros(2, dtype=np.float32))
    assert LA.digest(0.0) != LA.digest(-0.0) and LA.digest(1) != LA.digest(True)


# ------------------------------------------------------ official-source scan
STEP_ONLY_EXCLUSIONS = {
    # Reached only through reset() (hanoi's pending illegal auto-reset, refused by the guard) or
    # render paths that are off during a lookahead (image_obs False, render_mode rgb_array).
    "reset", "render", "_render_frames", "_compute_observation_images", "_prime_rgb_array_renderer",
    "_update_camera_screen_texture", "_set_offscreen_geom_group_visibility", "_set_human_geom_group_visibility",
}


def _step_writes(path, cls_name):
    """``self.X = ...`` and ``self._model.F[...] = ...`` targets in step() and the methods it calls."""
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    seen, todo, attrs, model = set(), ["step"], set(), set()
    while todo:
        name = todo.pop()
        if name in seen or name not in methods or name in STEP_ONLY_EXCLUSIONS:
            continue
        seen.add(name)
        for node in ast.walk(methods[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                todo.append(node.func.attr)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] \
                if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
            for target in targets:
                for t in (target.elts if isinstance(target, ast.Tuple) else [target]):
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                        attrs.add(t.attr)
                    base = t.value if isinstance(t, ast.Subscript) else None
                    if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Attribute) and \
                            base.value.attr in ("_model", "model"):
                        model.add(base.attr)
    return attrs, model, seen


TASK_CLASSES = {
    "water_plant": ("panda_water_plant_env.py", "PandaWaterPlantGymEnv"),
    "fold_glasses": ("panda_fold_glasses_env.py", "PandaFoldGlassesGymEnv"),
    "bimanual_microwave_cook": ("panda_bimanual_microwave_cook_env.py", "PandaBimanualMicrowaveCookGymEnv"),
    "bimanual_photograph": ("panda_bimanual_photograph_env.py", "PandaBimanualPhotographGymEnv"),
    "bimanual_hanoi": ("panda_bimanual_hanoi_env.py", "PandaBimanualHanoiGymEnv"),
}


@needs_source
@pytest.mark.parametrize("task", list(TASK_CLASSES))
def test_required_attrs_cover_the_official_step(task):
    filename, cls = TASK_CLASSES[task]
    path = TK.source_root() / "dexjoco" / "dexjoco" / "sim" / "envs" / filename
    attrs, model, methods = _step_writes(path, cls)
    assert "step" in methods
    assert attrs <= set(LA.REQUIRED_ATTRS[task]), sorted(attrs - set(LA.REQUIRED_ATTRS[task]))
    assert model <= set(LA.MODEL_FIELDS.get(task, ())), sorted(model)
    # Each required attribute is a plain value the generic snapshot copies.
    for name in LA.REQUIRED_ATTRS[task]:
        assert LA.classify(name, 0) == "value" and LA.classify(name, {"a": [1]}) == "container"


# ----------------------------------------------------------- session fakes
CAP = 40


class FakeRaw:
    """A raw environment: step moves its MjData stand-in, its counters and the global RNG, and
    counts renders and pacing."""

    def __init__(self, cap=CAP):
        self._model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002), geom_rgba=np.ones((3, 4)),
                                      geom_group=np.zeros(3, np.int32), body_mass=np.ones(3))
        self._data = FakeMjData()
        self._viewer, self._random = Bomb(), np.random.RandomState(0)
        self.render_mode, self.image_obs, self.hz = "rgb_array", True, 30
        self.env_step, self.cap = 0, cap
        self._trigger_pulled, self._cone_alpha_counter, self._success_counter = False, 0, 0
        self.renders = self.paced = 0
        self.fail_at = self.succeed_at = self.illegal_at = None
        self.cached = None  # A stale reference to the live data, written by a buggy step.
        self.reset_trigger = False

    @property
    def unwrapped(self):
        return self

    def step(self, action):
        if self.fail_at is not None and self.env_step + 1 >= self.fail_at:
            raise RuntimeError("physics went unstable")
        self._data.qpos += np.asarray(action[:4]) * 0.1
        if self.cached is not None:
            self.cached.qpos[0] += 1.0
        self.env_step += 1
        self._success_counter += 1
        self._trigger_pulled = not self._trigger_pulled
        self._model.geom_rgba[0, 3] = 0.0  # water_plant's cone alpha (MODEL_FIELDS)
        np.random.random()
        random.random()
        if self.image_obs:
            self.renders += 1
        if 1.0 / self.hz > 0:
            self.paced += 1
        if self.illegal_at is not None and self.env_step >= self.illegal_at:
            self._pending_illegal_reset, self._illegal_trigger_reason = True, "disk on the table"
        success = self.succeed_at is not None and self.env_step >= self.succeed_at
        obs = {"state": np.concatenate([self._data.qpos, np.zeros(19)])}
        if self.image_obs:
            obs.update(front=np.zeros((4, 4, 3), np.uint8), wrist=np.zeros((4, 4, 3), np.uint8))
        return obs, 0.0, self.env_step >= self.cap or success, False, {"succeed": success}


class FakeAdapter:
    def __init__(self, raw):
        self.env = raw

    @property
    def unwrapped(self):
        return self.env

    def step(self, action):
        return self.env.step(action)


class FakeOfficialWrapper:
    """DexJoCoOpenPIEnv's interface: _process_action, step, obs, raw images, is_done."""

    def __init__(self, tap, cameras):
        self.env, self.cameras = tap, cameras
        self._done = self._success = False
        self.obs = {"state": np.zeros(23)}
        self._raw_obs = {c: np.zeros((4, 4, 3), np.uint8) for c in cameras}

    def _process_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        return np.concatenate([action[:3], [1.0, 0.0, 0.0, 0.0], action[6:]])

    def step(self, action):
        obs, _, terminated, _, info = self.env.step(self._process_action(action))
        self._done, self._success = bool(terminated), bool(info.get("succeed"))
        self._raw_obs = {c: np.zeros((4, 4, 3), np.uint8) for c in self.cameras}
        self.obs = {"state": obs["state"][:23]}

    def get_raw_images(self):
        return self._raw_obs

    @property
    def is_done(self):
        return self._done

    @property
    def is_success(self):
        return self._success


def fake_mjdata_fingerprint(model, data):
    return {"mj.qpos": LA.digest(data.qpos)}


def fake_object_states(data, env_name, dual_arm, *, level="objects", **_):
    return {"privileged_level": level, "objects": {"thing": {"position_m": data.qpos[:3].tolist()}}}


@pytest.fixture
def session_for(tmp_path, monkeypatch):
    from services.dexjoco import session as S

    monkeypatch.setattr(LA, "mjdata_fingerprint", fake_mjdata_fingerprint)
    monkeypatch.setattr(S, "object_states", fake_object_states)

    def make(level="full", lookahead="on", cap=CAP):
        monkeypatch.setenv("RAX_PRIVILEGED_STATE", level)
        monkeypatch.setenv("RAX_LOOKAHEAD", lookahead)
        session = S.DexjocoSession(tmp_path / f"sim-{level}-{lookahead}-{cap}", "water_plant", 0, include_depth=False)
        session.info["env_step_cap"] = cap  # The fake's own cap.
        raw = FakeRaw(cap)
        tap = S._Tap(FakeAdapter(raw))
        session.env, session.tap = FakeOfficialWrapper(tap, session.cameras), tap
        tap.last_obs, tap.last_info = {"state": np.zeros(23)}, {}
        session.layout_id, session.layout = 0, TK.layout("water_plant", 0, 0)
        session.episode_id = "water_plant_exploration_0_test"
        session.episode_dir = session.output / session.episode_id
        session.episode_dir.mkdir(parents=True)
        session.obs = session._agent_observation(depth=False)
        session._write_operator_state()
        return session, raw

    return make


def call(session, op, **args):
    return session.dispatch(op, {"episode_id": session.episode_id, "step_id": session.step_id, **args})


def rows(n, value=0.5):
    return np.full((n, 22), value)


@needs_source
def test_lookahead_leaves_the_episode_untouched(session_for):
    session, raw = session_for()
    call(session, "chunk_step", actions=rows(3))
    live, qpos = raw._data, raw._data.qpos.copy()
    obs, frames, step_id = session.obs, session._video_frames, session.step_id
    tap_obs, wrapper_obs, ever_true = session.tap.last_obs, session.env.obs, set(session.tap.ever_true)
    files = sorted(p for p in session.output.rglob("*") if p.is_file())
    mtimes = {p: p.stat().st_mtime_ns for p in files}
    rng, prng = np.random.get_state()[1].copy(), random.getstate()
    counters = (raw.env_step, raw._success_counter, raw._trigger_pulled, raw.renders, raw.paced)
    alpha = raw._model.geom_rgba.copy()

    reply = call(session, "lookahead", actions=rows(12, 1.0), report_every=5)
    assert reply["rows_requested"] == reply["rows_simulated"] == 12 and reply["stop_reason"] is None
    assert [r["row"] for r in reply["reports"]] == [5, 10] and reply["final"]["row"] == 12
    # The reports show the lookahead's state; the live state is the one before it.
    assert reply["final"]["object_states"]["objects"]["thing"]["position_m"] == pytest.approx(
        (qpos[:3] + 12 * 0.1).tolist())
    assert reply["final"]["state"][:4] == pytest.approx((qpos + 1.2).tolist())
    assert raw._data is live and raw._data.qpos.tolist() == qpos.tolist()
    assert (raw.env_step, raw._success_counter, raw._trigger_pulled, raw.renders, raw.paced) == counters
    assert np.array_equal(raw._model.geom_rgba, alpha) and raw.image_obs is True and raw.hz == 30
    assert np.array_equal(np.random.get_state()[1], rng) and random.getstate() == prng
    assert session.obs is obs and session._video_frames == frames and session.step_id == step_id == 3
    assert session.tap.last_obs is tap_obs and session.env.obs is wrapper_obs and session.tap.ever_true == ever_true
    assert not (session.terminated or session.truncated or session.success or session.invalid_reason)
    assert {p: p.stat().st_mtime_ns for p in files} == mtimes
    assert reply["live_state_restored"] is True and reply["end_cause"] is None
    assert reply["lookahead_budget"] == {"rows_used": 12, "rows_left": 3 * session.step_limit - 12,
                                         "calls_used": 1, "calls_left": 299, "seconds_per_call": 30.0}
    # A real step afterwards continues from the live state.
    call(session, "chunk_step", actions=rows(1))
    assert raw._data.qpos[:3].tolist() == pytest.approx((qpos[:3] + 0.05).tolist()) and session.step_id == 4


@needs_source
def test_the_reply_survives_the_agent_filter(session_for):
    session, _ = session_for()
    reply = json.loads(json.dumps(call(session, "lookahead", actions=rows(10), report_every=5)))
    assert _agent_safe(reply) == reply

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    found = set(keys(reply))
    assert not found & (_TASK_EVALUATION_KEYS | {"done", "succeed", "episode_ended", "task_complete"})
    assert not any(k.startswith("native_") for k in found)
    assert {"episode_id", "step_id", "rows_requested", "rows_simulated", "stop_reason", "end_row", "end_cause",
            "reports", "final", "live_state_restored", "lookahead_budget", "elapsed_s"} <= found


@needs_source
@pytest.mark.parametrize("level", ["full", "objects"])
def test_the_episode_end_is_predicted_not_taken(session_for, level):
    session, raw = session_for(level=level)
    call(session, "chunk_step", actions=rows(CAP - 10))
    reply = call(session, "lookahead", actions=rows(30))
    assert reply["stop_reason"] == "episode_end" and reply["end_row"] == 10 and reply["rows_simulated"] == 10
    assert session.lookahead_usage["rows"] == 10  # Charged for the rows simulated.
    if level == "full":
        assert reply["end_cause"] == "step_cap"
    else:
        assert "end_cause" not in reply  # It would disclose success.
    assert not session.terminated and raw.env_step == CAP - 10
    result = call(session, "chunk_step", actions=rows(30))
    assert len(result["steps"]) == 10 and session.terminated
    with pytest.raises(RejectedBeforeAction, match="ended"):
        call(session, "lookahead", actions=rows(1))


@needs_source
def test_a_task_condition_end_and_an_illegal_move(session_for):
    session, raw = session_for()
    raw.succeed_at = 4
    reply = call(session, "lookahead", actions=rows(20))
    assert (reply["stop_reason"], reply["end_row"], reply["end_cause"]) == ("episode_end", 4, "task_condition")
    raw.succeed_at, raw.illegal_at, raw._pending_illegal_reset = None, 3, False
    reply = call(session, "lookahead", actions=rows(20))
    assert (reply["stop_reason"], reply["end_row"], reply["illegal_reason"]) == ("illegal_move", 3, "disk on the table")
    assert raw._pending_illegal_reset is False and not hasattr(raw, "_illegal_trigger_reason")


@needs_source
def test_rendering_and_pacing_come_back_when_a_row_raises(session_for):
    session, raw = session_for()
    live, qpos = raw._data, raw._data.qpos.copy()
    raw.fail_at = 3
    with pytest.raises(RuntimeError, match="unstable"):
        call(session, "lookahead", actions=rows(10))
    assert raw.image_obs is True and raw.hz == 30 and raw._data is live and raw._data.qpos.tolist() == qpos.tolist()
    assert raw.env_step == 0 and session.invalid_reason is None
    assert session.lookahead_usage["calls"] == 1 and session.lookahead_usage["rows"] == 2


@needs_source
def test_a_restore_mismatch_invalidates_the_episode(session_for):
    session, raw = session_for()
    raw.cached = raw._data  # The step writes the live MjData behind the lookahead's back.
    with pytest.raises(LA.LookaheadRestoreError, match="mj.qpos"):
        call(session, "lookahead", actions=rows(5))
    assert session.invalid_reason == "lookahead_restore_mismatch" and session.truncated
    outcome = json.loads((session.output / "evaluation_outcome.json").read_text())
    assert outcome["valid_for_score"] is False and outcome["lookahead"]["calls"] == 1
    with pytest.raises(RejectedBeforeAction, match="ended"):
        call(session, "lookahead", actions=rows(1))


@needs_source
def test_guards_and_budgets_reject_before_anything_runs(session_for, monkeypatch):
    session, raw = session_for()
    state = raw._data.qpos.copy()
    cases = [
        (lambda: setattr(raw, "reset_trigger", True), "pending reset"),
        (lambda: setattr(raw, "_pending_illegal_reset", True), "illegal move"),
        (lambda: setattr(raw, "_camera_screen_effect", True), "screen effect"),
        (lambda: setattr(raw, "render_mode", "human"), "window"),
    ]
    for arm, message in cases:
        saved = dict(vars(raw))
        arm()
        with pytest.raises(RejectedBeforeAction, match=message):
            call(session, "lookahead", actions=rows(2))
        vars(raw).clear()
        vars(raw).update(saved)
    for kwargs, message in [({"actions": rows(2), "report_every": 3}, "report_every"),
                            ({"actions": np.zeros((2, 44))}, "22 numbers"),
                            ({"actions": rows(151)}, "1..150"),
                            ({"actions": np.full((2, 22), np.nan)}, "finite")]:
        with pytest.raises(RejectedBeforeAction, match=message):
            call(session, "lookahead", **kwargs)
    assert session.lookahead_usage == {"calls": 0, "rows": 0, "seconds": 0.0, "rejected": 8}
    assert raw._data.qpos.tolist() == state.tolist() and raw.renders == 0
    # Budgets: rows, then calls.
    session.lookahead_usage["rows"] = 3 * session.step_limit - 5
    with pytest.raises(RejectedBeforeAction, match="5 rows left"):
        call(session, "lookahead", actions=rows(6))
    call(session, "lookahead", actions=rows(5))
    session.lookahead_usage.update(rows=0, calls=LA.CALLS_PER_EPISODE)
    with pytest.raises(RejectedBeforeAction, match="300 calls"):
        call(session, "lookahead", actions=rows(1))


@needs_source
def test_switched_off_the_op_rejects(session_for):
    session, raw = session_for(lookahead="off")
    with pytest.raises(RejectedBeforeAction, match="RAX_LOOKAHEAD=off"):
        call(session, "lookahead", actions=rows(2))
    assert raw.env_step == 0 and session.metadata["lookahead"]["enabled"] is False


@needs_source
def test_operator_files_record_lookahead_use(session_for):
    session, _ = session_for()
    assert session.metadata["lookahead"] == {"enabled": True, "rows_per_episode": 3000, "calls_per_episode": 300,
                                             "seconds_per_call": 30.0}
    call(session, "lookahead", actions=rows(7))
    call(session, "chunk_step", actions=rows(1))
    state = json.loads((session.output / "operator_state.json").read_text())
    assert state["lookahead"]["calls"] == 1 and state["lookahead"]["rows"] == 7 and state["lookahead"]["enabled"]
    assert state["lookahead"]["seconds"] >= 0 and state["lookahead"]["rows_per_episode"] == 3000
    session.finish("done")
    for name in ("summary.json", "evaluation_outcome.json"):
        assert json.loads((session.output / name).read_text())["lookahead"]["rows"] == 7


def test_the_wall_clock_alarm_waits_for_the_restore():
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers live in the main thread")
    fired = []

    def expire(signum, frame):
        fired.append(signum)
        raise TimeoutError("episode wall clock")

    previous = signal.signal(signal.SIGALRM, expire)
    try:
        with pytest.raises(TimeoutError):
            with LA.deferred_signals():
                os.kill(os.getpid(), signal.SIGALRM)
                for _ in range(1000):  # Python runs the handler between bytecodes.
                    pass
                assert fired == []
        assert fired == [signal.SIGALRM]
    finally:
        signal.signal(signal.SIGALRM, previous)


# ------------------------------------------------------------- host wiring
def test_the_tool_is_registered_and_not_mutating():
    bench = BENCHMARKS["dexjoco"]
    assert T.LOOKAHEAD == LA.TOOL in bench.robot_tools and T.LOOKAHEAD not in bench.mutating
    assert T.LOOKAHEAD in Contract(benchmark="dexjoco").robot_names
    assert T.LOOKAHEAD in Contract(observation_profile="rgb-only", benchmark="dexjoco").robot_names
    assert T.LOOKAHEAD in {t["name"] for t in bench.definitions()}
    text = T.lookahead_description("full")
    for words in ("never changes the episode", "compare candidate rows", "execute the chosen rows with dexjoco_step",
                  "wall-clock time", "lookahead_budget", "3x the episode step limit", "300 calls", "30 s per call",
                  "end_cause"):
        assert words in text, words
    assert "end_cause" not in T.lookahead_description("objects") and "goal" not in T.lookahead_description("objects")


def test_requests_are_validated_before_dispatch():
    T.validate_request(T.LOOKAHEAD, {"actions": [[0.0] * 22] * 3})
    T.validate_request(T.LOOKAHEAD, {"actions": [[0.0] * 44], "report_every": 5}, action_dim=44)
    for arguments, message in [({"actions": [[0.0] * 22], "steps": 1}, "Unknown arguments"),
                               ({"actions": [[0.0] * 22], "report_every": 4}, "report_every"),
                               ({"actions": [[0.0] * 22], "report_every": 151}, "report_every"),
                               ({"actions": [[0.0] * 22], "report_every": 5.0}, "report_every"),
                               ({"actions": [[0.0] * 21]}, "22"),
                               ({"actions": []}, "1..150")]:
        with pytest.raises(RejectedBeforeAction, match=message):
            T.validate_request(T.LOOKAHEAD, arguments)
    with pytest.raises(RejectedBeforeAction, match="44-D"):
        T.validate_request(T.LOOKAHEAD, {"actions": [[0.0] * 22]}, action_dim=44)


def test_switched_off_the_agent_text_omits_the_tool(monkeypatch):
    from services.dexjoco.benchmark import task_robot
    monkeypatch.setenv("RAX_LOOKAHEAD", "off")
    contract = Contract(benchmark="dexjoco")
    assert T.LOOKAHEAD not in {t["name"] for t in T.tool_definitions()}
    assert T.LOOKAHEAD not in contract.robot_names and T.LOOKAHEAD not in contract.manifest()["robot_tools"]
    with pytest.raises(PermissionError, match="RAX_LOOKAHEAD=off"):
        contract.require_robot(T.LOOKAHEAD, {"actions": [[0.0] * 22]})
    with pytest.raises(RejectedBeforeAction, match="RAX_LOOKAHEAD=off"):
        T.validate_request(T.LOOKAHEAD, {"actions": [[0.0] * 22]})
    assert task_robot("water_plant")["lookahead"] == ""
    off = render_robot_interface()
    monkeypatch.setenv("RAX_LOOKAHEAD", "on")
    on = render_robot_interface()
    assert "lookahead" not in off.lower() and "## Lookahead" in on
    assert ("lookahead never changes the episode; use it to\ncompare candidate rows, then execute the chosen rows "
            "with `dexjoco_step`") in on
    assert 'ctx.call("dexjoco_lookahead"' in on and "lookahead_budget" in on and "wall-clock time" in on


def render_robot_interface():
    from services.workspace.templates import render
    root = Path(__file__).resolve().parents[2]
    text = (root / "agent_skills/control/dexjoco/robot-interface/SKILL.md").read_text()
    facts = {"robot": BENCHMARKS["dexjoco"].robot_facts("water_plant"), "camera_geometry": True, "multi_env": False,
             "agent_formal": False, "expert_demonstrations": False, "formal_episode": False}
    return render(text, SimpleNamespace(context=lambda: facts), vocab={})


def test_the_settings_reach_the_frontend_and_the_workers():
    from harness.codex_cli.auto_research import PROJECT, frontend_environment
    from services.dexjoco.objects import SESSION_KEPT_ENV
    from services.exploration.pool import WORKER_KEPT_ENV
    assert {"RAX_LOOKAHEAD", "RAX_LOOKAHEAD_ROWS_FACTOR"} <= SESSION_KEPT_ENV == WORKER_KEPT_ENV
    seen = frontend_environment({"RAX_LOOKAHEAD": "off", "RAX_LOOKAHEAD_ROWS_FACTOR": "2", "RAX_OTHER": "1"})
    assert seen == {"RAX_LOOKAHEAD": "off", "RAX_LOOKAHEAD_ROWS_FACTOR": "2", "PYTHONPATH": str(PROJECT)}


def test_the_goldens_render_at_the_default_lookahead_setting(monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import render_workspaces
    seen = []
    monkeypatch.setattr(render_workspaces, "cases", lambda: ["one"])
    monkeypatch.setattr(render_workspaces, "render_case",
                        lambda name, scratch: seen.append(os.environ.get("RAX_LOOKAHEAD")) or scratch)
    monkeypatch.setattr(render_workspaces, "index", lambda path: {})
    monkeypatch.setenv("RAX_LOOKAHEAD", "off")
    assert render_workspaces.render_all() == {"one": {}}
    assert seen == [None] and os.environ["RAX_LOOKAHEAD"] == "off"


class LookaheadProcess(FakeProcess):
    def __init__(self, *args, move=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.move = move

    def request(self, op, **kwargs):
        if op != "lookahead":
            return super().request(op, **kwargs)
        self.requests.append((op, kwargs))
        if self.move:
            self.step_id += 1
        return {"episode_id": self.episode_id, "step_id": self.step_id, "rows_requested": len(kwargs["actions"]),
                "rows_simulated": len(kwargs["actions"]), "stop_reason": None, "end_row": None,
                "reports": [], "final": {"row": len(kwargs["actions"]), "state": [0.0] * 23,
                                         "object_states": {"objects": {}, "success": True}},
                "live_state_restored": True, "lookahead_budget": {"rows_used": 3}, "elapsed_s": 0.1}


def test_the_backend_routes_the_tool_without_moving_the_episode(tmp_path):
    backend = backend_for(tmp_path)
    backend.sim = LookaheadProcess(tmp_path / "native" / "results")
    backend.start(formal=False, seed=0)
    backend.call("dexjoco_step", {"actions": [[0.0] * 22] * 2})
    evaluation = backend.evaluate()
    cached = backend._evaluation
    value, images = backend.call(T.LOOKAHEAD, {"actions": [[0.0] * 22] * 3, "report_every": 0})
    assert images == [] and value["rows_simulated"] == 3 and value["step_id"] == 2
    assert "success" not in value["final"]["object_states"]  # The agent filter applies.
    op, kwargs = backend.sim.requests[-1]
    assert op == "lookahead" and kwargs["actions"].shape == (3, 22) and kwargs["report_every"] == 0
    assert backend._evaluation is cached and backend.evaluate() == evaluation
    assert backend._status()["lookahead"]["rows_per_episode"] == 3 * 1200
    backend.sim.move = True
    with pytest.raises(RuntimeError, match="moved the episode"):
        backend.call(T.LOOKAHEAD, {"actions": [[0.0] * 22]})


def test_the_gateway_serializes_it_with_motion(tmp_path):
    """Lookahead runs under the gateway's lock, the one every motion of this environment holds."""
    from services.controller.gateway import Gateway
    backend = backend_for(tmp_path)
    backend.sim = LookaheadProcess(tmp_path / "native" / "results")
    gateway = Gateway(backend, None, contract=backend.contract, audit=lambda *_: None, charge=lambda *_: None,
                      refund=lambda *_: None, budget=lambda: {})
    token = gateway.acquire()
    held = []
    original = backend.call

    def call(name, arguments):
        held.append((name, gateway._lock._is_owned()))
        return original(name, arguments)

    backend.call = call
    backend.start(formal=False, seed=0)
    listed = gateway.handle(token, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert T.LOOKAHEAD in {t["name"] for t in listed}
    reply = gateway.handle(token, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": T.LOOKAHEAD, "arguments": {"actions": [[0.0] * 22] * 2}}})
    value = json.loads(reply["result"]["content"][0]["text"])
    assert value["rows_simulated"] == 2 and value["episode_ended"] is False and held == [(T.LOOKAHEAD, True)]
