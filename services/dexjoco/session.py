"""One DexJoCo episode inside the simulator process (runs in runtime/envs/dexjoco).

The official evaluation wrapper ``DexJoCoOpenPIEnv`` (dexjoco_openpi_client) owns environment
creation, reset, stepping, the hold (``stay``), termination and success, exactly as
``dexjoco-openpi-eval`` drives it: ``_set_seed(seed)``, ``start()``, then one ``reset()`` per
episode, the click_mouse pre-roll, and policy rows through ``step()``. The only difference is
timing: the harness steps synchronously when the agent acts (zero-latency evaluation), while
the official loop holds with ``stay()`` during inference.

The session adds the harness contract: identity-checked RPC ops, agent observations (the
evaluation cameras at full resolution, optional depth rendered with the official
``collect_depth_frames``, calibration, the official state vector, the prompt), and the trusted
operator files that dashboards and formal scoring read. Experiment
(exp/add-state-info/object): the ``object_states`` op gives the agent the objects' live poses
and the arm joints (services/dexjoco/objects.py object_states) and, at the ``full`` privileged
level (RAX_PRIVILEGED_STATE), velocities, sizes, contacts and the task's goal specification
(services/dexjoco/goals.py); the layout, the success info and the reset tail stay in
operator-only files.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np

from services.dexjoco import tasks as T
from services.dexjoco.objects import object_states, privileged_level, read_all as read_live, scene_index
from services.dexjoco.tools import FRAME_STRIDE
from services.simulator.io import RejectedBeforeAction, action_execution_scope, reject_unless, require, write_json

SCHEMA = "dexjoco_operator_state_v1"
# The official evaluation's click_mouse pre-roll row (eval_dexjoco_openpi.py main(), "Align with dataset").
CLICK_MOUSE_PREROLL = np.array([
    -4.4294e-01, 1.3729e-06, 1.5170e00,
    -3.14156462e00, -6.91584035e-05, -1.40317984e-03,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0.263, 0, 0, 0])
VIDEO_FPS = T.CONTROL_HZ


MAX_INFO_SEQUENCE = 200  # Entries kept of each list-valued info key (e.g. the iPad's pressed digits).


def hold_row(state, dual_arm):
    """The official row stay() sends for a held state (DexJoCoOpenPIEnv.stay): each flange's
    position and its quaternion as a rotation vector, then the hand joints."""
    from scipy.spatial.transform import Rotation

    s = np.asarray(state, dtype=np.float64)

    def arm(pose):
        return [*pose[:3], *Rotation.from_quat(pose[[4, 5, 6, 3]]).as_rotvec()]

    if dual_arm:
        row = [*arm(s[0:7]), *s[14:30], *arm(s[7:14]), *s[30:46]]
    else:
        row = [*arm(s[0:7]), *s[7:23]]
    return [float(v) for v in row]


class _Tap:
    """Pass-through around the wrapped gym env that keeps the last full observation and info.

    The official wrapper discards ``info`` and slices the state; the operator files need the
    environment's own success info and the privileged state tail. Since the last reset it also
    remembers which boolean info keys were ever true (e.g. ``hammer_hit``, ``shutter_pressed``)
    and the entries of list-valued ones (``pressed_digits``), for episode reviews. Behaviour is
    unchanged."""

    def __init__(self, env):
        self._env = env
        self.last_obs = None
        self.last_info = {}
        self.ever_true = set()
        self.sequences = {}

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        self.last_obs, self.last_info = obs, dict(info or {})
        self.ever_true, self.sequences = set(), {}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.last_obs, self.last_info = obs, dict(info or {})
        for key, value in self.last_info.items():
            if value is True or isinstance(value, np.bool_) and bool(value):
                self.ever_true.add(key)
            elif isinstance(value, (list, tuple)) and value:
                kept = self.sequences.setdefault(key, [])
                kept.extend(value[:MAX_INFO_SEQUENCE - len(kept)])
        return obs, reward, terminated, truncated, info

    def history(self):
        return {"ever_true": sorted(self.ever_true), "sequences": _jsonable_info(self.sequences)}


def _jsonable_info(info):
    out = {}
    for key, value in info.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[key] = value
        elif isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, (list, tuple, np.ndarray)):
            try:
                out[key] = np.asarray(value).tolist()
            except Exception:  # noqa: BLE001 - diagnostics only
                out[key] = repr(value)
        elif isinstance(value, dict):
            out[key] = _jsonable_info(value)
        else:
            out[key] = repr(value)
    return out


class DexjocoSession:
    def __init__(self, output, task, collection, *, include_depth):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.info = T.task_info(task)
        self.config = T.official_config(task)
        self.collection = int(collection)
        self.episode_mode = "formal" if self.collection == T.FORMAL_COLLECTION else "exploration"
        self.include_depth = bool(include_depth)
        self.privileged = privileged_level()  # Fails fast on an invalid RAX_PRIVILEGED_STATE.
        self.scene = None  # Full level: object labels, robot bodies and object sizes at reset.
        self.cameras = list(self.config["camera_mapping"].values())
        self.step_limit = T.native_step_limit(task)
        self.env = None
        self.tap = None
        self.episode_id = None
        self.episode_dir = None
        self.layout_id = None
        self.layout = None
        self.step_id = 0
        self.terminated = self.truncated = self.success = False
        self.invalid_reason = None  # set only for a concrete reason, e.g. episode_wall_timeout
        self.finish_reason = None
        self.in_stay = False
        self.obs = None
        self.video = None
        self.metadata = {
            "task": task, "benchmark": "dexjoco", "env_name": self.info["env_name"], "regime": self.info["regime"],
            "robot_type": self.info["robot_type"], "instruction": self.config["prompt"],
            "camera_mapping": dict(self.config["camera_mapping"]), "cameras": self.cameras,
            "control_frequency_hz": T.CONTROL_HZ, "physics_seconds_per_step": T.PHYSICS_SECONDS_PER_STEP,
            "native_step_limit": self.step_limit, "env_step_cap": self.info["env_step_cap"],
            "preroll_steps": self.info["preroll_steps"], "action_dim": T.action_dim(task),
            "state_dim": T.state_dim(task), "depth": self.include_depth, "frame_stride": FRAME_STRIDE,
            "episode_mode": self.episode_mode,
        }

    # ------------------------------------------------------------- environment
    def _build(self, seed):
        from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
        from dexjoco_openpi_client.eval_dexjoco_openpi import _set_seed

        # As eval_dexjoco_openpi.main(): global seeds, then the wrapper and its environment.
        _set_seed(seed)
        env = DexJoCoOpenPIEnv(
            env_name=self.info["env_name"], camera_mapping=self.config["camera_mapping"], seed=seed,
            rand_full=self.info["regime"] == "rand_full", randomize_dynamics=False,
            dual_arm=self.info["dual_arm"], prompt=self.config["prompt"], render_mode="rgb_array",
            password=self.config.get("password", None))
        env.start()
        self.tap = _Tap(env.env)
        env.env = self.tap
        self.env = env

    def _image_keys(self):
        return [key for key in self.tap.last_obs if key != "state"]

    def _calibration(self, obs):
        import mujoco
        from dexjoco.data.depth_capture import _env_viewer_and_camera_ids

        raw = self.tap.unwrapped
        _, ids = _env_viewer_and_camera_ids(raw, self._image_keys())
        require(ids is not None, "camera ids unavailable for this environment")
        by_key = dict(zip(self._image_keys(), ids))
        intrinsics, poses = {}, {}
        for camera in self.cameras:
            cid = int(by_key[camera])
            image = obs[camera]
            height, width = int(image.shape[0]), int(image.shape[1])
            fovy = float(raw._model.cam_fovy[cid])
            focal = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
            intrinsics[camera] = {"matrix": [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
                                  "width": width, "height": height, "fovy_deg": fovy}
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.asarray(raw._data.cam_xmat[cid], dtype=np.float64))
            poses[camera] = {"position_m": [float(v) for v in raw._data.cam_xpos[cid]],
                             "quaternion_wxyz": [float(v) for v in quat], "frame": "world",
                             "convention": "OpenGL camera: looks along -Z, +Y up; image row 0 at the top"}
        return intrinsics, poses

    def _agent_observation(self, *, depth=None):
        """Official evaluation inputs at full resolution, plus depth and calibration (RGB-D)."""
        depth = self.include_depth if depth is None else depth
        raw = self.env.get_raw_images()
        obs = {camera: np.ascontiguousarray(np.asarray(raw[camera])[..., :3]).astype(np.uint8) for camera in self.cameras}
        state = np.asarray(self.env.obs["state"], dtype=np.float64)
        obs["state"] = state.tolist()
        obs["instruction"] = self.config["prompt"]
        obs["step_id"] = int(self.step_id)
        if depth:
            from dexjoco.data.depth_capture import collect_depth_frames
            frames = collect_depth_frames(self.tap, self._image_keys())
            for camera in self.cameras:
                if camera in frames:
                    obs[f"{camera}_depth_m"] = np.asarray(frames[camera], dtype=np.float32)
            obs["camera_intrinsics"], obs["camera_poses"] = self._calibration(obs)
        return obs

    # ----------------------------------------------------------------- episode
    def reset(self, layout_id, source="agent", policy_version="mcp"):
        require(self.episode_id is None, "One fresh episode per process; no reset")
        self.layout_id = int(layout_id)
        self.layout = T.layout(self.task, self.collection, self.layout_id)
        started = time.monotonic()
        self._build(self.layout["seed"])
        # Official episode k of a seed is the (k+1)-th reset of the environment built with it.
        for _ in range(self.layout["episode"] + 1):
            self.env.reset()
        self.initial_state = np.asarray(self.tap.last_obs["state"], dtype=np.float64).tolist()
        self.initial_info = _jsonable_info(self.tap.last_info)
        self.initial_live = read_live(self.tap.unwrapped._data, self.info["env_name"])
        if self.privileged == "full":
            raw = self.tap.unwrapped
            try:
                self.scene = scene_index(raw._model, raw._data, self.info["env_name"])
            except Exception as exc:  # noqa: BLE001 - reported by the state tool, never breaks the episode
                self.scene = {"error": f"{type(exc).__name__}: {exc}"}
        for _ in range(self.info["preroll_steps"]):
            self.env.step(action=CLICK_MOUSE_PREROLL)
        self.episode_id = f"{self.task}_{self.episode_mode}_{self.layout_id}_{uuid.uuid4().hex[:8]}"
        self.episode_dir = self.output / self.episode_id
        self.episode_dir.mkdir(parents=True)
        self.step_id = 0
        self.in_stay = False
        self._open_video()
        self.obs = self._agent_observation()
        self._write_video_frame()
        write_json(self.output / "reset.json", {
            "episode_id": self.episode_id, "task": self.task, "env_name": self.info["env_name"],
            "regime": self.info["regime"], "collection": self.collection, "layout_id": self.layout_id,
            "official_seed": self.layout["seed"], "official_episode_index": self.layout["episode"],
            "preroll_steps": self.info["preroll_steps"], "source": source, "policy_version": policy_version,
            "native_step_limit": self.step_limit, "reset_seconds": round(time.monotonic() - started, 3),
            "started_at_unix_s": time.time()})
        (self.output / "reset.json").chmod(0o600)
        write_json(self.output / "private_initial_state.json", {
            "episode_id": self.episode_id, "state": self.initial_state, "info": _jsonable_info(self.tap.last_info)})
        (self.output / "private_initial_state.json").chmod(0o600)
        self._write_operator_state()
        return {"episode_id": self.episode_id, "step_id": 0, "native_step_limit": self.step_limit}

    def _open_video(self):
        import imageio

        self.video = imageio.get_writer(
            str(self.output / "sensors.mp4"), fps=VIDEO_FPS, codec="libx264", pixelformat="yuv420p",
            macro_block_size=16, output_params=["-movflags", "+faststart"])

    def _write_video_frame(self):
        if self.video is None:
            return
        raw = self.env.get_raw_images()
        self.video.append_data(np.concatenate([np.asarray(raw[c])[..., :3] for c in self.cameras], axis=1))

    def _close_video(self):
        if self.video is not None:
            try:
                self.video.close()
            finally:
                self.video = None

    def _after_step(self):
        self.step_id += 1
        self.terminated = bool(self.env.is_done)
        self.success = bool(self.env.is_success)
        self._write_video_frame()
        return self.terminated

    def _execute(self, rows, *, kind):
        """Run official rows (or holds); keep every FRAME_STRIDE-th frame plus the final one.
        Operator state is written once, at the end of the call."""
        require(not (self.terminated or self.truncated), "Episode already ended")
        frames, steps = [], []
        for row in rows:
            executed = row()
            ended = self._after_step()
            steps.append({"step_id": self.step_id, "executed_action": executed,
                          "terminated": self.terminated, "truncated": False,
                          **({"success": self.success} if ended else {})})
            if self.step_id % FRAME_STRIDE == 0 and not ended:
                frames.append(self._agent_observation(depth=False))
            if ended:
                break
        self.obs = self._agent_observation()
        frames.append(self.obs)
        if self.terminated:
            self._write_summary("terminal")
        else:
            self._write_operator_state()
        return {"episode_id": self.episode_id, "step_id": self.step_id, "kind": kind,
                "observation_frequency_hz": T.CONTROL_HZ / FRAME_STRIDE, "control_frequency_hz": T.CONTROL_HZ,
                "frame_stride": FRAME_STRIDE, "frame_count": len(frames), "frames": frames, "steps": steps,
                "terminated": self.terminated, "truncated": False,
                **({"success": self.success} if self.terminated else {})}

    def chunk_step(self, actions):
        matrix = np.asarray(actions, dtype=np.float64)
        dim = T.action_dim(self.task)
        reject_unless(matrix.ndim == 2 and matrix.shape[1] == dim and len(matrix) >= 1,
                      f"actions must be rows of {dim} numbers")
        reject_unless(bool(np.isfinite(matrix).all()), "actions must be finite")

        def row(values):
            def run():
                self.env.step(values)
                self.in_stay = False
                return values.tolist()
            return run

        return self._execute((row(r) for r in matrix), kind="official_actions")

    def hold(self, steps):
        reject_unless(type(steps) is int and steps >= 1, "steps must be a positive integer")

        def run():
            # As the official loop: the first stay() targets the measured state, the next ones keep it.
            self.env.stay(continue_stay=self.in_stay)
            self.in_stay = True
            # Report the row the hold sent, like an official row's, so held steps can be recorded.
            return hold_row(self.env.last_stay_state, self.info["dual_arm"])

        return self._execute((run for _ in range(steps)), kind="hold")

    # ---------------------------------------------------------- operator files
    def _write_operator_state(self, reason=None):
        if self.episode_id is None:
            return
        state = {
            "schema": SCHEMA, "simulator_pid": os.getpid(), "updated_at_unix_s": time.time(),
            "benchmark": "dexjoco", "task": self.task, "env_name": self.info["env_name"],
            "regime": self.info["regime"], "instruction": self.config["prompt"], "episode_id": self.episode_id,
            "collection": self.collection, "layout_id": self.layout_id, "official_seed": self.layout["seed"],
            "official_episode_index": self.layout["episode"], "episode_mode": self.episode_mode,
            "step_id": int(self.step_id), "native_reference_step_limit": self.step_limit,
            "step_limit_enforced": True, "episode_step_limit": self.step_limit,
            # Latest RGB frames for the operator's live view (overwritten; history is in sensors.mp4).
            "observation_file": f"{self.episode_id}/live.npz", "cameras": self.cameras, "reason": reason,
            "reward": {"native_success": bool(self.success), "native_reward": 1.0 if self.success else 0.0,
                       "native_score_percent": 100.0 if self.success else 0.0,
                       "terminated": bool(self.terminated), "truncated": False,
                       "valid_for_success_rate": bool(self.terminated and not self.invalid_reason)},
            "env_info": _jsonable_info(self.tap.last_info) if self.tap is not None else {},
        }
        live = self.episode_dir / "live.npz"
        with open(live.with_suffix(".tmp"), "wb") as stream:
            np.savez(stream, **{camera: self.obs[camera] for camera in self.cameras if camera in self.obs})
        live.with_suffix(".tmp").replace(live)
        path = self.output / "operator_state.json"
        write_json(path, state)
        path.chmod(0o600)

    def _write_goal_report(self, reason):
        """Ground truth for operator-side episode reviews (services/dexjoco/analysis.py): the
        environment's own info at reset and at the end, which boolean info keys were ever true and
        the entries of list-valued ones, the official state with its privileged tail (the objects'
        reset poses) at reset and at the end, and the objects' poses and the points and angles the
        success conditions use at reset and at the end, read from the environment's MuJoCo data
        (services/dexjoco/objects.py)."""
        try:
            final = np.asarray(self.tap.last_obs["state"], dtype=np.float64).tolist()
            report = {"schema": "dexjoco_goal_report_v1", "episode_id": self.episode_id, "task": self.task,
                      "env_name": self.info["env_name"], "instruction": self.config["prompt"],
                      "episode_mode": self.episode_mode, "reason": reason, "step_id": int(self.step_id),
                      "step_limit": self.step_limit, "success": bool(self.success),
                      "terminated": bool(self.terminated), "env_info": _jsonable_info(self.tap.last_info),
                      "initial_env_info": self.initial_info, "info_history": self.tap.history(),
                      "initial_state": self.initial_state, "final_state": final,
                      "state_dim": T.state_dim(self.task), "initial_live": self.initial_live,
                      "live": read_live(self.tap.unwrapped._data, self.info["env_name"])}
        except Exception as exc:  # noqa: BLE001 - the review must never break scoring or control
            report = {"schema": "dexjoco_goal_report_v1", "episode_id": self.episode_id,
                      "error": f"{type(exc).__name__}: {exc}"}
        path = self.output / "private_goal_report.json"
        write_json(path, report)
        path.chmod(0o600)

    def _write_summary(self, reason):
        if self.episode_id is None:
            return
        reason = self.finish_reason or reason
        complete = bool(self.terminated)
        eligible = complete and not self.invalid_reason
        write_json(self.output / "evaluation_outcome.json", {
            "complete": complete, "valid_for_success_rate": eligible,
            "native_success": self.success if eligible else None,
            # Binary scoring: once closed, the outcome is decided unless the episode was invalidated
            # (e.g. a wall timeout); a controller that returned early scores what it reached.
            "native_score": None if self.invalid_reason else (1.0 if self.success else 0.0),
            "valid_for_score": not self.invalid_reason, "score_source": "dexjoco_success",
            "invalid_reason": self.invalid_reason, "status": "native_completed" if eligible else "incomplete",
            "reason": reason,
            "end_reason": "task_complete" if self.success else ("environment_terminated" if self.terminated else None),
            "native_control_steps": self.step_id, "native_reference_step_limit": self.step_limit,
            "step_limit_enforced": True, "episode_step_limit": self.step_limit})
        write_json(self.output / "summary.json", {
            "episode_id": self.episode_id, "step_id": self.step_id, "success": self.success,
            "terminated": self.terminated, "truncated": False, "complete": complete, "reason": reason,
            "control_dt": 1.0 / T.CONTROL_HZ, "no_rollback": True, "collection": self.collection,
            "layout_id": self.layout_id, "official_seed": self.layout["seed"],
            "official_episode_index": self.layout["episode"], "episode_mode": self.episode_mode})
        self._write_operator_state(reason=reason)
        self._write_goal_report(reason)

    def finish(self, reason):
        self.finish_reason = self.finish_reason or reason
        self._write_summary(reason)
        self._close_video()
        return {"episode_id": self.episode_id, "step_id": self.step_id, "reason": reason}

    def close(self):
        self._close_video()
        if self.env is not None:
            self.env.close()
            self.env = None

    # --------------------------------------------------------------- dispatch
    def _check_identity(self, episode_id, step_id):
        require(self.episode_id is not None, "No active episode")
        reject_unless(episode_id == self.episode_id, "Stale episode")
        reject_unless(int(step_id) == self.step_id, f"Stale step: {step_id} != {self.step_id}")

    def dispatch(self, op, args):
        with action_execution_scope(self, op):
            return self._dispatch(op, args)

    def _dispatch(self, op, args):
        if op == "metadata":
            return self.metadata
        if op == "reset":
            return self.reset(**args)
        if op == "finish_pilot":
            require(args["episode_id"] == self.episode_id, "Stale episode")
            return self.finish(str(args.get("reason", "finish")))
        self._check_identity(args["episode_id"], args["step_id"])
        data = {k: v for k, v in args.items() if k not in ("episode_id", "step_id")}
        if op == "observe":
            return self.obs
        if op == "object_states":
            raw = self.tap.unwrapped
            return {"episode_id": self.episode_id, "step_id": self.step_id,
                    **object_states(raw._data, self.info["env_name"], self.info["dual_arm"], level=self.privileged,
                                    model=raw._model, env=raw, scene=self.scene)}
        if op == "chunk_step":
            return self.chunk_step(data["actions"])
        if op == "hold":
            return self.hold(int(data["steps"]))
        raise RejectedBeforeAction(f"Unknown op: {op}")

