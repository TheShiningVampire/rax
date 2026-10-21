"""Trusted auto-research backend for DexJoCo; same interface as controller NativeBackend.

Runs in the host harness interpreter and never imports dexjoco or mujoco. Each episode is a
fresh simulator process (``services.dexjoco.process``) running the official evaluation
environment; exploration uses seeds outside the official ones (collection 0) and formal uses
the 150 official evaluation episodes (collection 1).
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path

import numpy as np

from services.artifact_io import write_artifacts
from services.dexjoco import lookahead as LA
from services.dexjoco import rotation_math as RM
from services.dexjoco import tasks as TK
from services.dexjoco import tools as T
from services.dexjoco.preview import PreviewProcess
from services.dexjoco.process import DexjocoProcess
from services.mcp_reply import _agent_safe, _depth_png_data, _jsonable, _png_data
from services.simulator.io import action_execution_scope, exit_report, reject_unless


class DexjocoBackend:
    # evaluate() reads the simulator's operator_state.json, which outlives its process.
    evaluates_from_record = True

    def __init__(self, config):
        from services.mcp_contract import Contract

        self.config = config
        self.info = TK.task_info(config.task)
        root = Path(config.root) / "native"
        root.mkdir(mode=0o700, exist_ok=True)
        self.contract = Contract.from_config(config, isolated=True)
        self.observation_profile = self.contract.observation_profile
        self.frame_workspace = root / "workspace"
        self.frame_root = self.frame_workspace / "runtime" / "frames"
        self.sim = DexjocoProcess(output_root=root / "results")
        # Robot-only preview on the simulator GPU, started on first use.
        self.preview = PreviewProcess(output_root=root / "preview", sim_gpu=config.sim_gpu, task=config.task)
        self._evaluation = None
        self._raw = None  # Last raw observation (host memory only), for pixel geometry.
        self._environment_spec = None

    # ------------------------------------------------------------------ tools
    def tools(self):
        return [t for t in self.contract.select(T.tool_definitions()) if t["name"] in self.contract.robot_names]

    @staticmethod
    def validate(name, arguments):
        T.validate_request(name, arguments)

    def _geometry(self):
        return self.contract.observation.geometry

    def call(self, name, arguments):
        # Each motion sends one mutating RPC, then assembles the packet from its frames.
        with action_execution_scope(self.sim, name):
            return self._call(name, arguments)

    def _call(self, name, arguments):
        self.contract.require_robot(name, arguments)
        T.validate_request(name, arguments, action_dim=TK.action_dim(self.config.task))
        if name == T.STATUS:
            return self._status(), []
        if name == T.ROTATION_MATH:  # Pure geometry: with or without an episode.
            return RM.rotation_math(arguments), []
        if name == T.PREVIEW_DESCRIBE:
            return self._preview_description(), []
        if name == T.PREVIEW_RUN:
            return self._preview_run(arguments)
        sim = self.sim
        if not sim.active:
            raise RuntimeError("No active DexJoCo episode; start an episode first")
        before = (sim.episode_id, sim.step_id)
        if name == T.OBSERVE:
            return self._observation_packet(sim.request("observe"))
        if name == T.OBJECT_STATES:
            return _agent_safe(_jsonable(sim.request("object_states"))), []
        if name == T.LOOKAHEAD:
            return self._lookahead(arguments, before), []
        if name == T.PIXEL_TO_POSITION:
            if not self._geometry():
                raise PermissionError("Pixel-to-position is unavailable with RGB-only observations")
            return self._pixel_to_position(arguments), []
        try:
            if name == T.STEP:
                result = sim.request("chunk_step", actions=np.asarray(arguments["actions"], dtype=np.float64))
            elif name == T.HOLD:
                result = sim.request("hold", steps=int(arguments["steps"]))
            else:
                raise ValueError(f"Unknown DexJoCo tool: {name}")
        finally:
            if (sim.episode_id, sim.step_id) != before:
                self._evaluation = None
        return self._motion_packet(result)

    def _lookahead(self, arguments, before):
        """dexjoco_lookahead: simulated rows from the live state, restored exactly by the simulator.
        Not a motion: the episode, its step and the cached evaluation stay as they were."""
        sim = self.sim
        result = sim.request("lookahead", actions=np.asarray(arguments["actions"], dtype=np.float64),
                             report_every=int(arguments.get("report_every", 0)))
        if (sim.episode_id, sim.step_id) != before:
            raise RuntimeError(f"dexjoco_lookahead moved the episode from {before} to {(sim.episode_id, sim.step_id)}")
        return _agent_safe(_jsonable(result))

    # ---------------------------------------------------------------- packets
    def _cameras(self):
        return list((self.sim.metadata or {}).get("cameras") or TK.official_config(self.config.task)["camera_mapping"].values())

    def _status(self):
        sim = self.sim
        dual = self.info["dual_arm"]
        return {"active": sim.active, "finished": sim.finished, "task": self.config.task,
                "episode_id": sim.episode_id, "step_id": sim.step_id if sim.episode_id else None,
                "environment_spec": self._environment_spec,
                "instruction": TK.official_config(self.config.task)["prompt"],
                "robot": "Franka Panda arm" + ("s" if dual else "") + " with 16-DoF Allegro hand" + ("s" if dual else ""),
                "action_layout": T.action_layout(dual), "state_layout": T.state_layout(dual),
                "cameras": self._cameras(), "max_action_rows": T.MAX_ACTION_ROWS,
                "control_frequency_hz": TK.CONTROL_HZ, "physics_seconds_per_step": TK.PHYSICS_SECONDS_PER_STEP,
                **({"lookahead": LA.limits(TK.native_step_limit(self.config.task))} if LA.lookahead_enabled() else {})}

    def _attachments(self, observation):
        content, attachments, depth = [], [], {}
        cameras = self._cameras()
        for camera in cameras:
            if camera in observation:
                content.append({"type": "image", "data": _png_data(np.asarray(observation[camera], dtype=np.uint8)),
                                "mimeType": "image/png"})
                attachments.append({"kind": "rgb", "camera": camera, "content_index": len(content)})
        if self._geometry():
            for camera in cameras:
                key = f"{camera}_depth_m"
                if key in observation:
                    values = np.asarray(observation[key], dtype=np.float32)
                    encoded, stats = _depth_png_data(np.where(np.isfinite(values), values, 0.0))
                    content.append({"type": "image", "data": encoded, "mimeType": "image/png"})
                    depth[camera] = dict(stats, content_index=len(content))
                    attachments.append({"kind": "depth", "camera": camera, "content_index": len(content)})
        return content, attachments, depth

    def _observation_fields(self, observation, attachments, depth):
        fields = {"state": observation.get("state"),
                  "images": [a["camera"] for a in attachments if a["kind"] == "rgb"],
                  "attachments": attachments}
        if self._geometry() and depth:
            fields.update(depth=depth, camera_intrinsics=observation.get("camera_intrinsics"),
                          camera_poses=observation.get("camera_poses"))
        return fields

    def _observation_packet(self, observation, *, transition=None):
        self._raw = (self.sim.episode_id, self.sim.step_id, observation)
        content, attachments, depth = self._attachments(observation)
        packet = {"episode_id": self.sim.episode_id, "step_id": self.sim.step_id, "task": self.config.task,
                  "instruction": observation.get("instruction"), "environment_spec": self._environment_spec,
                  **self._observation_fields(observation, attachments, depth)}
        if transition is not None:
            packet["transition"] = _jsonable(transition)
        return self.contract.observation.clean(_agent_safe(packet)), content

    def _motion_packet(self, result):
        frames = result.pop("frames", []) or []
        transition = dict(result)
        if not frames:
            return self._observation_packet(self.sim.request("observe"), transition=transition)
        records, contents = [], []
        for index, observation in enumerate(frames):
            content, attachments, depth = self._attachments(observation)
            records.append({"frame_index": index, "step_id": int(observation["step_id"]),
                            **self._observation_fields(observation, attachments, depth)})
            contents.append(content)
        sequence_id, manifest_path = self._save_sequence(records, contents, transition)
        self._raw = (self.sim.episode_id, self.sim.step_id, frames[-1])
        final, final_content = records[-1], contents[-1]
        directory = Path(manifest_path).parent.as_posix()
        packet = {"episode_id": self.sim.episode_id, "step_id": self.sim.step_id, "task": self.config.task,
                  "instruction": frames[-1].get("instruction"), "environment_spec": self._environment_spec,
                  **{k: v for k, v in final.items() if k != "frame_index"},
                  "transition": _jsonable(transition),
                  "frame_sequence": {
                      "sequence_id": sequence_id, "frequency_hz": TK.CONTROL_HZ / T.FRAME_STRIDE,
                      "frame_stride_steps": T.FRAME_STRIDE, "frame_count": len(records),
                      "manifest_path": manifest_path,
                      "frame_path_pattern": f"{directory}/frame_{{frame_index:06d}}_{{camera}}_{{kind}}.png",
                      "frame_index_range": [0, len(records) - 1],
                      "streams": [{"camera": a["camera"], "kind": a["kind"]} for a in final["attachments"]],
                      "intermediate_frames": "rgb only; the final frame adds depth when the profile has it",
                      "returned_images": "final_frame_only"}}
        return self.contract.observation.clean(_agent_safe(packet)), final_content

    # ---------------------------------------------------------------- preview
    def _preview_description(self):
        dual = self.info["dual_arm"]
        return {
            "tool": T.PREVIEW_RUN,
            "simulator": "robot-only MuJoCo: this task's official Panda + Allegro model ("
                         + ("two arms" if dual else "one arm") + ") on a floor plane",
            "control": "the official per-step control: rows converted by the official rotation-vector conversion, "
                       "the official operational-space flange controller with the environment's own gains, "
                       f"{TK.PHYSICS_SECONDS_PER_STEP * 1000:g} ms of physics per row, the Allegro position servos",
            "frame": "the episode's world frame: row targets and state vectors mean the same as in dexjoco_step; "
                     "the cameras stay fixed",
            "cameras": {name: "fixed third-person RGB, 480x480" for name in T.PREVIEW_CAMERAS},
            "inputs": {"actions": f"1..{T.MAX_ACTION_ROWS} dexjoco_step rows ({TK.action_dim(self.config.task)}-D)",
                       "initial_state": f"optional official state vector ({TK.state_dim(self.config.task)}-D); "
                                        "default: the episode's measured state"},
            "start": "each run starts the arm in its home posture with the start hand joints, then the same control "
                     "drives the flange to the start pose and lets it come to rest before the rows run; start_settle "
                     "reports how closely it got there. The state has no arm joint angles, so the elbow posture is "
                     "the controller's own, not necessarily the episode arm's.",
            "outputs": ["start_state and end_state: official state vectors",
                        f"third-person frames every {T.FRAME_STRIDE}th row with the state vector, final frames attached"],
            "not_available": ["table, objects or contact with the scene", "grasps, forces on objects or task outcome",
                              "the episode cameras"],
            "counts_toward_episode_steps": False}

    def _measured_state(self):
        """The live episode's official state vector, from the last observation when it is current."""
        sim = self.sim
        if self._raw is not None and self._raw[:2] == (sim.episode_id, sim.step_id):
            return [float(v) for v in self._raw[2]["state"]]
        observation = sim.request("observe")
        self._raw = (sim.episode_id, sim.step_id, observation)
        return [float(v) for v in observation["state"]]

    def _preview_run(self, arguments):
        sim = self.sim
        live = sim.active
        rows = np.asarray(arguments["actions"], dtype=np.float64)
        if "initial_state" in arguments:
            state, source = [float(v) for v in arguments["initial_state"]], "initial_state"
        else:
            if not live:
                raise RuntimeError("No active episode to take the start state from; pass initial_state")
            state, source = self._measured_state(), "episode_measured_state"
        result = self.preview.run(state, rows.tolist())
        records, contents = [], []
        for index, frame in enumerate(result["frames"]):
            content, attachments = [], []
            for camera in T.PREVIEW_CAMERAS:
                content.append({"type": "image", "data": _png_data(np.asarray(frame["images"][camera], dtype=np.uint8)),
                                "mimeType": "image/png"})
                attachments.append({"kind": "rgb", "camera": camera, "content_index": len(content)})
            records.append({"frame_index": index, "preview_row": int(frame["step"]), "state": frame["state"],
                            "attachments": attachments})
            contents.append(content)
        summary = {"status": result["status"], "rows": result["rows"], "start_source": source,
                   "requested_start_state": state, "start_state": result["start_state"],
                   "start_settle": result["settle"], "end_state": result["end_state"],
                   "rows_with_clipped_hand_targets": result["rows_with_clipped_hand_targets"],
                   **({"physics_warnings": result["physics_warnings"]} if "physics_warnings" in result else {}),
                   "preview_seconds": result["seconds"]}
        sequence_id, manifest_path = self._save_sequence(records, contents, summary, kind="dexjoco_preview")
        directory = Path(manifest_path).parent.as_posix()
        packet = {**summary,
                  "simulator": "robot-only preview (official robot and control on a floor plane; no table, "
                               "objects or contact with the scene)",
                  "cameras": (self.preview.metadata or {}).get("cameras"),
                  # Not `step_id`: the preview never advances or records the live episode.
                  "live_episode_id": sim.episode_id if live else None, "live_step_id": sim.step_id if live else None,
                  "frame_sequence": {
                      "sequence_id": sequence_id, "frame_stride_rows": T.FRAME_STRIDE, "frame_count": len(records),
                      "manifest_path": manifest_path,
                      "frame_path_pattern": f"{directory}/frame_{{frame_index:06d}}_{{camera}}_{{kind}}.png",
                      "frame_index_range": [0, len(records) - 1],
                      "streams": [{"camera": camera, "kind": "rgb"} for camera in T.PREVIEW_CAMERAS],
                      "returned_images": "final_frame_only"}}
        return _agent_safe(_jsonable(packet)), contents[-1]

    def _save_sequence(self, records, contents, transition, *, kind="dexjoco"):
        run_id = self.sim.run_id or "unscoped"
        sequence_id = f"seq_{self.sim.step_id:06d}_{uuid.uuid4().hex[:8]}"
        logical = Path("runtime") / "frames" / kind / run_id / sequence_id
        artifacts, saved = {}, []
        for record, content in zip(records, contents, strict=True):
            files = []
            for attachment in record["attachments"]:
                item = content[attachment["content_index"] - 1]
                filename = f"frame_{record['frame_index']:06d}_{attachment['camera']}_{attachment['kind']}.png"
                artifacts[filename] = base64.b64decode(item["data"])
                files.append({"kind": attachment["kind"], "camera": attachment["camera"],
                              "path": (logical / filename).as_posix()})
            entry = {k: v for k, v in record.items() if k not in ("attachments",)}
            if "depth" in entry:
                entry["depth"] = {c: {k: v for k, v in d.items() if k != "content_index"} for c, d in entry["depth"].items()}
            entry["files"] = files
            saved.append(entry)
        manifest = {"sequence_id": sequence_id, "source": kind, "artifact_id": run_id,
                    "frequency_hz": TK.CONTROL_HZ / T.FRAME_STRIDE, "frame_count": len(saved),
                    "transition": _agent_safe(transition), "frames": saved}
        artifacts["manifest.json"] = (json.dumps(self.contract.observation.clean(_jsonable(manifest)),
                                                 ensure_ascii=False, indent=2) + "\n").encode()
        write_artifacts(self.frame_root / kind / run_id / sequence_id, artifacts)
        return sequence_id, (logical / "manifest.json").as_posix()

    def _pixel_to_position(self, arguments):
        if self._raw is None or self._raw[0] != self.sim.episode_id:
            raise ValueError("Observe first; no current observation")
        episode, step, observation = self._raw
        reject_unless(arguments["observation_step_id"] == self.sim.step_id and step == self.sim.step_id,
                      f"Stale pixel selection: observation_step_id={arguments['observation_step_id']}, "
                      f"current step_id={self.sim.step_id}; observe again and reselect")
        results = []
        for item in arguments["selections"]:
            camera = item["camera"]
            depth = observation.get(f"{camera}_depth_m")
            if depth is None:
                results.append({"camera": camera, "valid": False, "reason": "no depth for this camera"})
                continue
            point = T.pixel_to_world(depth, observation["camera_intrinsics"][camera], observation["camera_poses"][camera],
                                     float(item["u"]), float(item["v"]), int(arguments.get("depth_window_radius", 1)))
            results.append({"camera": camera, "u": item["u"], "v": item["v"], **point})
        return {"episode_id": episode, "observation_step_id": step, "selections": results}

    # -------------------------------------------------------------- lifecycle
    def start(self, *, formal, seed):
        self._evaluation = None
        self._raw = None
        c = self.config
        collection = c.formal_collection if formal else c.eval_seed
        TK.layout(c.task, collection, int(seed))
        if self.sim.process is not None:
            try:
                if self.sim.active:
                    self.sim.finish("formal_episode" if formal else "reset")
            finally:
                self.sim.stop()
        mode = "formal" if formal else "exploration"
        run_id = f"{c.task}_{mode}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        timeouts = TK.task_timeouts(c.task)
        reset = self.sim.start(task=c.task, collection=collection, layout_id=int(seed), run_id=run_id,
                               sim_gpu=c.sim_gpu, startup_timeout=timeouts["startup_seconds"],
                               include_depth=self._geometry(),
                               # The supervisor's own wall limit also bounds the episode.
                               episode_timeout_seconds=int(c.formal.wall_seconds))
        limit = int(reset["native_step_limit"])
        self._environment_spec = {"benchmark": "dexjoco", "episode_mode": mode, "episode_step_limit": limit,
                                  "native_step_limit": limit, "control_frequency_hz": TK.CONTROL_HZ,
                                  "startup_seconds": round(self.sim.startup_seconds or 0.0, 1)}
        packet, images = self._observation_packet(self.sim.request("observe"))
        packet.update(reset=_agent_safe({"episode_id": reset["episode_id"], "step_id": reset["step_id"]}),
                      environment_spec=self._environment_spec, artifact_id=self.sim.run_id,
                      artifact_storage="service_managed_not_agent_accessible")
        return packet, images

    def evaluate(self, agent_assessment=None, confidence=None):
        sim = self.sim
        identity = (sim.episode_id, sim.step_id)
        if self._evaluation is not None and self._evaluation[0] == identity:
            return dict(self._evaluation[1])
        if sim.output_dir is None or sim.episode_id is None:
            raise RuntimeError("No DexJoCo episode to evaluate")
        state = json.loads((sim.output_dir / "sim" / "operator_state.json").read_text())
        if state.get("episode_id") != sim.episode_id or state.get("step_id") != sim.step_id:
            raise RuntimeError(
                f"Native evaluation identity mismatch: the simulator last recorded episode "
                f"{state.get('episode_id')!r} at step {state.get('step_id')!r}, but this episode is "
                f"{sim.episode_id!r} at step {sim.step_id!r}; "
                + ("the simulator process was already stopped" if sim.process is None
                   else exit_report(sim.process, sim.log_path, since=sim.log_mark)))
        success = (state.get("reward") or {}).get("native_success")
        if type(success) is not bool:
            raise RuntimeError("Native evaluation unavailable")
        evaluation = {"status": "completed", "task_complete": success, "episode_id": sim.episode_id,
                      "step_id": sim.step_id, "task": self.config.task, "evaluation_kind": "binary_task_completion",
                      "decided_at_unix_s": time.time()}
        if agent_assessment:
            evaluation["agent_assessment"] = str(agent_assessment)[:2000]
        self._evaluation = (identity, dict(evaluation))
        return dict(evaluation)

    def close(self):
        try:
            if self.sim.active:
                self.sim.finish("close")
        finally:
            try:
                self.sim.stop()
            finally:
                self.preview.stop()
