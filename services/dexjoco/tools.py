"""DexJoCo MCP tool schemas, host-side request validation and depth-pixel geometry.

Pure Python + numpy: safe to import in the host harness interpreter. The action and state
layouts are the official evaluation interface (``DexJoCoOpenPIEnv``): actions are the policy's
rotation-vector rows, states the environment's first 23 (single arm) or 46 (dual arm) values.
"""

from __future__ import annotations

import math

import numpy as np

from services.benchmarks import DEXJOCO_TOOLS
from services.dexjoco import rotation_math as RM
from services.simulator.io import RejectedBeforeAction, reject_unless

STATUS, OBSERVE = "dexjoco_status", "dexjoco_observe"
STEP, HOLD = "dexjoco_step", "dexjoco_hold"
PIXEL_TO_POSITION = "dexjoco_pixel_to_position"
# Experiment (exp/add-state-info/object): privileged live object state from the simulator.
OBJECT_STATES = "dexjoco_object_states"
ROTATION_MATH = RM.TOOL
PREVIEW_DESCRIBE, PREVIEW_RUN = "dexjoco_preview_describe", "dexjoco_preview_run"
PREVIEW_CAMERAS = ("preview_front", "preview_side", "preview_top")
# Robot-only preview: never advances the episode or counts against its step limit.
PREVIEW_TOOLS = frozenset({PREVIEW_DESCRIBE, PREVIEW_RUN})

ROBOT_TOOLS = frozenset({STATUS, OBSERVE, STEP, HOLD, PIXEL_TO_POSITION, OBJECT_STATES, ROTATION_MATH, *PREVIEW_TOOLS})
assert ROBOT_TOOLS == DEXJOCO_TOOLS, "services/benchmarks.py and the DexJoCo tool set disagree"
MUTATING = frozenset({STEP, HOLD})
MAX_ACTION_ROWS = 150
# Every FRAME_STRIDE-th executed step is saved as a frame (plus the final step).
FRAME_STRIDE = 10
# Every camera name an official evaluation config maps (env image keys).
CAMERAS = ("front", "random_camera", "ego", "ego_right", "wrist", "wrist_left", "wrist_right")
# Geometry-only fields removed by the RGB-only profile.
GEOMETRY_KEYS = frozenset({"camera_poses", "camera_intrinsics", *(f"{camera}_depth_m" for camera in CAMERAS)})

HAND_JOINTS = tuple(f"{finger}{i}" for finger in ("ff", "mf", "rf", "th") for i in range(4))
# The official left Allegro hand reports and takes its joints mirrored: rf, mf, ff, th.
LEFT_HAND_JOINTS = tuple(f"{finger}{i}" for finger in ("rf", "mf", "ff", "th") for i in range(4))


def action_layout(dual_arm):
    """Official policy action rows (DexJoCoOpenPIEnv.step)."""
    if dual_arm:
        return {"dim": 44, "fields": [
            {"slice": [0, 3], "name": "right_flange_position_m", "frame": "world"},
            {"slice": [3, 6], "name": "right_flange_rotvec_rad", "frame": "world", "note": "axis-angle"},
            {"slice": [6, 22], "name": "right_hand_joint_targets_rad", "joints": list(HAND_JOINTS)},
            {"slice": [22, 25], "name": "left_flange_position_m", "frame": "world"},
            {"slice": [25, 28], "name": "left_flange_rotvec_rad", "frame": "world", "note": "axis-angle"},
            {"slice": [28, 44], "name": "left_hand_joint_targets_rad", "joints": list(LEFT_HAND_JOINTS)}],
            "semantics": "absolute targets; the arm tracks the flange pose with the official operational-space "
                         "controller, the hands track joint targets with the official position actuators"}
    return {"dim": 22, "fields": [
        {"slice": [0, 3], "name": "flange_position_m", "frame": "world"},
        {"slice": [3, 6], "name": "flange_rotvec_rad", "frame": "world", "note": "axis-angle"},
        {"slice": [6, 22], "name": "hand_joint_targets_rad", "joints": list(HAND_JOINTS)}],
        "semantics": "absolute targets; the arm tracks the flange pose with the official operational-space "
                     "controller, the hand tracks joint targets with the official position actuators"}


def state_layout(dual_arm):
    """The official evaluation state (env observation `state`, first 23 or 46 values)."""
    if dual_arm:
        return {"dim": 46, "fields": [
            {"slice": [0, 3], "name": "right_flange_position_m", "frame": "world"},
            {"slice": [3, 7], "name": "right_flange_quaternion_wxyz", "frame": "world"},
            {"slice": [7, 10], "name": "left_flange_position_m", "frame": "world"},
            {"slice": [10, 14], "name": "left_flange_quaternion_wxyz", "frame": "world"},
            {"slice": [14, 30], "name": "right_hand_joint_positions_rad", "joints": list(HAND_JOINTS)},
            {"slice": [30, 46], "name": "left_hand_joint_positions_rad", "joints": list(LEFT_HAND_JOINTS)}]}
    return {"dim": 23, "fields": [
        {"slice": [0, 3], "name": "flange_position_m", "frame": "world"},
        {"slice": [3, 7], "name": "flange_quaternion_wxyz", "frame": "world"},
        {"slice": [7, 23], "name": "hand_joint_positions_rad", "joints": list(HAND_JOINTS)}]}


def tool_definitions():
    return [
        {"name": STATUS, "description": (
            "Episode status, step budget, official action and state layouts, cameras and instruction. No physics."),
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": OBSERVE, "description": (
            "Current official DexJoCo observation: the evaluation cameras' RGB (640x640) and depth attachments, "
            "calibration, the official state vector (flange poses and hand joints) and the instruction. "
            "No physics; the observation is cached after the last motion."),
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": STEP, "description": (
            f"Execute 1..{MAX_ACTION_ROWS} rows of the official DexJoCo policy action, one environment step each "
            f"({CONTROL_TEXT}): single arm 22-D [flange xyz (m, world), flange rotation vector (rad, world), "
            "16 Allegro joint targets (rad)]; dual arm 44-D [right xyz, right rotvec, right hand 16, left xyz, "
            "left rotvec, left hand 16]. Targets are absolute; status gives the exact layout. Stops early when the "
            f"episode ends. Returns the final frame and saves every {FRAME_STRIDE}th frame."),
         "inputSchema": {"type": "object", "required": ["actions"], "additionalProperties": False, "properties": {
             "actions": {"type": "array", "minItems": 1, "maxItems": MAX_ACTION_ROWS,
                         "items": {"type": "array", "items": {"type": "number"}, "minItems": 22, "maxItems": 44}}}}},
        {"name": HOLD, "description": (
            f"Hold the current pose for 1..{MAX_ACTION_ROWS} environment steps with the official stay(): the first "
            "held step targets the current measured flange pose and hand joints, consecutive holds keep that target. "
            "Each held step reports the row it sent as its executed_action and counts against the episode limit."),
         "inputSchema": {"type": "object", "required": ["steps"], "additionalProperties": False, "properties": {
             "steps": {"type": "integer", "minimum": 1, "maximum": MAX_ACTION_ROWS}}}},
        {"name": PIXEL_TO_POSITION, "description": (
            "Back-project 1..32 pixels of the current observation's depth images into world-frame points "
            "(metres), using each camera's calibration. Pass the observation's step_id."),
         "inputSchema": {"type": "object", "required": ["observation_step_id", "selections"],
                         "additionalProperties": False, "properties": {
                             "observation_step_id": {"type": "integer"},
                             "selections": {"type": "array", "minItems": 1, "maxItems": 32, "items": {
                                 "type": "object", "required": ["camera", "u", "v"], "additionalProperties": False,
                                 "properties": {"camera": {"type": "string", "enum": list(CAMERAS)},
                                                "u": {"type": "number"}, "v": {"type": "number"}}}},
                             "depth_window_radius": {"type": "integer", "minimum": 0, "maximum": 5}}}},
        {"name": OBJECT_STATES, "description": object_states_description(),
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": ROTATION_MATH, "description": (
            "Rotation and pose arithmetic for dexjoco_step rows, converted exactly as the official row conversion "
            "does (scipy Rotation: rotation vector <-> quaternion wxyz). Operations: convert_rotation (to "
            "quaternion_wxyz, rotation_vector and matrix); compose_pose (pose_a, then pose_b expressed in pose_a's "
            "frame); relative_pose (pose_b expressed in pose_a's frame); flange_target (the flange pose that puts a "
            "point fixed to the hand, given by its offset in the flange frame, at point_target_m with the flange "
            "orientation `rotation`); row_from_state (the hold row the official stay() builds from a state "
            "vector). Poses reply with row_prefix: position then rotation vector, the flange part of a row. "
            "near_rotation_vector selects the equivalent rotation vector closest to it, so consecutive rows do "
            "not jump across the angle-pi boundary. Pure geometry: needs no episode and reads no scene state."),
         "inputSchema": {"type": "object", "required": ["operation"], "additionalProperties": False, "properties": {
             "operation": {"type": "string", "enum": list(RM.OPERATIONS)},
             "rotation": _ROTATION, "pose_a": _POSE, "pose_b": _POSE,
             "point_target_m": {**_vector(3), "description": "flange_target: world position the hand point must reach"},
             "point_offset_m": {**_vector(3), "description": "flange_target: the hand point in the flange frame (m)"},
             "near_rotation_vector": {**_vector(3), "description": "e.g. the previous row's rotation vector"},
             "state": {"type": "array", "items": _NUMBER, "minItems": 23, "maxItems": 46,
                       "description": "row_from_state: the official state vector (23 or 46 values)"}}}},
        {"name": PREVIEW_DESCRIBE, "description": (
            "Contract of the robot-only motion preview: robot model, control, cameras, inputs, outputs and limits. "
            "Starts nothing."),
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": PREVIEW_RUN, "description": (
            "Dry-run dexjoco_step rows on a robot-only simulator: this task's Panda + Allegro model under the "
            f"official control (the same flange controller and gains, {CONTROL_TEXT}, the same hand servos) on a "
            "floor plane, watched by fixed third-person cameras. No table, objects or contact with the scene. The "
            "start defaults to the episode's measured state vector; initial_state (an official state vector) "
            "overrides it. Each run first drives the arm from its home posture to the start flange pose (not shown; "
            "reported as start_settle). Returns the start and end state vectors, saves every "
            f"{FRAME_STRIDE}th row as a frame and attaches the final frames. Never moves the episode or spends its "
            "steps."),
         "inputSchema": {"type": "object", "required": ["actions"], "additionalProperties": False, "properties": {
             "actions": {"type": "array", "minItems": 1, "maxItems": MAX_ACTION_ROWS,
                         "items": {"type": "array", "items": _NUMBER, "minItems": 22, "maxItems": 44}},
             "initial_state": {"type": "array", "items": _NUMBER, "minItems": 23, "maxItems": 46,
                               "description": "Official state vector (23 single arm, 46 dual arm); default: the "
                                              "episode's measured state."}}}},
    ]


CONTROL_TEXT = "30 Hz, 20 ms of physics per step"


def object_states_description(level=None):
    """dexjoco_object_states' text at the privileged level (RAX_PRIVILEGED_STATE; objects.py)."""
    from services.dexjoco.objects import privileged_level

    level = level or privileged_level()
    text = ("Ground-truth state of the current episode read from the simulator: the world-frame pose (position_m, "
            "quaternion_wxyz) of each task object and fixture, named object points (position_m), articulation "
            "angles (angle_rad), and the arm joint positions (rad, joint1..7 per arm). The hand joints and flange "
            "poses are in the official state vector. Exact at the current step_id; no physics")
    if level != "full":
        return text + ", no success information."
    return text + ". privileged_level names the level (full). " + (
        "Each free or body object also has linear_velocity_mps and angular_velocity_radps (world frame) and size "
        "(half_extents_m and center_offset_m: its collision geoms' bounding box in its own frame). contacts: the "
        "current contacts on task objects and the hands per pair (a, b: object label or MuJoCo body name; "
        "normal_force_n, count, position_m), strongest first. goal: the task's official success specification: "
        "description, conditions and failure_conditions (name, satisfied, measured, threshold, note), targets, "
        "holding (steps_satisfied of steps_required consecutive steps) where the task needs a hold, and "
        "satisfied_now (every condition holds at this step); {\"available\": false} for tasks without one.")
_NUMBER = {"type": "number"}


def _vector(n):
    return {"type": "array", "items": _NUMBER, "minItems": n, "maxItems": n}


_ROTATION = {
    "type": "object", "minProperties": 1, "maxProperties": 1, "additionalProperties": False,
    "description": "Exactly one representation (world frame unless a pose says otherwise).",
    "properties": {
        "quaternion_wxyz": _vector(4),
        "rotation_vector": {**_vector(3), "description": "axis times angle (rad), as in action rows"},
        "matrix": {"type": "array", "items": _vector(3), "minItems": 3, "maxItems": 3, "description": "rows of R"},
        "axis_angle": {"type": "object", "required": ["axis", "angle_rad"], "additionalProperties": False,
                       "properties": {"axis": _vector(3), "angle_rad": _NUMBER}}}}
_POSE = {"type": "object", "minProperties": 1, "additionalProperties": False,
         "description": "position_m (m) and rotation; a missing position is zero, a missing rotation the identity.",
         "properties": {"position_m": _vector(3), "rotation": _ROTATION}}


def _only(arguments, allowed):
    unknown = set(arguments) - set(allowed)
    reject_unless(not unknown, f"Unknown arguments: {sorted(unknown)}")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_actions(actions, dim=None):
    reject_unless(isinstance(actions, list) and 1 <= len(actions) <= MAX_ACTION_ROWS,
                  f"actions must be a list of 1..{MAX_ACTION_ROWS} rows")
    width = len(actions[0]) if isinstance(actions[0], list) else None
    reject_unless(width in (22, 44), "each action row must have 22 (single arm) or 44 (dual arm) numbers")
    if dim is not None:
        reject_unless(width == dim, f"this task takes {dim}-D action rows, got {width}")
    for row in actions:
        reject_unless(isinstance(row, list) and len(row) == width, "all action rows must have the same length")
        reject_unless(all(_finite(v) for v in row), "action rows must contain finite numbers")


def validate_state(state, dim):
    """An official state vector of this robot: finite, with non-zero flange quaternions."""
    reject_unless(isinstance(state, list) and len(state) == dim and all(_finite(v) for v in state),
                  f"initial_state must be the official {dim}-D state vector of this robot (dexjoco_status state_layout)")
    for start in ((3,) if dim == 23 else (3, 10)):
        reject_unless(math.hypot(*state[start:start + 4]) > 1e-6,
                      f"initial_state[{start}:{start + 4}] must be a non-zero flange quaternion [w, x, y, z]")


def validate_request(name, arguments, *, action_dim=None):
    """Complete schema/shape validation before dispatch; raises RejectedBeforeAction."""
    reject_unless(isinstance(arguments, dict), "arguments must be an object")
    if name in (STATUS, OBSERVE, OBJECT_STATES):
        _only(arguments, ())
    elif name == STEP:
        _only(arguments, ("actions",))
        validate_actions(arguments.get("actions"), action_dim)
    elif name == HOLD:
        _only(arguments, ("steps",))
        steps = arguments.get("steps")
        reject_unless(type(steps) is int and 1 <= steps <= MAX_ACTION_ROWS, f"steps must be an integer 1..{MAX_ACTION_ROWS}")
    elif name == ROTATION_MATH:
        RM.validate(arguments)
    elif name == PREVIEW_DESCRIBE:
        _only(arguments, ())
    elif name == PREVIEW_RUN:
        _only(arguments, ("actions", "initial_state"))
        validate_actions(arguments.get("actions"), action_dim)
        if "initial_state" in arguments:
            validate_state(arguments["initial_state"], 46 if len(arguments["actions"][0]) == 44 else 23)
    elif name == PIXEL_TO_POSITION:
        _only(arguments, ("observation_step_id", "selections", "depth_window_radius"))
        reject_unless(type(arguments.get("observation_step_id")) is int, "observation_step_id must be an integer")
        selections = arguments.get("selections")
        reject_unless(isinstance(selections, list) and 1 <= len(selections) <= 32, "selections must have 1..32 items")
        for item in selections:
            reject_unless(isinstance(item, dict) and set(item) == {"camera", "u", "v"}, "selection needs camera, u, v")
            reject_unless(item["camera"] in CAMERAS, f"unknown camera {item['camera']!r}")
            reject_unless(_finite(item["u"]) and _finite(item["v"]), "pixel coordinates must be finite numbers")
        radius = arguments.get("depth_window_radius", 1)
        reject_unless(type(radius) is int and 0 <= radius <= 5, "depth_window_radius must be 0..5")
    else:
        raise RejectedBeforeAction(f"Unknown DexJoCo tool: {name}")


def quaternion_matrix(q_wxyz):
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pixel_to_world(depth_m, intrinsics, pose, u, v, radius=1):
    """Back-project one pixel of a linear-depth image into the world frame.

    MuJoCo cameras look down their -Z axis with +Y up (OpenGL); images are row 0 at the top.
    Depth is the official linearised z-buffer: distance along the optical axis. Uses the
    median valid depth in a (2r+1)^2 window.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    height, width = depth.shape[:2]
    reject_unless(0 <= u < width and 0 <= v < height, f"pixel ({u}, {v}) outside {width}x{height}")
    iu, iv = int(round(u)), int(round(v))
    window = depth[max(0, iv - radius):iv + radius + 1, max(0, iu - radius):iu + radius + 1]
    valid = window[np.isfinite(window) & (window > 0)]
    if valid.size == 0:
        return {"valid": False, "reason": "no valid depth near the pixel"}
    z = float(np.median(valid))
    K = np.asarray(intrinsics["matrix"], dtype=np.float64)
    point_camera = np.array([(u - K[0, 2]) / K[0, 0] * z, -(v - K[1, 2]) / K[1, 1] * z, -z])
    point = quaternion_matrix(pose["quaternion_wxyz"]) @ point_camera + np.asarray(pose["position_m"], dtype=np.float64)
    return {"valid": True, "position_m": point.tolist(), "depth_m": z, "frame": "world",
            "depth_spread_m": float(valid.max() - valid.min())}


def world_to_pixel(point, intrinsics, pose):
    """Project a world-frame point into a camera (inverse of pixel_to_world)."""
    local = quaternion_matrix(pose["quaternion_wxyz"]).T @ (np.asarray(point, dtype=np.float64)
                                                            - np.asarray(pose["position_m"], dtype=np.float64))
    if local[2] >= 0:
        return None
    z = -local[2]
    K = np.asarray(intrinsics["matrix"], dtype=np.float64)
    return float(K[0, 0] * local[0] / z + K[0, 2]), float(-K[1, 1] * local[1] / z + K[1, 2]), float(z)
