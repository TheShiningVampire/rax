"""Experiment (exp/add-state-info/lookahead): exact lookahead on the live DexJoCo episode.

``dexjoco_lookahead`` simulates official action rows from the current live state in the same
environment object and MjModel, on an exact copy of the live MjData, reads the object states at
the end (and every ``report_every`` rows), then restores everything so that a later real step is
bit-identical to one that never looked ahead. Design and measurements:
rax-reproduction/notes/lookahead/DESIGN.md.

This module holds the snapshot, restore and fingerprint logic and the rollout loop. It is pure
Python + numpy at import time (the host reads the settings below); ``mujoco`` is imported only
by the functions that touch MjData.

The rows run on a full copy of the live MjData (``copy.copy``: mj_copyData, sharing the MjModel)
that the raw environment holds as ``_data`` for the duration of the lookahead; the live MjData
object is never written, so every reader that keeps it (the renderers, the session's tools) sees
exactly the pre-lookahead fields afterwards, derived fields and contacts included. This is a
deliberate departure from DESIGN.md 1.1 (restore in place): an in-place restore cannot restore the
MjData arena layout. The efc_*/island views are arena pointers set by the last mj_step, so after a
lookahead whose contact count differs, copying the snapshot's arrays and contacts back through the
live views leaves the constraint arrays aliased differently (test_dexjoco_lookahead_physics.py
shows the exactness check catching it). The physics of the next real step was still bit-identical there, but
mj_contactForce (object_states contacts) would read the wrong efc_force rows until it.

Restore, in this order: the live MjData object back into the environment, the task's MjModel
fields, the Python attributes of the raw environment, the gym wrappers, the official wrapper and
the session's Tap, then the random number generators. A fingerprint of all of it, taken before
and after, must match (DESIGN.md section 4).
"""

from __future__ import annotations

import copy
import hashlib
import io
import os
import random
import signal
import threading
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

TOOL = "dexjoco_lookahead"
LOOKAHEAD_ENV = "RAX_LOOKAHEAD"
LOOKAHEAD_VALUES = ("on", "off")
DEFAULT_LOOKAHEAD = "on"
ROWS_FACTOR_ENV = "RAX_LOOKAHEAD_ROWS_FACTOR"
DEFAULT_ROWS_FACTOR = 3
# Debug: "full" also names every differing component of a failed exactness check.
VERIFY_ENV = "RAX_LOOKAHEAD_VERIFY"
# Settings the launcher's frontend and the exploration worker keep (objects.SESSION_KEPT_ENV).
KEPT_ENV = frozenset({LOOKAHEAD_ENV, ROWS_FACTOR_ENV, VERIFY_ENV})

CALLS_PER_EPISODE = 300
WALL_SECONDS = 30.0
REPORT_EVERY_MIN = 5
MAX_REPORTS = 15
OFF_MESSAGE = (f"{TOOL} is off in this session ({LOOKAHEAD_ENV}=off); "
               "use dexjoco_step to execute rows")

STOP_EPISODE_END, STOP_ILLEGAL, STOP_WALL = "episode_end", "illegal_move", "wall_time"


# ------------------------------------------------------------------ settings
def lookahead_setting(environ=None):
    """RAX_LOOKAHEAD: ``on`` (the default on this branch) or ``off``. Read the same way by the host
    (tool list, tool text, workspace) and the simulator (the op)."""
    value = (os.environ if environ is None else environ).get(LOOKAHEAD_ENV, DEFAULT_LOOKAHEAD)
    if value not in LOOKAHEAD_VALUES:
        raise ValueError(f"{LOOKAHEAD_ENV} must be one of {LOOKAHEAD_VALUES}, not {value!r}")
    return value


def lookahead_enabled(environ=None):
    return lookahead_setting(environ) == "on"


def rows_factor(environ=None):
    """RAX_LOOKAHEAD_ROWS_FACTOR: lookahead rows per episode as a multiple of the native step limit."""
    value = (os.environ if environ is None else environ).get(ROWS_FACTOR_ENV, str(DEFAULT_ROWS_FACTOR))
    try:
        factor = int(value)
    except (TypeError, ValueError):
        factor = 0
    if not 1 <= factor <= 100:
        raise ValueError(f"{ROWS_FACTOR_ENV} must be an integer 1..100, not {value!r}")
    return factor


def rows_budget(native_step_limit, environ=None):
    return rows_factor(environ) * int(native_step_limit)


def limits(native_step_limit, environ=None):
    return {"rows_per_episode": rows_budget(native_step_limit, environ), "calls_per_episode": CALLS_PER_EPISODE,
            "seconds_per_call": WALL_SECONDS}


class LookaheadRestoreError(RuntimeError):
    """The live state after a lookahead differs from the state before it; the episode is invalid."""


# --------------------------------------------------------- per-task knowledge
# Raw-environment attributes each trial task's step() changes (DESIGN.md 1.3; the unit test scans
# the official source for them). They must be found and snapshotted, not merely identity-checked.
REQUIRED_ATTRS = {
    "water_plant": ("env_step", "_trigger_pulled", "_cone_alpha_counter", "_success_counter"),
    "fold_glasses": ("env_step", "_success_trigger_count"),
    "bimanual_microwave_cook": ("env_step",),
    "bimanual_photograph": ("env_step", "_latest_shutter_pressed"),
    "bimanual_hanoi": ("env_step", "_success_trigger_count", "_illegal_trigger_count", "_illegal_trigger_reason",
                       "_pending_illegal_reset", "_pending_illegal_deadline", "_pending_illegal_info"),
}
# MjModel fields a task's step() writes (visual only); snapshotted and restored.
MODEL_FIELDS = {
    "water_plant": ("geom_rgba", "geom_group"),
    "click_mouse": ("geom_rgba", "mat_rgba"),
}
# MjModel fields hashed before and after (checked, not restored): catches a model write that slips in.
MODEL_FINGERPRINT_FIELDS = ("body_pos", "body_quat", "body_mass", "geom_size", "geom_rgba", "geom_matid",
                            "geom_friction", "dof_damping", "actuator_gainprm", "actuator_biasprm", "cam_pos",
                            "cam_quat", "light_pos", "light_diffuse")

# Arena memory of the dense Newton path: differs between two fresh identical runs; copied, not hashed.
ARENA_NOISE = frozenset({"efc_AR_colind", "efc_AR_rowadr", "efc_AR_rownnz", "iefc_J_colind", "iefc_J_rowadr",
                         "iefc_J_rowsuper"})
READONLY_SCALARS = frozenset({"signature"})
STAT_LISTS = ("warning", "solver", "timer")

EXCLUDE_NAMES = frozenset({"_model", "_data", "model", "data", "_viewer", "_mj_viewer", "viewer", "renderer",
                           "_renderer", "env", "_env", "_random", "np_random", "_np_random", "observation_space",
                           "action_space", "spec", "metadata", "_action_space", "_observation_space", "_metadata",
                           "_cached_spec", "_np_random_seed"})
IMMUTABLE = (type(None), bool, int, float, complex, str, bytes, np.generic)
_EXCLUDED_TYPES = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType,
                   type(threading.Lock()), type(threading.RLock()), threading.Thread, io.IOBase)


def guard_reason(raw, env_name):
    """Why a lookahead cannot run from this live state (DESIGN.md 2.1), or None."""
    if getattr(raw, "render_mode", None) == "human":
        return "the environment renders to a window (render_mode human)"
    if getattr(raw, "reset_trigger", False):
        return "the environment has a pending reset (reset_trigger)"
    if getattr(raw, "_pending_illegal_reset", False):
        return "an illegal move is pending: the environment resets itself on its next step"
    if getattr(raw, "_camera_screen_effect", False) or getattr(raw, "camera_screen_effect", False):
        return "the camera screen effect writes the model's texture each step"
    for name in REQUIRED_ATTRS.get(env_name, ()):
        if not hasattr(raw, name):
            return f"the environment has no {name} attribute to snapshot"
    return None


# ---------------------------------------------------------- canonical hashing
def _feed(h, value, depth=0):
    if depth > 12:
        h.update(f"<depth {id(value)}>".encode())
    elif isinstance(value, np.ndarray):
        h.update(f"nd|{value.dtype.str}|{value.shape}|".encode())
        if value.dtype.hasobject:
            for item in value.ravel():
                _feed(h, item, depth + 1)
        else:
            h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, IMMUTABLE):
        h.update(f"{type(value).__name__}|{value!r}|".encode())
    elif isinstance(value, dict):
        h.update(f"dict{len(value)}|".encode())
        for key in sorted(value, key=repr):
            _feed(h, key, depth + 1)
            _feed(h, value[key], depth + 1)
    elif isinstance(value, (list, tuple)):
        h.update(f"{type(value).__name__}{len(value)}|".encode())
        for item in value:
            _feed(h, item, depth + 1)
    elif isinstance(value, (set, frozenset)):
        h.update(f"set{len(value)}|".encode())
        for item in sorted(value, key=repr):
            _feed(h, item, depth + 1)
    else:
        h.update(f"<{type(value).__qualname__} {id(value)}>".encode())


def digest(value):
    h = hashlib.blake2b(digest_size=16)
    _feed(h, value)
    return h.hexdigest()


def _bytes_digest(*arrays):
    h = hashlib.blake2b(digest_size=16)
    for array in arrays:
        _feed(h, np.asarray(array))
    return h.hexdigest()


# ------------------------------------------------------------------ MjData
@dataclass
class MjLayout:
    scalars: tuple
    arrays: tuple
    contact: tuple
    stats: dict


_LAYOUT = None


def mjdata_layout(d):
    """The MjData attribute names by kind, found once by introspection (mujoco 3.4.0: 26 scalars,
    164 ndarrays, the contact list's 19 fields and the warning/solver/timer statistics)."""
    global _LAYOUT
    if _LAYOUT is None:
        scalars, arrays = [], []
        for name in dir(d):
            if name.startswith("_") or name in READONLY_SCALARS:
                continue
            try:
                value = getattr(d, name)
            except Exception:  # noqa: BLE001 - not a data attribute
                continue
            if isinstance(value, np.ndarray):
                arrays.append(name)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                scalars.append(name)
        contact = tuple(n for n in dir(d.contact) if not n.startswith("_")
                        and isinstance(getattr(d.contact, n), np.ndarray))
        stats = {name: tuple(n for n in dir(getattr(d, name)) if not n.startswith("_")
                             and isinstance(getattr(getattr(d, name), n), np.ndarray)) for name in STAT_LISTS}
        _LAYOUT = MjLayout(tuple(scalars), tuple(arrays), contact, stats)
    return _LAYOUT


def copy_mjdata(d):
    """A full independent copy of an MjData sharing its MjModel (mj_copyData: arena, contacts and
    derived fields included; the bindings expose it as copy.copy)."""
    return copy.copy(d)


def restore_mjdata(m, d, snap):
    """In-place restore of ``d`` from ``snap`` (DESIGN.md 1.1): the INTEGRATION state, every scalar
    (before the arrays, so the variable-size views take the snapshot's shape), every ndarray field,
    the contacts and the statistics. Raises on a shape mismatch. Not used by the tool (see the
    module docstring): kept for the unit test that shows why."""
    import mujoco

    layout = mjdata_layout(d)
    mujoco.mj_copyState(m, snap, d, mujoco.mjtState.mjSTATE_INTEGRATION)
    for name in layout.scalars:
        setattr(d, name, getattr(snap, name))
    for name in layout.arrays:
        target, source = getattr(d, name), getattr(snap, name)
        if target.shape != source.shape:
            raise LookaheadRestoreError(f"MjData.{name} has shape {target.shape}, the snapshot {source.shape}")
        np.copyto(target, source)
    _copy_fields(d.contact, snap.contact, layout.contact, "contact")
    for name, fields in layout.stats.items():
        _copy_fields(getattr(d, name), getattr(snap, name), fields, name)


def _copy_fields(target, source, fields, label):
    for name in fields:
        to, src = getattr(target, name), getattr(source, name)
        if to.shape != src.shape:
            raise LookaheadRestoreError(f"MjData.{label}.{name} has shape {to.shape}, the snapshot {src.shape}")
        to[...] = src


def mjdata_fingerprint(m, d):
    """Component digests of the live MjData (minus the arena noise and the high-water marks)."""
    import mujoco

    layout = mjdata_layout(d)
    sig = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(m, sig))
    mujoco.mj_getState(m, d, state, sig)
    out = {"mj.state": _bytes_digest(state)}
    out["mj.scalars"] = digest([(n, getattr(d, n)) for n in layout.scalars
                                if n != "threadpool" and not n.startswith("maxuse_")])
    for name in layout.arrays:
        if name not in ARENA_NOISE:
            out[f"mj.{name}"] = _bytes_digest(getattr(d, name))
    out["mj.contact"] = _bytes_digest(*(getattr(d.contact, n) for n in layout.contact))
    for name, fields in layout.stats.items():
        out[f"mj.{name}"] = _bytes_digest(*(getattr(getattr(d, name), n) for n in fields))
    return out


# ------------------------------------------------------------------ MjModel
def snapshot_model(model, env_name):
    return {name: np.copy(getattr(model, name)) for name in MODEL_FIELDS.get(env_name, ())}


def restore_model(model, saved):
    for name, values in saved.items():
        np.copyto(getattr(model, name), values)


def model_fingerprint(model, env_name):
    names = sorted({*MODEL_FINGERPRINT_FIELDS, *MODEL_FIELDS.get(env_name, ())})
    out = {f"model.{n}": _bytes_digest(getattr(model, n)) for n in names if hasattr(model, n)}
    out["model.opt.timestep"] = digest(float(model.opt.timestep))
    return out


# ------------------------------------------------------- Python attributes
def _allowed_leaves(value, depth=0):
    if depth > 8:
        return False
    if isinstance(value, (IMMUTABLE, np.ndarray)):
        return not (isinstance(value, np.ndarray) and value.dtype.hasobject)
    if isinstance(value, dict):
        return all(_allowed_leaves(k, depth + 1) and _allowed_leaves(v, depth + 1) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_allowed_leaves(v, depth + 1) for v in value)
    return False


def classify(name, value):
    """``excluded`` (identity only), ``value``, ``array``, ``container`` or ``other`` (identity only,
    listed for review). Renderers, GL contexts, MuJoCo and gym objects are never copied."""
    kind = type(value)
    module = kind.__module__ or ""
    if (name in EXCLUDE_NAMES or module.startswith(("mujoco", "gymnasium", "gym."))
            or any(word in kind.__name__ for word in ("Render", "Viewer", "Context"))
            or isinstance(value, _EXCLUDED_TYPES) or callable(value) and not isinstance(value, np.ndarray)):
        return "excluded"
    if isinstance(value, tuple) and all(isinstance(v, IMMUTABLE) for v in value):
        return "value"
    if isinstance(value, IMMUTABLE):
        return "value"
    if isinstance(value, np.ndarray):
        return "array" if not value.dtype.hasobject else "other"
    if isinstance(value, (list, dict, set, tuple)) and _allowed_leaves(value):
        return "container"
    return "other"


@dataclass
class AttrSnapshot:
    label: str
    obj: object
    names: frozenset
    saved: dict = field(default_factory=dict)  # name -> (kind, reference, copy)
    other: dict = field(default_factory=dict)  # name -> type name (identity-checked only)

    def kind(self, name):
        return self.saved[name][0] if name in self.saved else None


def snapshot_attrs(label, obj):
    names = frozenset(vars(obj))
    snap = AttrSnapshot(label, obj, names)
    for name in names:
        value = vars(obj)[name]
        kind = classify(name, value)
        if kind == "value":
            snap.saved[name] = (kind, value, None)
        elif kind == "array":
            snap.saved[name] = (kind, value, value.copy())
        elif kind == "container":
            snap.saved[name] = (kind, value, copy.deepcopy(value))
        else:
            snap.saved[name] = (kind, value, None)
            if kind == "other":
                snap.other[name] = type(value).__qualname__
    return snap


def restore_attrs(snap):
    obj = snap.obj
    for name in set(vars(obj)) - snap.names:  # Created during the lookahead.
        delattr(obj, name)
    for name, (kind, ref, saved) in snap.saved.items():
        if kind == "array":
            if ref.shape != saved.shape or ref.dtype != saved.dtype:
                raise LookaheadRestoreError(f"{snap.label}.{name} changed shape or dtype in place")
            if ref.tobytes() != saved.tobytes():
                np.copyto(ref, saved)
        elif kind == "container" and digest(ref) != digest(saved):
            fresh = copy.deepcopy(saved)
            if isinstance(ref, dict):
                ref.clear()
                ref.update(fresh)
            elif isinstance(ref, list):
                ref[:] = fresh
            elif isinstance(ref, set):
                ref.clear()
                ref.update(fresh)
            else:  # A tuple holding arrays: rebuild it.
                ref = fresh
        if vars(obj).get(name, _MISSING) is not ref:
            setattr(obj, name, ref)


_MISSING = object()


def attrs_fingerprint(snapshots):
    """Each object's attribute names and each value's canonical hash (identity for excluded ones)."""
    out = {}
    for snap in snapshots:
        obj = snap.obj
        out[f"py.{snap.label}.__names__"] = digest(sorted(vars(obj)))
        for name, value in vars(obj).items():
            kind = classify(name, value)
            out[f"py.{snap.label}.{name}"] = (digest(value) if kind in ("value", "array", "container")
                                              else f"id:{id(value)}")
    return out


def env_chain(session):
    """(label, object) for every object a step or its bookkeeping can change: the session's Tap,
    the official wrapper, each gym wrapper below the Tap and the raw environment."""
    raw = session.tap.unwrapped
    chain = [("tap", session.tap), ("wrapper", session.env)]
    layer, depth = session.tap._env, 0
    while layer is not raw:
        chain.append((f"gym{depth}", layer))
        layer, depth = layer.env, depth + 1
        if depth > 16:
            raise RuntimeError("the wrapper chain does not reach the raw environment")
    chain.append(("raw", raw))
    return chain


# ------------------------------------------------------------------ RNG
def snapshot_rng(raw):
    state = {"numpy": np.random.get_state(), "random": random.getstate()}
    generator = getattr(raw, "_random", None)
    if isinstance(generator, np.random.RandomState):
        state["raw"] = generator.get_state()
    return state


def restore_rng(raw, state):
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])
    if "raw" in state:
        raw._random.set_state(state["raw"])


def rng_fingerprint(raw):
    return {f"rng.{k}": digest(v) for k, v in snapshot_rng(raw).items()}


# ------------------------------------------------------------------ session
SESSION_FIELDS = ("step_id", "terminated", "truncated", "success", "in_stay", "invalid_reason", "finish_reason",
                  "episode_id", "_video_frames")


def session_fingerprint(session):
    values = {name: getattr(session, name, None) for name in SESSION_FIELDS}
    values["id(obs)"], values["id(scene)"] = id(session.obs), id(session.scene)
    return {f"session.{k}": digest(v) for k, v in values.items()}


@dataclass
class Snapshot:
    live: object  # The live MjData object, put back into the environment on restore.
    model: dict
    attrs: list
    rng: dict
    overrides: dict


def take_snapshot(session, env_name):
    raw = session.tap.unwrapped
    attrs = [snapshot_attrs(label, obj) for label, obj in env_chain(session)]
    for name in REQUIRED_ATTRS.get(env_name, ()):
        kind = attrs[-1].kind(name)
        if kind not in ("value", "array", "container"):
            raise LookaheadRestoreError(f"required attribute raw.{name} cannot be snapshotted (kind {kind})")
    overrides = {name: getattr(raw, name) for name in ("image_obs", "hz") if hasattr(raw, name)}
    return Snapshot(raw._data, snapshot_model(raw._model, env_name), attrs, snapshot_rng(raw), overrides)


def restore_snapshot(session, snap):
    raw = session.tap.unwrapped
    raw._data = snap.live
    restore_model(raw._model, snap.model)
    for attrs in snap.attrs:
        restore_attrs(attrs)
    restore_rng(raw, snap.rng)


def fingerprint(session, env_name, snapshots):
    raw = session.tap.unwrapped
    return {**mjdata_fingerprint(raw._model, raw._data), **model_fingerprint(raw._model, env_name),
            **attrs_fingerprint(snapshots), **rng_fingerprint(raw), **session_fingerprint(session)}


def differing(before, after):
    return sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))


@contextmanager
def deferred_signals(signals=(signal.SIGALRM,)):
    """Hold the episode wall-clock alarm while the live state is being restored and verified; a
    deferred alarm fires right after. Python runs signal handlers in the main thread only, so this
    swaps the handler there (elsewhere it does nothing)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    pending, previous = [], {}

    def hold(signum, frame):
        pending.append((signum, frame))

    for signum in signals:
        previous[signum] = signal.signal(signum, hold)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        for signum, frame in pending:
            handler = previous.get(signum)
            if callable(handler):
                handler(signum, frame)


# ------------------------------------------------------------------ rollout
@dataclass
class Outcome:
    rows_simulated: int
    stop_reason: str | None
    end_row: int | None
    env_step: int | None  # The raw environment's step counter at the last simulated row.
    illegal_reason: str | None
    reports: list
    final: dict | None
    rollout_s: float
    total_s: float


def run(session, rows, *, env_name, report_every, report, progress=None, clock=time.monotonic,
        wall_seconds=WALL_SECONDS):
    """Simulate ``rows`` (official rotation-vector rows) from the live state and restore it.

    ``report(row_index, obs)`` reads what the reply shows (called before the restore, while the
    environment holds the lookahead's MjData). The rows go through the official conversion (``DexJoCoOpenPIEnv._process_action``) into the gym chain
    below the session's Tap, with rendering and the wall-clock pacing off. ``progress["rows"]``
    counts the simulated rows, also when a row raises. Raises ``LookaheadRestoreError`` when the
    restored live state differs from the one before."""
    started = clock()
    progress = {} if progress is None else progress
    progress["rows"] = 0
    raw = session.tap.unwrapped
    snap = take_snapshot(session, env_name)
    before = fingerprint(session, env_name, snap.attrs)
    reports, final, stop, end_row, env_step, illegal = [], None, None, None, None, None
    index, rollout_s, failure = 0, 0.0, None
    try:
        try:
            if "image_obs" in snap.overrides:
                raw.image_obs = False
            if "hz" in snap.overrides:
                raw.hz = float("inf")
            raw._data = copy_mjdata(snap.live)
            t0 = clock()
            obs = None
            for index, row in enumerate(rows, 1):
                obs, _reward, terminated, truncated, _info = session.tap._env.step(session.env._process_action(row))
                progress["rows"] = index
                if terminated or truncated:
                    stop, end_row = STOP_EPISODE_END, index
                elif getattr(raw, "_pending_illegal_reset", False):
                    stop, end_row = STOP_ILLEGAL, index
                    illegal = getattr(raw, "_illegal_trigger_reason", None)
                elif clock() - t0 > wall_seconds and index < len(rows):
                    stop = STOP_WALL
                if stop:
                    break
                if report_every and index % report_every == 0 and index < len(rows) and len(reports) < MAX_REPORTS:
                    reports.append(report(index, obs))
            rollout_s = clock() - t0
            env_step = getattr(raw, "env_step", None)
            final = report(index, obs)
        finally:
            raw._data = snap.live
            for name, value in snap.overrides.items():
                setattr(raw, name, value)
    except BaseException as exc:  # noqa: BLE001 - restored and verified below, then re-raised
        failure = exc
    with deferred_signals():
        try:
            restore_snapshot(session, snap)
        except Exception as exc:  # noqa: BLE001
            raise LookaheadRestoreError(f"restoring the live state failed: {type(exc).__name__}: {exc}") from (
                failure or exc)
        after = fingerprint(session, env_name, snap.attrs)
        diff = differing(before, after)
        if diff:
            shown = diff if os.environ.get(VERIFY_ENV) == "full" else diff[:12]
            raise LookaheadRestoreError(f"live state differs after the lookahead in {len(diff)} components: "
                                        + ", ".join(shown)) from failure
    if failure is not None:
        raise failure
    return Outcome(index, stop, end_row, env_step, illegal, reports, final, rollout_s, clock() - started)
