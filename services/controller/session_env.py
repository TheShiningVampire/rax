"""The session's own RAX_* settings (objects.SESSION_KEPT_ENV), recorded at its first launch.

The settings change what the agent is offered (RAX_LOOKAHEAD adds a tool and its skill text,
RAX_PRIVILEGED_STATE the object-state level), so a session keeps the values it started with: a
resume, a post-hoc formal batch and a parallel batch adopt the recorded values instead of taking
them from whatever shell runs them. ``session-env.json`` sits in the session root beside
``state.json``, outside the agent's view.

A variable the shell leaves unset takes the recorded value; one the shell sets to a different
value is a conflict and the launch is refused (a different setting is a different experiment).
A session launched before the record existed has none and is left as it is.
"""
import json
import os
from pathlib import Path

from services.dexjoco.objects import SESSION_KEPT_ENV

FILE = 'session-env.json'


class SessionEnvConflict(RuntimeError):
    """The shell sets a session setting to a value other than the session's recorded one."""


def current(environ=None):
    environ = os.environ if environ is None else environ
    return {name: environ.get(name) for name in sorted(SESSION_KEPT_ENV)}


def recorded(root):
    path = Path(root)/FILE
    if not path.is_file():
        return None
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or not all(isinstance(v, (str, type(None))) for v in values.values()):
        raise ValueError(f'{path} is not a record of session settings')
    return values


def record(root, environ=None):
    """Record the settings of a first launch (kept if already recorded); returns the record."""
    from .storage import persist

    existing = recorded(root)
    if existing is not None:
        return existing
    values = current(environ)
    persist(Path(root)/FILE, values)
    return values


def adopt(root, environ=None):
    """Apply ``root``'s recorded settings to ``environ`` (default ``os.environ``); returns the names
    it set or removed, or None when the session has no record. Raises ``SessionEnvConflict`` when
    the environment sets one to another value."""
    environ = os.environ if environ is None else environ
    values = recorded(root)
    if values is None:
        return None
    conflicts = [f'{name}={environ[name]!r} (the session ran with '
                 f'{"it unset" if value is None else repr(value)})'
                 for name, value in sorted(values.items()) if name in environ and environ[name] != value]
    if conflicts:
        raise SessionEnvConflict(f'{Path(root)/FILE}: this shell sets ' + '; '.join(conflicts)
                                 + '. Unset these variables (the session\'s values are then used) or match them.')
    changed = []
    for name, value in sorted(values.items()):
        if value is not None and name not in environ:
            environ[name] = value
            changed.append(name)
    return changed
