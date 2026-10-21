"""A session keeps the RAX_* settings it first launched with (services/controller/session_env.py)."""
import json
import sys

import pytest

from test_controller_backend import configuration
from services.controller import session_env
from services.dexjoco.objects import SESSION_KEPT_ENV


def test_the_record_holds_every_kept_setting_and_is_written_once(tmp_path):
    first = session_env.record(tmp_path, {"RAX_LOOKAHEAD": "off", "RAX_OTHER": "x"})
    assert set(first) == set(SESSION_KEPT_ENV) and first["RAX_LOOKAHEAD"] == "off"
    assert first["RAX_PRIVILEGED_STATE"] is None and "RAX_OTHER" not in first
    assert session_env.record(tmp_path, {"RAX_LOOKAHEAD": "on"}) == first  # A later launch keeps it.
    assert json.loads((tmp_path/session_env.FILE).read_text()) == first


def test_adopt_fills_unset_settings_and_refuses_a_different_one(tmp_path):
    assert session_env.adopt(tmp_path, {}) is None  # A session from before the record: left as it is.
    session_env.record(tmp_path, {"RAX_LOOKAHEAD": "off", "RAX_PRIVILEGED_STATE": "objects"})
    environ = {"RAX_PRIVILEGED_STATE": "objects"}
    assert session_env.adopt(tmp_path, environ) == ["RAX_LOOKAHEAD"]
    assert environ["RAX_LOOKAHEAD"] == "off" and "RAX_LOOKAHEAD_ROWS_FACTOR" not in environ
    with pytest.raises(session_env.SessionEnvConflict, match="RAX_LOOKAHEAD='on'"):
        session_env.adopt(tmp_path, {"RAX_LOOKAHEAD": "on"})
    with pytest.raises(session_env.SessionEnvConflict, match="RAX_LOOKAHEAD_ROWS_FACTOR='5' .*it unset"):
        session_env.adopt(tmp_path, {"RAX_LOOKAHEAD_ROWS_FACTOR": "5"})


def _launcher(tmp_path, monkeypatch):
    from harness.codex_cli import auto_research
    config = configuration(tmp_path)
    source = tmp_path/'input.json'
    source.write_text('{}')
    monkeypatch.setattr(auto_research.Configuration, 'from_dict', lambda _: config)
    monkeypatch.setattr(auto_research.Configuration, '__post_init__', lambda _: None)
    monkeypatch.setattr(auto_research.signal, 'signal', lambda *a: None)
    for name in SESSION_KEPT_ENV:
        monkeypatch.delenv(name, raising=False)
    return auto_research, config, source


class Deployed(Exception):
    pass


def test_a_first_launch_records_and_a_resume_adopts_the_settings(tmp_path, monkeypatch):
    auto_research, config, source = _launcher(tmp_path, monkeypatch)
    seen = []

    def deploy(*args, **kwargs):
        seen.append({name: auto_research.os.environ.get(name) for name in SESSION_KEPT_ENV})
        raise Deployed()

    monkeypatch.setattr(auto_research, 'deploy', deploy)
    monkeypatch.setattr(auto_research, 'redeploy', deploy)
    monkeypatch.setenv('RAX_LOOKAHEAD', 'off')
    monkeypatch.setattr(sys, 'argv', ['auto-research', '--config', str(source)])
    with pytest.raises(Deployed):
        auto_research.main()
    assert json.loads((config.root/session_env.FILE).read_text())['RAX_LOOKAHEAD'] == 'off'
    # Resumed from a shell without the setting: the session's value is used.
    monkeypatch.delenv('RAX_LOOKAHEAD')
    (config.root/'configuration.json').write_text('{}')
    monkeypatch.setattr(auto_research, 'resumable_lifecycle', lambda record: [])
    import services.agent_formal.checkpoint as checkpoint
    monkeypatch.setattr(checkpoint, 'exploration_closed', lambda config: False)
    monkeypatch.setattr(sys, 'argv', ['auto-research', '--config', str(source), '--resume'])
    with pytest.raises(Deployed):
        auto_research.main()
    assert seen[-1]['RAX_LOOKAHEAD'] == 'off'
    # Resumed from a shell that sets another value: refused before anything is deployed.
    monkeypatch.setenv('RAX_LOOKAHEAD', 'on')
    with pytest.raises(SystemExit, match="RAX_LOOKAHEAD='on'"):
        auto_research.main()
    assert len(seen) == 2


def test_a_formal_batch_adopts_its_source_sessions_settings(tmp_path, monkeypatch):
    from services.controller import batch
    source, output = tmp_path/'source', tmp_path/'out'
    source.mkdir()
    (source/'configuration.json').write_text('{}')
    session_env.record(source, {'RAX_LOOKAHEAD': 'off'})
    for name in SESSION_KEPT_ENV:
        monkeypatch.delenv(name, raising=False)
    seen = []

    def from_dict(_):
        seen.append(batch.os.environ.get('RAX_LOOKAHEAD'))
        raise Deployed()

    monkeypatch.setattr(batch.Configuration, 'from_dict', from_dict)
    monkeypatch.setattr(sys, 'argv', ['batch', '--source-session', str(source), '--bundle', 'b', '--output', str(output)])
    with pytest.raises(Deployed):
        batch.main()
    assert seen == ['off']
    monkeypatch.setenv('RAX_LOOKAHEAD', 'on')
    with pytest.raises(session_env.SessionEnvConflict):
        batch.main()
