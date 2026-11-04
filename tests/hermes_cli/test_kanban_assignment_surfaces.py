"""Exercise actual HTTP, tool and CLI adapters with the shared policy enabled."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / 'shared'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    for key in ('HERMES_KANBAN_DB', 'HERMES_KANBAN_BOARD', 'HERMES_KANBAN_TASK',
                'HERMES_KANBAN_RUN_ID', 'HERMES_SESSION_ID'):
        monkeypatch.delenv(key, raising=False)
    (home / 'config.yaml').write_text('kanban:\n  allow_default_assignee: false\n  orchestrator_profile: pm\n')
    return home


@pytest.fixture
def client(home):
    spec = importlib.util.spec_from_file_location('assignment_api', ROOT / 'plugins/kanban/dashboard/plugin_api.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router)
    return TestClient(app)


def test_http_create_patch_and_reassign_refuse_clearly(client):
    response = client.post('/tasks', json={'title': 'refused', 'assignee': 'DEFAULT'})
    assert response.status_code == 400, response.text
    assert 'reserved' in response.text
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='held', assignee='developer-a', initial_status='blocked')
    for method, url, payload in (
        ('patch', f'/tasks/{tid}', {'assignee': 'default'}),
        ('post', f'/tasks/{tid}/reassign', {'profile': 'default', 'reclaim_first': True}),
    ):
        response = getattr(client, method)(url, json=payload)
        assert response.status_code == 400, response.text
        assert 'reserved' in response.text
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task.assignee == 'developer-a'
        assert task.status == 'blocked'
        assert len(kb.list_tasks(conn)) == 1


def test_create_tool_returns_clear_error(home):
    from tools import kanban_tools
    result = json.loads(kanban_tools._handle_create({'title': 'bad', 'assignee': 'default'}))
    assert 'reserved' in result['error']
    with kbc.connect() as conn:
        assert not kb.list_tasks(conn)


def test_cli_create_and_reassign_return_failure(home):
    def run(*args):
        return subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'kanban', *args],
                              cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, timeout=40)
    created = run('create', 'bad', '--assignee', 'default', '--json')
    assert created.returncode != 0
    assert 'reserved' in created.stdout + created.stderr
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='held', assignee='developer-a', initial_status='blocked')
    reassigned = run('reassign', tid, 'default')
    assert reassigned.returncode != 0
    assert 'reserved' in reassigned.stdout + reassigned.stderr
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).assignee == 'developer-a'


def test_dispatch_never_autofills_reserved_default(home, monkeypatch):
    from hermes_cli import kanban_db_dispatch as dispatch
    monkeypatch.setattr(dispatch, '_profile_exists_fn', lambda: lambda _: True)
    assert dispatch._resolve_default_assignee('default') is None
    assert dispatch._resolve_default_assignee('developer-a') == 'developer-a'


def test_decomposer_fallback_does_not_enable_dispatch_autofill(home, monkeypatch):
    from hermes_cli import kanban_decompose as dc
    from hermes_cli import config
    monkeypatch.setattr(config, 'load_config_readonly', lambda: {'kanban': {'orchestrator_profile': 'pm'}})
    monkeypatch.setattr(dc.profiles_mod, 'profile_exists', lambda _: True)
    monkeypatch.setattr(dc.profiles_mod, 'get_active_profile_name', lambda: 'default')
    monkeypatch.setattr(dc, '_build_roster', lambda: ([], {'pm', 'developer-a'}))
    routing = dc._load_routing()
    assert routing.orchestrator == routing.default_assignee == 'pm'
    assert '\n  default_assignee:' not in (home / 'config.yaml').read_text()
