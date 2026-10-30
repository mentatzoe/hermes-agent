"""A conversational default profile can be reserved across all shared boards."""
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_graph as graph


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / 'shared'
    root.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(root))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_KANBAN_BOARD', raising=False)
    (root / 'config.yaml').write_text('kanban:\n  allow_default_assignee: false\n')
    return root


@pytest.mark.parametrize('board', ['default', 'existing', 'created-later'])
@pytest.mark.parametrize('name', ['default', 'DEFAULT', ' Default '])
def test_create_refuses_before_any_row_or_event(home, board, name):
    if board != 'default':
        kb.create_board(board)
    with kbc.connect(board=board) as conn:
        before = conn.execute('SELECT count(*) FROM task_events').fetchone()[0]
        with pytest.raises(ValueError, match='default.*reserved'):
            kb.create_task(conn, title='refused', assignee=name)
        assert kb.list_tasks(conn) == []
        assert conn.execute('SELECT count(*) FROM task_events').fetchone()[0] == before


@pytest.mark.parametrize('method', [kb.assign_task, kb.reassign_task])
def test_reassign_preserves_block_and_history(home, method):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='held', assignee='developer-a', initial_status='blocked')
        before = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
        with pytest.raises(ValueError, match='default.*reserved'):
            method(conn, tid, 'default')
        assert kb.get_task(conn, tid) == before
        assert kb.list_events(conn, tid) == events
        assert method(conn, tid, 'developer-b')
        assert kb.get_task(conn, tid).status == 'blocked'


def test_reassign_validates_before_reclaim(home, monkeypatch):
    called = []
    monkeypatch.setattr(kb, 'reclaim_task', lambda *a, **kw: called.append(True))
    with kbc.connect() as conn, pytest.raises(ValueError, match='default.*reserved'):
        kb.reassign_task(conn, 'no-such-task', 'default', reclaim_first=True)
    assert called == []


def test_profile_config_cannot_override_shared_policy(home, monkeypatch):
    profile = home / 'profiles' / 'developer-a'
    profile.mkdir(parents=True)
    (profile / 'config.yaml').write_text('kanban:\n  allow_default_assignee: true\n')
    monkeypatch.setenv('HERMES_HOME', str(profile))
    with kbc.connect() as conn, pytest.raises(ValueError, match='default.*reserved'):
        kb.create_task(conn, title='worker bypass', assignee='default')


def test_policy_is_read_again_and_history_remains_readable(home):
    (home / 'config.yaml').write_text('kanban:\n  allow_default_assignee: true\n')
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='legacy', assignee='default')
        (home / 'config.yaml').write_text('kanban:\n  allow_default_assignee: false\n')
        assert [t.id for t in kb.list_tasks(conn, assignee='default')] == [tid]
        with pytest.raises(ValueError, match='default.*reserved'):
            kb.create_task(conn, title='retry', assignee='default', idempotency_key='new')
        assert kb.assign_task(conn, tid, 'developer-a')
        assert kb.assign_task(conn, tid, None)


@pytest.mark.parametrize('config', ['', 'kanban:\n  allow_default_assignee: true\n'])
def test_upstream_opt_in_behavior_preserved(home, config):
    (home / 'config.yaml').write_text(config)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='allowed', assignee='default')
        assert kb.get_task(conn, tid).assignee == 'default'


@pytest.mark.parametrize('value', ['null', '"false"', '[]', '{}', '1'])
def test_malformed_policy_fails_closed(home, value):
    (home / 'config.yaml').write_text(f'kanban:\n  allow_default_assignee: {value}\n')
    with kbc.connect() as conn, pytest.raises(ValueError, match='allow_default_assignee'):
        kb.create_task(conn, title='malformed policy', assignee='default')


def test_specify_and_review_cannot_assign_default(home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='triage', assignee='developer-a', triage=True)
        with pytest.raises(ValueError, match='default.*reserved'):
            kb.specify_triage_task(conn, tid, assignee='default')
        assert kb.get_task(conn, tid).status == 'triage'
        ready = kb.create_task(conn, title='review', assignee='developer-a')
        with pytest.raises(ValueError, match='default.*reserved'):
            kb.request_review(conn, ready, reviewer='default', summary='done')
        assert kb.get_task(conn, ready).status == 'ready'


@pytest.mark.parametrize('root, child', [('default', 'developer-a'), ('developer-a', 'default')])
def test_decomposition_is_atomic(home, root, child):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='root', assignee='developer-a', triage=True)
        with pytest.raises(ValueError, match='default.*reserved'):
            graph.decompose_triage_task(conn, tid, root_assignee=root, children=[
                {'title': 'valid first', 'assignee': 'developer-a'},
                {'title': 'second', 'assignee': child},
            ])
        assert [t.id for t in kb.list_tasks(conn)] == [tid]
        assert kb.get_task(conn, tid).status == 'triage'


def test_picker_and_decomposer_exclude_reserved_profile(home, monkeypatch):
    from hermes_cli import kanban_decompose as dc
    monkeypatch.setattr(kb, 'list_profiles_on_disk', lambda: ['default', 'developer-a'])
    monkeypatch.setattr(dc.profiles_mod, 'list_profiles', lambda: [
        SimpleNamespace(name=n, description='') for n in ('default', 'developer-a')])
    with kbc.connect() as conn:
        assert [r['name'] for r in kb.known_assignees(conn)] == ['developer-a']
    roster, names = dc._build_roster()
    assert names == {'developer-a'}
    assert [r['name'] for r in roster] == ['developer-a']


def test_review_cannot_restore_legacy_default_implementer(home):
    (home / 'config.yaml').write_text('kanban:\n  allow_default_assignee: true\n')
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='legacy implementation', assignee='default')
        implementation = kb.claim_task(conn, tid, claimer='old:test')
        assert implementation is not None
        assert kb.request_review(conn, tid, reviewer='developer-b', summary='ready',
                                 expected_run_id=implementation.current_run_id)
        review = kb.claim_review_task(conn, tid, claimer='review:test')
        assert review is not None
        before = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
        (home / 'config.yaml').write_text('kanban:\n  allow_default_assignee: false\n')
        with pytest.raises(ValueError, match='default.*reserved'):
            kb.request_changes(conn, tid, reason='needs work', expected_run_id=review.current_run_id)
        assert kb.get_task(conn, tid) == before
        assert kb.list_events(conn, tid) == events


def test_unreadable_policy_fails_closed(home, monkeypatch):
    from hermes_cli import config_effective
    def unavailable(**kwargs):
        raise OSError('unreadable config')
    monkeypatch.setattr(config_effective, 'load_user_config_effective', unavailable)
    with kbc.connect() as conn, pytest.raises(ValueError, match='cannot read shared'):
        kb.create_task(conn, title='refused', assignee='default')


def test_decomposer_fallback_requires_worker(home, monkeypatch):
    from hermes_cli import kanban_decompose as dc
    monkeypatch.setattr(dc.profiles_mod, 'profile_exists', lambda n: True)
    monkeypatch.setattr(dc.profiles_mod, 'get_active_profile_name', lambda: 'default')
    assert dc._resolve_profile_from_cfg({}, 'default_assignee', fallback='developer-a') == 'developer-a'
    assert dc._resolve_profile_from_cfg({'kanban': {'default_assignee': 'developer-b'}}, 'default_assignee') == 'developer-b'
    with pytest.raises(ValueError, match='default.*reserved'):
        dc._resolve_profile_from_cfg({}, 'default_assignee')
