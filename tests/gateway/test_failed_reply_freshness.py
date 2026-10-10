"""Failed replies expire after 30 minutes, without expiring ambiguous sends."""
import pytest
from gateway import delivery_ledger as dl


@pytest.mark.parametrize('state,age,expected', [
    ('failed', 1801, False), ('failed', 1800, True), ('failed', 60, True),
    ('pending', 3600, True), ('attempting', 3600, True),
])
def test_restart_retry_freshness(tmp_path, monkeypatch, state, age, expected):
    monkeypatch.setattr(dl, '_db_path', lambda: tmp_path / 'state.db')
    monkeypatch.setattr(dl, '_owner_alive', lambda *_: False)
    dl.record_obligation(obligation_id='o', session_key='k', platform='discord',
                         chat_id='1', thread_id='1', content='reply')
    if state == 'failed':
        dl.mark_failed('o', '403 Forbidden (error code: 50001): Missing Access')
    elif state == 'attempting':
        dl.mark_attempting('o')
    with dl._connect() as conn:
        conn.execute('UPDATE delivery_obligations SET created_at=?, updated_at=? WHERE obligation_id=?',
                     (100000 - age, 100000 - age, 'o'))
    rows = dl.sweep_recoverable(now=100000)
    assert bool(rows) is expected
    if expected:
        assert rows[0]['needs_marker'] is (state != 'pending')
    else:
        with dl._connect() as conn:
            assert conn.execute('SELECT state FROM delivery_obligations').fetchone()[0] == 'abandoned'
