"""Accepted input conservation at the canonical row merge, not the model wire."""
from copy import deepcopy

import pytest

from agent.agent_runtime_helpers import _merge_consecutive_users
from agent.context_compressor import _DB_PERSISTED_MARKER


def entry(sender):
    return {'text': '[rendered sender] same words', 'facts': {
        'sender_id': sender, 'typed_user_content': 'same words',
        'recoverable_audio_paths': ['/audio/' + sender + '.ogg']}}


@pytest.mark.parametrize('left_has_entries', [False, True])
@pytest.mark.parametrize('right_empty', [False, True])
def test_merge_conserves_ordered_entries_even_without_text_change(left_has_entries, right_empty):
    before = [entry('alex')] if left_has_entries else []
    incoming = [entry('blair'), entry('blair')]
    left = {'role': 'user', 'content': 'opener', _DB_PERSISTED_MARKER: True}
    if before:
        left['steer_entries'] = deepcopy(before)
    right = {'role': 'user', 'content': '' if right_empty else 'same words',
             'steer_entries': deepcopy(incoming)}
    rows, repairs = _merge_consecutive_users([left, right])
    assert repairs == 1
    assert rows[0]['steer_entries'] == before + incoming
    assert _DB_PERSISTED_MARKER not in rows[0]
    original = deepcopy(rows)
    again, repairs = _merge_consecutive_users(rows)
    assert repairs == 0 and again == original
    right['steer_entries'][0]['facts']['sender_id'] = 'mutated'
    assert again == original


def carrier(kind, sender):
    row = {'role': 'user', 'content': 'same words'}
    if kind == 'plain':
        return row
    if kind == 'followup':
        row['display_metadata'] = {'steer_entries': [entry(sender)]}
    elif kind == 'restored':
        row['display_metadata'] = {'mid_turn_entries': [entry(sender)]}
    else:
        row['steer_entries'] = [entry(sender)]
        row['display_kind'] = kind
    return row


def retained_entries(rows):
    result = []
    for row in rows:
        metadata = row.get('display_metadata') or {}
        result.extend(metadata.get('steer_entries') or [])
        result.extend(row.get('steer_entries', metadata.get('mid_turn_entries')) or [])
    return result


@pytest.mark.parametrize('left_kind', ['plain', 'active_turn_redirect', 'steer', 'followup', 'restored'])
@pytest.mark.parametrize('right_kind', ['plain', 'active_turn_redirect', 'steer', 'followup', 'restored'])
def test_transform_conserves_both_carriers(left_kind, right_kind):
    rows = [carrier(left_kind, 'alex'), carrier(right_kind, 'blair')]
    expected = deepcopy(retained_entries(rows))
    merged, _ = _merge_consecutive_users(rows)
    assert retained_entries(merged) == expected
    again, repairs = _merge_consecutive_users(merged)
    assert repairs == 0 and retained_entries(again) == expected


@pytest.mark.parametrize('merged', [False, True])
def test_accepted_entries_survive_flush_reload(tmp_path, merged):
    import json
    import sqlite3
    from types import SimpleNamespace
    from agent.mid_turn_input import mid_turn_human_row
    from agent.session_persistence import _db_flush_row
    from hermes_state import SessionDB

    agent = SimpleNamespace()
    expected = [entry('blair')]
    delivered = mid_turn_human_row(agent, expected, content='same words')
    db = SessionDB(tmp_path / 'facts.db')
    try:
        db.create_session(session_id='facts', source='test')
        if merged:
            # An already-durable opener absorbs metadata but no additional text.
            rows = [{'role': 'user', 'content': 'opener'}]
            db.append_messages_batch('facts', rows)
            with sqlite3.connect(tmp_path / 'facts.db') as connection:
                connection.execute('UPDATE messages SET display_metadata = ?',
                                   (json.dumps({'reactions': ['fresh reaction']}),))
            rows[0][_DB_PERSISTED_MARKER] = True
            delivered['content'] = ''
            rows, _ = _merge_consecutive_users([*rows, delivered])
            assert _DB_PERSISTED_MARKER not in rows[0]
        else:
            rows = [delivered]
        db.append_messages_batch('facts', [_db_flush_row(agent, row, False) for row in rows])
        restored = db.get_messages_as_conversation('facts')
        assert len(restored) == 1
        assert retained_entries(restored) == expected
        if merged:
            assert restored[0]['display_metadata']['reactions'] == ['fresh reaction']
        restored, repairs = _merge_consecutive_users(restored)
        assert repairs == 0 and retained_entries(restored) == expected
    finally:
        db.close()


def test_concurrent_source_metadata_change_is_not_overwritten(tmp_path):
    import json
    import sqlite3
    from types import SimpleNamespace
    from agent.message_metadata import CANONICAL_ROW
    from agent.mid_turn_input import mid_turn_human_row
    from agent.session_persistence import _db_flush_row
    from hermes_state import SessionDB

    agent = SimpleNamespace()
    db = SessionDB(tmp_path / 'facts.db')
    try:
        db.create_session(session_id='facts', source='test')
        row = mid_turn_human_row(agent, [entry('alex')], content='same words')
        db.append_messages_batch('facts', [row])
        winner = [entry('other-writer')]
        with sqlite3.connect(tmp_path / 'facts.db') as connection:
            connection.execute('UPDATE messages SET display_metadata = ?',
                               (json.dumps({'mid_turn_entries': winner}),))
        merged, _ = _merge_consecutive_users([row, carrier('active_turn_redirect', 'blair')])
        batch = [_db_flush_row(agent, merged[0], False)]
        db.append_messages_batch('facts', batch)
        assert retained_entries(db.get_messages_as_conversation('facts')) == winner
        assert batch[0][CANONICAL_ROW]['content'] == 'same words'
        assert retained_entries([batch[0][CANONICAL_ROW]]) == winner
        from agent.transcript_repair import sync_flushed_message_markers
        sync_flushed_message_markers(merged, batch)
        assert retained_entries(merged) == winner
    finally:
        db.close()


@pytest.mark.parametrize('left_kind', ['active_turn_redirect', 'steer', 'followup', 'restored'])
def test_left_only_facts_survive_merge_and_durable_reload(tmp_path, left_kind):
    from types import SimpleNamespace
    from agent.session_persistence import _db_flush_row
    from hermes_state import SessionDB

    rows = [carrier(left_kind, 'alex'), carrier('plain', 'blair')]
    expected = retained_entries(rows)
    merged, _ = _merge_consecutive_users(rows)
    db = SessionDB(tmp_path / 'facts.db')
    try:
        db.create_session(session_id='facts', source='test')
        db.append_messages_batch('facts', [_db_flush_row(SimpleNamespace(), row, False) for row in merged])
        assert retained_entries(db.get_messages_as_conversation('facts')) == expected
    finally:
        db.close()
