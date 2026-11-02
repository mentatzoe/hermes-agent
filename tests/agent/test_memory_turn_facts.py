"""Host-owned facts reach providers without deriving either from rendered text."""
from types import SimpleNamespace
import pytest
from agent.memory_manager import MemoryManager
from tests.agent.test_memory_provider import FakeMemoryProvider


class FactsProvider(FakeMemoryProvider):
    def sync_turn(self, user_content, assistant_content, *, session_id='', messages=None,
                  turn_author=None, typed_user_content=None, current_turn_user_idx=None,
                  recoverable_audio_paths=None):
        self.synced_turns.append(dict(user=user_content, messages=messages,
            typed=typed_user_content, current_turn_user_idx=current_turn_user_idx,
            recoverable_audio_paths=recoverable_audio_paths))


@pytest.mark.parametrize('micro_compaction', [False, True])
def test_real_conversation_passes_owned_facts_and_resets_next_turn(monkeypatch, micro_compaction):
    from run_agent import AIAgent
    if micro_compaction:
        # Post-turn history maintenance removes old rows, not the completed-turn snapshot.
        monkeypatch.setattr('agent.turn_finalizer._micro_compact_after_turn',
                            lambda _agent, rows, *args: rows.__delitem__(slice(1, 3)))
    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content='Done.', tool_calls=None, reasoning=None, reasoning_content=None),
                finish_reason='stop')], usage=None, model='test-model')
    monkeypatch.setattr('agent.process_bootstrap.OpenAI',
        lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr('model_tools.get_tool_definitions', lambda *a, **kw: [])
    agent = AIAgent(model='test-model', api_key='k', base_url='http://localhost:1/v1',
                    platform='cli', max_iterations=2, quiet_mode=True, skip_memory=True,
                    skip_background_review=True)
    agent._disable_streaming = True
    provider = FactsProvider('facts')
    manager = MemoryManager()
    manager.add_provider(provider)
    agent._memory_manager = manager
    history = [{'role':'user', 'content':'previous'}, {'role':'assistant','content':'previous answer'}]
    agent.run_conversation('[Discord trigger] caption', conversation_history=history,
        typed_user_message='Caption.\n\nAutomatic words.',
        recoverable_audio_paths=['/fixture/current.ogg'], persist_user_message='Caption.',
        turn_author={'id':'human','name':'Alex','is_bot':False})
    assert manager.flush_pending(timeout=5)
    first = provider.synced_turns[-1]
    assert first['typed'] == 'Caption.\n\nAutomatic words.'
    assert first['recoverable_audio_paths'] == ['/fixture/current.ogg']
    assert type(first['current_turn_user_idx']) is int
    assert first['messages'][first['current_turn_user_idx']]['content'] == 'Caption.'
    agent.run_conversation('Next human turn.')
    assert manager.flush_pending(timeout=5)
    assert provider.synced_turns[-1]['recoverable_audio_paths'] == []
    manager.shutdown_all()


def test_codex_early_return_passes_the_loop_boundary(monkeypatch):
    from agent.conversation_loop import _codex_app_server_turn
    from agent.codex_runtime import run_codex_app_server_turn
    from tests.agent.test_codex_app_server_persist import _make_agent
    agent = _make_agent()
    agent._run_codex_app_server_turn = lambda **kw: run_codex_app_server_turn(agent, **kw)
    state = SimpleNamespace(user_message='List-shaped rendered opener', original_user_message='Persisted opener',
        messages=[{'role':'system','content':'system'}, {'role':'user','content':['image', 'text']}],
        current_turn_user_idx=1, effective_task_id='task', _should_review_memory=False)
    result = _codex_app_server_turn(agent, state)
    assert result['completed'] is True
    assert agent._sync_external_memory_for_turn.call_args.kwargs['current_turn_user_idx'] == 1
