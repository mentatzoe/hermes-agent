"""Delegated children get an opt-in read-only provider, never the parent's writer."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from agent.memory_manager import MemoryManager, external_memory_active
from agent.memory_provider import MemoryProvider
from tools.delegate_tool import _build_child_agent


class ReadMemory(MemoryProvider):
    name = 'synthetic-memory'

    def __init__(self, read_only=False):
        self.read_only = read_only
        self.closed = False

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        raise AssertionError('A captured read-only child must not initialize the parent provider')

    def fork_read_only(self, *, session_id):
        return ReadMemory(read_only=True)

    def system_prompt_block(self):
        return 'Declared-persona-once; this memory is read-only.'

    def get_tool_schemas(self):
        return [{'name': 'synthetic_memory_search', 'description': 'Read saved facts.',
                 'parameters': {'type': 'object', 'properties': {}}}]

    def handle_tool_call(self, name, args, **kwargs):
        assert self.read_only
        return 'saved evidence'

    def shutdown(self):
        self.closed = True


def make_child(manager, tmp_path, *, disabled=None):
    parent = SimpleNamespace(_delegate_depth=0, session_id='parent', _session_db=None,
        enabled_toolsets=['memory'], disabled_toolsets=disabled or [], tools=[],
        _memory_manager=manager, model='test-model', provider='openai',
        api_key='synthetic-test-key', base_url='https://isolated.invalid/v1',
        api_mode='chat_completions', _active_children=[], _credential_pool=None,
        prefill_messages=None, request_overrides={}, terminal_cwd=str(tmp_path))
    import threading
    parent._active_children_lock = threading.Lock()
    with (patch('model_tools.get_tool_definitions', return_value=[]),
          patch('model_tools.check_toolset_requirements', return_value={}),
          patch('hermes_cli.config.load_config', return_value={}),
          patch('agent.process_bootstrap.OpenAI'),
          patch('tools.delegate_tool._load_config', return_value={})):
        child = _build_child_agent(0, 'Find saved evidence.', None, None, None, 2, 1, parent)
    return parent, child


def test_delegate_build_installs_a_distinct_read_only_provider_before_first_prompt(tmp_path):
    manager = MemoryManager()
    provider = ReadMemory()
    manager.add_provider(provider)
    _, child = make_child(manager, tmp_path)
    try:
        assert child._memory_manager is not None
        assert child._memory_manager is not manager
        assert child._memory_manager.providers[0] is not provider
        assert external_memory_active(child)
        names = {tool['function']['name'] for tool in child.tools}
        assert 'synthetic_memory_search' in names
        assert 'memory' not in names
        assert child._memory_store is None
        prompt = child._build_system_prompt()
        assert prompt.count('Declared-persona-once') == 1
        assert child._memory_manager.handle_tool_call('synthetic_memory_search', {}) == 'saved evidence'
        child._memory_manager.shutdown_all()
        assert not provider.closed, 'a child must not close the parent provider'
    finally:
        child.close()


@pytest.mark.parametrize('mode', ['disabled', 'no-opt-in', 'shared', 'builtin'])
def test_child_reader_does_not_inherit_an_unpermitted_or_shared_writer(mode):
    from agent.read_only_memory import inherit_read_only_memory
    provider = ReadMemory()
    if mode == 'no-opt-in':
        provider.fork_read_only = lambda **kwargs: None
    elif mode == 'shared':
        provider.fork_read_only = lambda **kwargs: provider
    elif mode == 'builtin':
        provider.name = 'builtin'
    manager = MemoryManager()
    manager.add_provider(provider)
    parent = SimpleNamespace(_memory_manager=manager, tools=[], enabled_toolsets=['memory'],
                             disabled_toolsets=['memory'] if mode == 'disabled' else [])
    child = SimpleNamespace(_memory_manager=None, session_id='helper')
    inherit_read_only_memory(parent, child)
    assert child._memory_manager is None
    assert not provider.closed


def test_unusable_child_reader_is_closed_without_closing_the_parent():
    from agent.read_only_memory import inherit_read_only_memory
    parent_provider = ReadMemory()
    class UnusableReader(ReadMemory):
        def get_tool_schemas(self):
            raise RuntimeError('synthetic unavailable tool schema')
    candidate = UnusableReader(read_only=True)
    parent_provider.fork_read_only = lambda **kwargs: candidate
    manager = MemoryManager()
    manager.add_provider(parent_provider)
    parent = SimpleNamespace(_memory_manager=manager, tools=[], enabled_toolsets=['memory'], disabled_toolsets=[])
    child = SimpleNamespace(_memory_manager=None, session_id='helper')
    inherit_read_only_memory(parent, child)
    assert child._memory_manager is None
    assert candidate.closed
    assert not parent_provider.closed
