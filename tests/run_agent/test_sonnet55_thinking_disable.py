"""Sonnet 5.5 proxy rejection and continuation regressions (2 Oct 2026)."""
import copy

import pytest

from agent import anthropic_adapter as aa
from agent.error_classifier import FailoverReason, classify_api_error
from tests.run_agent.test_thinking_disable_rejected_20261001 import _FakeApiError
from tests.run_agent.test_thinking_disable_review import make, response, run_sdk

SONNET_55_400 = 'To turn thinking off on this model, send "thinking": {"type": "between_tools"} instead'


@pytest.fixture(autouse=True)
def clear_learning():
    aa._RUNTIME_DISABLE_REJECTED_MODELS.clear()
    yield
    aa._RUNTIME_DISABLE_REJECTED_MODELS.clear()


@pytest.mark.parametrize('model', [
    'claude-sonnet-5-5', 'claude-sonnet-5.5',
    'anthropic/claude-sonnet-5.5', 'claude-sonnet-5-5-20260929',
])
def test_static_verdict(model):
    assert aa.thinking_disable_is_rejected(model)


@pytest.mark.parametrize('model', ['claude-sonnet-5', 'claude-sonnet-5-50', 'claude-sonnet-4-6'])
def test_other_versions_unaffected(model):
    assert not aa.thinking_disable_is_rejected(model)


def test_actual_sonnet_error_classifies_as_repairable():
    result = classify_api_error(_FakeApiError(400, SONNET_55_400), provider='custom', model='claude-sonnet-5-5')
    assert result.reason == FailoverReason.thinking_disable_rejected
    assert result.retryable is False
    assert result.should_compress is False


def test_real_continuation_retains_configured_max():
    agent = make('claude-sonnet-5-5')
    original = copy.deepcopy(agent.reasoning_config)
    seen = []
    result = run_sdk(agent, [response('', 'length'), response()], seen)
    assert result['completed'] is True, result.get('error')
    assert [k.get('reasoning_effort') for k in seen] == ['max', 'low']
    assert agent.reasoning_config == original
    later = []
    assert run_sdk(agent, [response()], later, result['messages'])['completed'] is True
    assert [k.get('reasoning_effort') for k in later] == ['max']


def test_unlisted_alias_learns_sonnet_error_once():
    agent = make('private-sonnet-alias', {'enabled': False, 'effort': 'none'})
    seen = []
    result = run_sdk(agent, [_FakeApiError(400, SONNET_55_400), response()], seen)
    assert result['completed'] is True, result.get('error')
    assert [k.get('reasoning_effort') for k in seen] == ['none', 'low']
    assert agent.reasoning_config == {'enabled': False, 'effort': 'none'}


def test_second_sonnet_rejection_terminates():
    agent = make('private-sonnet-alias', {'enabled': False, 'effort': 'none'})
    seen = []
    result = run_sdk(agent, [_FakeApiError(400, SONNET_55_400)] * 2, seen)
    assert result.get('completed') is not True
    assert [k.get('reasoning_effort') for k in seen] == ['none', 'low']


def test_all_disable_builders_omit_unsupported_sonnet_shape():
    from agent.auxiliary_client import _build_call_kwargs
    from agent.transports.chat_completions import ChatCompletionsTransport
    import model_tools  # noqa: F401
    from providers import get_provider_profile

    cfg = {'enabled': False, 'effort': 'none'}
    model = 'claude-sonnet-5-5'
    msg = [{'role': 'user', 'content': 'hi'}]
    native = aa.build_anthropic_kwargs(model=model, messages=msg, tools=None, max_tokens=4096, reasoning_config=cfg)
    assert native.get('thinking') != {'type': 'disabled'}
    legacy = ChatCompletionsTransport().build_kwargs(model=model, messages=msg, reasoning_config=cfg, supports_reasoning=True)
    assert 'reasoning' not in (legacy.get('extra_body') or {})
    custom = ChatCompletionsTransport().build_kwargs(model=model, messages=msg, reasoning_config=cfg, provider_profile=get_provider_profile('custom'))
    assert custom.get('reasoning_effort') != 'none'
    for provider in ['custom', 'unregistered-synthetic', 'anthropic']:
        aux = _build_call_kwargs(model=model, provider=provider, messages=msg, reasoning_config=cfg)
        assert aux.get('reasoning_effort') != 'none'
        assert (aux.get('extra_body') or {}).get('reasoning', {}).get('enabled') is not False
        if '_reasoning_config' in aux:
            assert aux['_reasoning_config'] == {'enabled': True, 'effort': 'low'}


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_wrapped_relay_error_is_repairable(depth):
    import json
    message = SONNET_55_400
    for _ in range(depth):
        message = json.dumps({'error': {'message': message, 'type': 'invalid_request_error'}})
    assert classify_api_error(_FakeApiError(400, message)).reason == FailoverReason.thinking_disable_rejected


@pytest.mark.parametrize('message', [
    'Unknown parameter thinking: {"type": "between_tools"}',
    'To turn thinking off on this model, send "tools": {"type": "between_tools"} instead',
])
def test_similar_unrelated_error_is_not_repaired(message):
    assert classify_api_error(_FakeApiError(400, message)).reason != FailoverReason.thinking_disable_rejected


@pytest.mark.parametrize('provider', ['custom', 'anthropic', 'unregistered-synthetic'])
def test_auxiliary_task_config_reaches_protected_wire(provider):
    from unittest.mock import patch, MagicMock
    from agent.auxiliary_client import _build_call_kwargs, _get_task_extra_body, _AnthropicCompletionsAdapter
    import model_tools  # noqa: F401
    base = 'https://relay.invalid/v1'
    model = 'claude-opus-5.7'
    aa.remember_thinking_disable_rejected(model, provider=provider, base_url=base)
    with patch('agent.auxiliary_client._get_auxiliary_task_config', return_value={'reasoning_effort': 'none'}):
        body = _get_task_extra_body('title')
    original = copy.deepcopy(body)
    kwargs = _build_call_kwargs(provider=provider, model=model, base_url=base, messages=[{'role': 'user', 'content': 'hi'}], extra_body=body)
    assert body == original
    if provider == 'custom':
        assert kwargs['reasoning_effort'] == 'low'
        assert 'reasoning' not in kwargs.get('extra_body', {})
    elif provider == 'anthropic':
        assert kwargs['_reasoning_config'] == {'enabled': True, 'effort': 'low'}
        # Exercise the final native consumer, not just the private marker.
        adapter = _AnthropicCompletionsAdapter(MagicMock(), model, base_url=base)
        with patch('agent.anthropic_adapter.create_anthropic_message', side_effect=RuntimeError('stop before SDK')) as send:
            with pytest.raises(RuntimeError, match='stop before SDK'):
                adapter.create(**kwargs)
        assert send.call_args.args[1].get('thinking') != {'type': 'disabled'}
    else:
        assert kwargs['extra_body']['reasoning'] == {'enabled': True, 'effort': 'low'}


def test_auxiliary_explicit_reasoning_wins_without_mutation():
    from agent.auxiliary_client import _build_call_kwargs
    import model_tools  # noqa: F401
    body = {'reasoning': {'enabled': False, 'effort': 'none'}, 'unrelated': 123}
    config = {'enabled': True, 'effort': 'max'}
    original = copy.deepcopy(body)
    kwargs = _build_call_kwargs(provider='custom', model='claude-sonnet-5-5', base_url='https://relay.invalid/v1', messages=[{'role': 'user', 'content': 'hi'}], extra_body=body, reasoning_config=config)
    assert kwargs['reasoning_effort'] == 'max'
    assert kwargs['extra_body'] == {'unrelated': 123}
    assert body == original
    assert config == {'enabled': True, 'effort': 'max'}
