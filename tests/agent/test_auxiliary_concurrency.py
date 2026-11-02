"""Tests for per-task concurrency limiting on auxiliary LLM calls (#23324)."""

import asyncio
import threading
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from agent.auxiliary_client import (
    call_llm,
    async_call_llm,
    _acquire_sync_aux_semaphore,
    _acquire_async_aux_semaphore,
    _get_task_max_concurrency,
    _reset_aux_semaphores,
)


@pytest.fixture(autouse=True)
def _clean_semaphore_cache():
    _reset_aux_semaphores()
    yield
    _reset_aux_semaphores()


class TestGetTaskMaxConcurrency:


    def test_does_not_reuse_vision_cpu_limit_for_llm_calls(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 1},
        ):
            assert _get_task_max_concurrency("vision") is None

    def test_returns_int_when_configured(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 3},
        ):
            assert _get_task_max_concurrency("compression") == 3

    def test_returns_none_for_non_numeric(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": "not-a-number"},
        ):
            assert _get_task_max_concurrency("compression") is None

    def test_returns_none_for_zero_or_negative(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 0},
        ):
            assert _get_task_max_concurrency("compression") is None
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": -2},
        ):
            assert _get_task_max_concurrency("compression") is None


class TestSemaphoreCache:
    def test_sync_returns_none_when_unset(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config", return_value={}
        ):
            assert _acquire_sync_aux_semaphore("title_generation") is None




    def test_async_returns_none_with_no_running_loop(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 2},
        ):
            # Called outside an asyncio loop — should bail rather than crash.
            assert _acquire_async_aux_semaphore("compression") is None


class TestSyncCallEnforcesLimit:
    def test_call_llm_caps_concurrent_inflight(self):
        from tests.agent.thread_sync_helpers import CheckedThread, join_checked

        limit, n_callers = 2, 6
        active = max_active = 0
        lock = threading.Lock()
        ready = threading.Barrier(n_callers + 1)
        saturated = threading.Event()
        release = threading.Event()
        response = object()

        def fake_create(**kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                if active == limit:
                    saturated.set()
            try:
                assert release.wait(timeout=5), "provider fixture was not released"
                return response
            finally:
                with lock:
                    active -= 1

        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create.return_value = response

        def call():
            return call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])

        def caller():
            ready.wait(timeout=5)
            return call()

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch("agent.auxiliary_client._get_cached_client", return_value=(client, "test-model")),
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, *_a, **_kw: resp),
            patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"max_concurrency": limit}),
        ):
            # Exercise the real preparation once before measuring contention. Otherwise
            # one caller spends its join budget discovering plugins while others queue (#60).
            assert call() is response
            client.chat.completions.create.reset_mock()
            client.chat.completions.create.side_effect = fake_create
            semaphore = _acquire_sync_aux_semaphore("title_generation")
            assert semaphore is not None
            original_acquire = semaphore.acquire
            all_attempted = threading.Event()
            attempts = []

            def observed_acquire():
                # Probe the REAL permit first, then take the normal blocking path.
                # Knowing all six reached acquire prevents a late-scheduled caller
                # from hiding an excess permit behind the provider release signal.
                immediate = original_acquire(blocking=False)
                with lock:
                    attempts.append(immediate)
                    if len(attempts) == n_callers:
                        all_attempted.set()
                return immediate or original_acquire()

            threads = [CheckedThread(caller, name=f"aux-caller-{i}") for i in range(n_callers)]
            with patch.object(semaphore, "acquire", observed_acquire):
                for t in threads:
                    t.start()
                try:
                    ready.wait(timeout=5)
                    assert all_attempted.wait(timeout=5), "not every caller attempted its permit"
                    assert saturated.wait(timeout=5), "two calls never overlapped"
                    assert sum(attempts) == limit
                except BaseException:
                    ready.abort()
                    raise
                finally:
                    release.set()
                    # Finish every caller before unpatching the semaphore/provider.
                    join_checked(*threads)

        assert max_active == limit, f"observed peak {max_active}, expected {limit}"
        assert active == 0
        assert client.chat.completions.create.call_count == n_callers
        assert all(t.value is response for t in threads)


    def test_semaphore_released_on_exception(self):
        """Errors inside call_llm must release the semaphore so the next call proceeds."""
        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create.side_effect = RuntimeError("boom")

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "test-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda resp, _task, **_kwargs: resp,
            ),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": 1},
            ),
        ):
            for _ in range(3):
                with pytest.raises(RuntimeError, match="boom"):
                    call_llm(
                        task="title_generation",
                        messages=[{"role": "user", "content": "hi"}],
                    )

    def test_stream_holds_permit_until_consumed_and_preserves_options(self):
        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create.side_effect = [iter(["chunk"]), MagicMock()]
        second_call_started = threading.Event()

        def make_second_call():
            second_call_started.set()
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "second"}],
            )

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "test-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda response, _task, **_kwargs: response,
            ),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": 1},
            ),
        ):
            stream = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "first"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            thread = threading.Thread(target=make_second_call)
            thread.start()
            assert second_call_started.wait(timeout=1)
            time.sleep(0.05)
            assert client.chat.completions.create.call_count == 1
            assert list(stream) == ["chunk"]
            thread.join(timeout=1)

        assert not thread.is_alive()
        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[0].kwargs["stream"] is True
        assert client.chat.completions.create.call_args_list[0].kwargs["stream_options"] == {
            "include_usage": True
        }



class TestAsyncCallEnforcesLimit:
    @pytest.mark.asyncio
    async def test_async_call_llm_caps_concurrent_inflight(self):
        limit = 2
        n_callers = 6

        active = 0
        max_active = 0

        async def fake_create(**kwargs):
            nonlocal active, max_active
            active += 1
            if active > max_active:
                max_active = active
            try:
                await asyncio.sleep(0.05)
            finally:
                active -= 1
            return MagicMock()

        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create = AsyncMock(side_effect=fake_create)

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "test-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda resp, _task, **_kwargs: resp,
            ),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": limit},
            ),
        ):
            await asyncio.gather(*[
                async_call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                )
                for _ in range(n_callers)
            ])

        assert max_active <= limit, f"observed {max_active} > limit {limit}"
        assert client.chat.completions.create.await_count == n_callers

    @pytest.mark.asyncio
    async def test_async_semaphore_released_on_exception(self):
        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "test-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda resp, _task, **_kwargs: resp,
            ),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": 1},
            ),
        ):
            for _ in range(3):
                with pytest.raises(RuntimeError, match="boom"):
                    await async_call_llm(
                        task="compression",
                        messages=[{"role": "user", "content": "hi"}],
                    )
