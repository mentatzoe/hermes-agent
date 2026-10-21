"""The shared float reader preserves caller policy and native epoch units."""
from types import SimpleNamespace

import pytest

from hermes_cli.process_start_time import process_create_time


def test_public_only_process_keeps_full_epoch_precision():
    proc = SimpleNamespace(create_time=lambda: 1792487588.691234)
    assert process_create_time(proc) == 1792487588.691234


@pytest.mark.platforms("macos")
def test_supported_private_read_avoids_the_adjusted_public_read():
    def raw(*, monotonic):
        assert monotonic is True
        return 1792487588.691234

    def public():
        raise AssertionError("the adjusted public API must not be read")

    proc = SimpleNamespace(_proc=SimpleNamespace(create_time=raw), create_time=public)
    assert process_create_time(proc) == 1792487588.691234


@pytest.mark.platforms("macos")
def test_older_private_api_without_the_keyword_keeps_the_public_fallback():
    proc = SimpleNamespace(_proc=SimpleNamespace(create_time=lambda: 123.0), create_time=lambda: 456.25)
    assert process_create_time(proc) == 456.25


@pytest.mark.platforms("macos")
def test_a_failed_private_read_does_not_become_a_successful_identity():
    def raw(*, monotonic):
        raise PermissionError("identity is inaccessible")

    proc = SimpleNamespace(_proc=SimpleNamespace(create_time=raw), create_time=lambda: 456.25)
    with pytest.raises(PermissionError):
        process_create_time(proc)


def test_an_unavailable_public_read_propagates_to_the_caller():
    def public():
        raise ProcessLookupError("process has exited")

    with pytest.raises(ProcessLookupError):
        process_create_time(SimpleNamespace(create_time=public))
