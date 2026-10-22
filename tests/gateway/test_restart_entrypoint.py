"""Restart admission uses the running host across imported/module gateway entrypoints."""
import asyncio
import runpy
from unittest.mock import patch

import pytest

import gateway.run as gateway_run
import gateway.restart as restart
from gateway.config import GatewayConfig
from tools.restart_wait import launch_block_reason


@pytest.mark.parametrize("module_executed", [False, True])
def test_imported_and_module_executed_runner_close_launch_admission(monkeypatch, module_executed):
    monkeypatch.setattr(restart, "_active_gateway_runner_ref", None, raising=False)
    # Execute definitions under a different module identity, as python -m does,
    # without entering __main__'s service-starting branch.
    runner_class = gateway_run.GatewayRunner
    if module_executed:
        with pytest.warns(RuntimeWarning, match="found in sys.modules"):
            namespace = runpy.run_module("gateway.run", run_name="gateway_entrypoint_probe")
        runner_class = namespace["GatewayRunner"]
    runner = None
    try:
        with patch.object(gateway_run, "_gateway_runner_ref", lambda: None):
            runner = runner_class(GatewayConfig())
            runner._restart_requested = True
            assert launch_block_reason() is not None
    finally:
        if runner is not None:
            runner.session_store.close_all_db_handles()
            db = getattr(runner, "_session_db", None)
            if db is not None:
                asyncio.run(db.close())
