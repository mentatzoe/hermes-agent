"""Discord heartbeat stacks containing percent signs must remain readable."""
import logging

import pytest


@pytest.mark.parametrize("stack", ["SQL LIKE '%voice%'", "format %s %d", "mapping %(key)s"])
def test_heartbeat_stack_survives_logging_setup(tmp_path, stack):
    from hermes_logging import setup_logging

    logger = logging.getLogger("discord.gateway")
    old_filters = logger.filters[:]
    try:
        setup_logging(hermes_home=tmp_path, force=True, mode="gateway")
        setup_logging(hermes_home=tmp_path, force=True, mode="gateway")
        record = logging.LogRecord(
            logger.name, logging.WARNING, __file__, 1,
            "Shard ID %s heartbeat blocked for more than %s seconds.\n" + stack,
            (None, 10), None,
        )
        assert logger.filter(record)
        assert stack in record.getMessage()
        assert "10" in record.getMessage()
        assert len(logger.filters) == len(old_filters) + 1
    finally:
        logger.filters[:] = old_filters


def test_well_formed_heartbeat_stays_formatted(tmp_path):
    from hermes_logging import setup_logging

    setup_logging(hermes_home=tmp_path, force=True, mode="gateway")
    logger = logging.getLogger("discord.gateway")
    record = logging.LogRecord(logger.name, logging.WARNING, __file__, 1,
                               "Shard %s blocked %s seconds", (None, 10), None)
    assert logger.filter(record)
    assert record.getMessage() == "Shard None blocked 10 seconds"
