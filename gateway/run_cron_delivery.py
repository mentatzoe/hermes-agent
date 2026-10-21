"""Drain cron delivery queues for exactly the homes the gateway's ticker serves.

Adapter multiplexing controls connections, not cron ownership. A credentialless
satellite may use a primary adapter only through an explicit matching route.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("gateway.run")


def cron_delivery_scopes(runner) -> list:
    """Match cron's profile enumeration and independent-gateway stand-down policy."""
    from gateway import run
    from hermes_constants import get_process_hermes_home, hermes_home_key

    if runner is None:
        return [(None, None)]
    config = getattr(runner, "config", None)
    launch = hermes_home_key(get_process_hermes_home())
    scopes = [(None, None)]
    try:
        for name, home in run._cron_tick_profile_homes(config):
            if hermes_home_key(home) == launch or not run._cron_profile_gate(name, home):
                continue
            scopes.append((name, home))
    except Exception:
        logger.warning("Could not enumerate cron delivery homes", exc_info=True)
    return scopes


def drain_restart_safe_cron_deliveries(adapters, loop, runner=None) -> None:
    """Queue rows are claimed under their owning profile and sent via its allowed adapters."""
    from gateway import run
    from cron import scheduler, scheduler_preflight as preflight

    if runner is None:
        if adapters is not None:
            scheduler.drain_delivery_queue(adapters, loop)
        return
    for profile_name, profile_home in cron_delivery_scopes(runner):
        profile_adapters = (adapters if profile_name is None else
                            getattr(runner, "_profile_adapters", {}).get(profile_name, {}))
        if profile_adapters is None:
            continue
        with run._profile_runtime_scope(profile_home or run.get_hermes_home()):
            if profile_name is not None and not profile_adapters and adapters:
                profile_adapters = preflight.SharedRouteAdapters(
                    adapters, preflight._primary_profile_routes_for_current_home())
            scheduler.drain_delivery_queue(profile_adapters, loop)
