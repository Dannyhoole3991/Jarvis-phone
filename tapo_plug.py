"""
Local (LAN, no cloud) control of the Tapo P100 that switches mains power
to Danny's main PC and other equipment.

This exists to let Jarvis Phone (always-on, on the Ubuntu server) turn
the PC back on when it's fully powered off -- something the PC's own
Jarvis_FINAL_WORKING.py obviously can't do for itself, since it isn't
running yet at that point.

Uses the mihai-dinculescu/tapo Python library (pip: "tapo") for direct
local device control -- confirmed live that python-kasa cannot talk to
this exact P100 firmware at all (it advertises a newer "TPAP" auth
scheme python-kasa doesn't recognize), while this library does support
it, PROVIDED:
  1. "Third-Party Compatibility" is enabled in the Tapo app, on the
     account that actually owns this device (Me > Third-Party Services).
  2. TAPO_USERNAME/TAPO_PASSWORD (env vars, see /etc/jarvis-phone.env)
     are that same owning account's Tapo login -- confirmed live that a
     different (non-owning) Tapo account's credentials fail with a
     HASH_MISMATCH error, distinct from the FORBIDDEN error you get when
     Third-Party Compatibility itself isn't the problem.

Deliberately exposes only get_status() and turn_on() here -- Danny was
explicit that this must never turn the plug off (it's live mains power
to his PC and other equipment), so there is intentionally no turn_off()
anywhere in this file for anything to accidentally call.
"""

import asyncio
import os

from tapo import ApiClient

PLUG_IP = "192.168.0.203"


async def _connect():
    username = os.environ["TAPO_USERNAME"]
    password = os.environ["TAPO_PASSWORD"]
    client = ApiClient(username, password)
    return await client.p100(PLUG_IP)


async def _get_status_async():
    device = await _connect()
    info = await device.get_device_info()
    return info.to_dict()


async def _turn_on_async():
    device = await _connect()
    await device.on()
    info = await device.get_device_info()
    return info.to_dict()


def get_status():
    """Read-only. Raises on any connection/auth failure -- callers decide how to report it."""
    return asyncio.run(_get_status_async())


def turn_on():
    """
    Idempotent -- safe to call even if the plug is already on (which is
    the common case: most callers of this just want to GUARANTEE the PC
    ends up on, not specifically toggle it).
    """
    return asyncio.run(_turn_on_async())
