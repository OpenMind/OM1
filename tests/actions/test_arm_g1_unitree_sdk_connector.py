import sys
import types
import logging

import pytest


def _install_fake_unitree_sdk(monkeypatch):
    """Install a minimal fake `unitree` SDK module tree so importing the connector works in CI."""

    unitree = types.ModuleType("unitree")
    unitree_sdk2py = types.ModuleType("unitree.unitree_sdk2py")
    g1 = types.ModuleType("unitree.unitree_sdk2py.g1")
    arm = types.ModuleType("unitree.unitree_sdk2py.g1.arm")
    mod = types.ModuleType("unitree.unitree_sdk2py.g1.arm.g1_arm_action_client")

    class G1ArmActionClient:  # noqa: N801 (keep upstream class name)
        def SetTimeout(self, *_args, **_kwargs):
            return None

        def Init(self, *_args, **_kwargs):
            return None

        def ExecuteAction(self, *_args, **_kwargs):
            return None

    mod.G1ArmActionClient = G1ArmActionClient

    for m in [unitree, unitree_sdk2py, g1, arm, mod]:
        monkeypatch.setitem(sys.modules, m.__name__, m)


@pytest.mark.asyncio
async def test_arm_g1_connector_logs_action(monkeypatch, caplog):
    _install_fake_unitree_sdk(monkeypatch)

    from actions.base import ActionConfig
    from actions.arm_g1.connector.unitree_sdk import ARMUnitreeSDKConnector

    # Avoid any side effects in __init__ other than creating the object.
    caplog.set_level(logging.INFO)

    connector = ARMUnitreeSDKConnector(ActionConfig())

    # Build a minimal object with the expected interface.
    dummy = types.SimpleNamespace(action="idle")

    await connector.connect(dummy)

    assert any(
        "Arm command.action: idle" in rec.message for rec in caplog.records
    ), "expected connect() to log the action string"
