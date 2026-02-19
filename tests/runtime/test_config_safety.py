"""Tests for runtime config with safety sandbox."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.runtime.config import ModeConfig, ModeSystemConfig, _load_mode_components


def _make_mode_config(raw_safety_sandbox=None):
    return ModeConfig(
        version="1.0.1",
        name="test_mode",
        display_name="Test Mode",
        description="Test",
        system_prompt_base="You are a test.",
        _raw_safety_sandbox=raw_safety_sandbox or {},
        _raw_inputs=[],
        _raw_simulators=[],
        _raw_actions=[],
        _raw_backgrounds=[],
        _raw_llm={"type": "OpenAILLM", "config": {"agent_name": "test"}},
    )


def _make_system_config():
    return ModeSystemConfig(
        version="1.0.1",
        name="test_system",
        default_mode="test_mode",
        api_key="test_key",
        global_cortex_llm={"type": "OpenAILLM", "config": {"agent_name": "test"}},
    )


def _make_provider_modules(
    mock_sandbox,
    mock_robot_state,
    mock_env_model,
    mock_teleops,
    odom_cls=None,
    lidar_cls=None,
):
    return {
        "providers.safety_sandbox_provider": MagicMock(
            SafetySandboxProvider=MagicMock(return_value=mock_sandbox)
        ),
        "providers.robot_state_provider": MagicMock(
            RobotStateProvider=MagicMock(return_value=mock_robot_state)
        ),
        "providers.environment_model_provider": MagicMock(
            EnvironmentModelProvider=MagicMock(return_value=mock_env_model)
        ),
        "providers.teleops_status_provider": MagicMock(
            TeleopsStatusProvider=MagicMock(return_value=mock_teleops)
        ),
        "providers.unitree_go2_odom_provider": MagicMock(
            UnitreeGo2OdomProvider=odom_cls or MagicMock()
        ),
        "providers.unitree_go2_rplidar_provider": MagicMock(
            UnitreeGo2RPLidarProvider=lidar_cls or MagicMock()
        ),
    }


def _minimal_config(extra_top="", extra_mode=""):
    return f"""
    {{
        version: "v1.0.2", default_mode: "test", api_key: "dummy",
        system_governance: "Be safe.",
        cortex_llm: {{ type: "OpenAILLM", config: {{ agent_name: "test" }} }},
        {extra_top}
        modes: {{
            test: {{
                display_name: "Test", description: "Test mode",
                system_prompt_base: "You are a test.", hertz: 1,
                agent_inputs: [], agent_actions: []
                {extra_mode}
            }}
        }}
    }}
    """


def test_load_schema_file_not_found():
    from src.runtime.config import _load_schema

    with patch("src.runtime.config.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Schema file not found"):
            _load_schema("nonexistent_schema.json")


def test_validate_config_schema_file_not_found():
    from src.runtime.config import validate_config_schema

    with patch(
        "src.runtime.config._load_schema", side_effect=FileNotFoundError("no file")
    ):
        with pytest.raises(FileNotFoundError):
            validate_config_schema({"modes": {}, "default_mode": "x"})


def test_validate_config_schema_validation_error():
    from jsonschema import ValidationError

    from src.runtime.config import validate_config_schema

    with patch("src.runtime.config._load_schema", return_value={}):
        with patch(
            "src.runtime.config.validate",
            side_effect=ValidationError("bad field", path=["field1"]),
        ):
            with pytest.raises(ValidationError):
                validate_config_schema({"modes": {}, "default_mode": "x"})


def test_validate_config_schema_validation_error_no_path():
    from jsonschema import ValidationError

    from src.runtime.config import validate_config_schema

    with patch("src.runtime.config._load_schema", return_value={}):
        with patch(
            "src.runtime.config.validate", side_effect=ValidationError("bad", path=[])
        ):
            with pytest.raises(ValidationError):
                validate_config_schema({"modes": {}, "default_mode": "x"})


def test_add_meta_all_fields():
    from src.runtime.config import add_meta

    result = add_meta({}, "key", "eth0", "urid123", "192.168.1.1", "mode_a")
    assert result["api_key"] == "key"
    assert result["unitree_ethernet"] == "eth0"
    assert result["URID"] == "urid123"
    assert result["robot_ip"] == "192.168.1.1"
    assert result["mode"] == "mode_a"


def test_add_meta_skips_existing_fields():
    from src.runtime.config import add_meta

    config = {"api_key": "existing", "URID": "existing_urid"}
    result = add_meta(config, "new_key", "eth0", "new_urid", "1.2.3.4")
    assert result["api_key"] == "existing"
    assert result["URID"] == "existing_urid"
    assert result["unitree_ethernet"] == "eth0"


def test_to_runtime_config_no_llm_raises():
    mode_config = _make_mode_config()
    mode_config.cortex_llm = None
    with pytest.raises(ValueError, match="No LLM configured for mode"):
        mode_config.to_runtime_config(_make_system_config())


def test_to_runtime_config_success():
    from src.runtime.config import RuntimeConfig

    mode_config = _make_mode_config()
    mode_config.cortex_llm = MagicMock()
    result = mode_config.to_runtime_config(_make_system_config())
    assert isinstance(result, RuntimeConfig)
    assert result.mode == "test_mode"


@patch("src.runtime.config._load_mode_components")
def test_mode_config_load_components(mock_load):
    mode_config = _make_mode_config()
    system_config = _make_system_config()
    mode_config.load_components(system_config)
    mock_load.assert_called_once_with(mode_config, system_config)


@pytest.mark.asyncio
async def test_mode_config_execute_lifecycle_hooks():
    from src.runtime.hook import LifecycleHookType

    mode_config = _make_mode_config()
    with patch(
        "src.runtime.config.execute_lifecycle_hooks",
        new_callable=AsyncMock,
        return_value=True,
    ):
        result = await mode_config.execute_lifecycle_hooks(LifecycleHookType.ON_ENTRY)
    assert result is True


@pytest.mark.asyncio
async def test_mode_config_execute_lifecycle_hooks_with_context():
    from src.runtime.hook import LifecycleHookType

    mode_config = _make_mode_config()
    with patch(
        "src.runtime.config.execute_lifecycle_hooks",
        new_callable=AsyncMock,
        return_value=False,
    ):
        result = await mode_config.execute_lifecycle_hooks(
            LifecycleHookType.ON_ENTRY, context={"extra": "value"}
        )
    assert result is False


@pytest.mark.asyncio
async def test_execute_global_lifecycle_hooks():
    from src.runtime.hook import LifecycleHookType

    system_config = _make_system_config()
    with patch(
        "src.runtime.config.execute_lifecycle_hooks",
        new_callable=AsyncMock,
        return_value=True,
    ):
        result = await system_config.execute_global_lifecycle_hooks(
            LifecycleHookType.ON_ENTRY
        )
    assert result is True


def test_load_mode_config_with_safety_sandbox():
    from src.runtime.config import load_mode_config

    config_content = """
    {
        version: "v1.0.2", default_mode: "test", api_key: "dummy",
        system_governance: "Be safe.",
        cortex_llm: { type: "OpenAILLM", config: { agent_name: "test" } },
        modes: {
            test: {
                display_name: "Test", description: "Test mode",
                system_prompt_base: "You are a test.", hertz: 1,
                safety_sandbox: {
                    enabled: true, simulator: "WebSim",
                    simulation_timeout: 2.0, obstacle_margin: 0.3
                },
                agent_inputs: [], agent_actions: []
            }
        }
    }
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
        f.write(config_content)
        tmp_path = f.name
    try:
        config = load_mode_config("dummy", mode_source_path=tmp_path)
        assert config.modes["test"]._raw_safety_sandbox == {
            "enabled": True,
            "simulator": "WebSim",
            "simulation_timeout": 2.0,
            "obstacle_margin": 0.3,
        }
    finally:
        os.unlink(tmp_path)


def test_load_mode_config_without_safety_sandbox():
    from src.runtime.config import load_mode_config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
        f.write(_minimal_config())
        tmp_path = f.name
    try:
        config = load_mode_config("dummy", mode_source_path=tmp_path)
        assert config.modes["test"]._raw_safety_sandbox == {}
    finally:
        os.unlink(tmp_path)


def test_load_mode_config_invalid_json5():
    from src.runtime.config import load_mode_config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
        f.write("{ this is not valid json5 !!!: }")
        tmp_path = f.name
    try:
        with pytest.raises(ValueError, match="Failed to parse configuration file"):
            load_mode_config("dummy", mode_source_path=tmp_path)
    finally:
        os.unlink(tmp_path)


def test_load_mode_config_env_var_fallbacks():
    """OM_API_KEY env var is used when api_key='openmind_free'. Covers lines 578-579."""
    from src.runtime.config import load_mode_config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
        f.write(_minimal_config().replace('"dummy"', '"openmind_free"'))
        tmp_path = f.name
    try:
        with patch.dict(os.environ, {"OM_API_KEY": "env_key_from_env"}):
            with patch("src.runtime.config.validate_config_schema"):
                config = load_mode_config("dummy", mode_source_path=tmp_path)
        assert config.api_key == "env_key_from_env"
    finally:
        os.unlink(tmp_path)


def test_load_mode_config_with_unitree_ethernet():
    from src.runtime.config import load_mode_config

    config_content = """
    {
        version: "v1.0.2", default_mode: "test", api_key: "dummy",
        system_governance: "Be safe.", unitree_ethernet: "eth0",
        cortex_llm: { type: "OpenAILLM", config: { agent_name: "test" } },
        modes: {
            test: {
                display_name: "Test", description: "Test mode",
                system_prompt_base: "You are a test.", hertz: 1,
                agent_inputs: [], agent_actions: []
            }
        }
    }
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
        f.write(config_content)
        tmp_path = f.name
    try:
        with patch("src.runtime.config.load_unitree") as mock_load_unitree:
            config = load_mode_config("dummy", mode_source_path=tmp_path)
        mock_load_unitree.assert_called_once_with("eth0")
        assert config.unitree_ethernet == "eth0"
    finally:
        os.unlink(tmp_path)


def test_load_mode_config_with_transition_rules():
    from src.runtime.config import load_mode_config

    config_content = """
    {
        version: "v1.0.2", default_mode: "mode_a", api_key: "dummy",
        system_governance: "Be safe.",
        cortex_llm: { type: "OpenAILLM", config: { agent_name: "test" } },
        modes: {
            mode_a: {
                display_name: "Mode A", description: "First mode",
                system_prompt_base: "You are mode A.", hertz: 1,
                agent_inputs: [], agent_actions: []
            },
            mode_b: {
                display_name: "Mode B", description: "Second mode",
                system_prompt_base: "You are mode B.", hertz: 1,
                agent_inputs: [], agent_actions: []
            }
        },
        transition_rules: [{
            from_mode: "mode_a", to_mode: "mode_b",
            transition_type: "input_triggered",
            trigger_keywords: ["switch", "change"],
            priority: 2, cooldown_seconds: 5.0
        }]
    }
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
        f.write(config_content)
        tmp_path = f.name
    try:
        config = load_mode_config("dummy", mode_source_path=tmp_path)
        rule = config.transition_rules[0]
        assert rule.from_mode == "mode_a"
        assert rule.to_mode == "mode_b"
        assert rule.trigger_keywords == ["switch", "change"]
        assert rule.priority == 2
        assert rule.cooldown_seconds == 5.0
    finally:
        os.unlink(tmp_path)


@patch("src.runtime.config.load_llm")
@patch("src.runtime.config.load_background")
@patch("src.runtime.config.load_action")
@patch("src.runtime.config.load_simulator")
@patch("src.runtime.config.load_input")
def test_load_mode_components_no_safety_sandbox(
    mock_input, mock_sim, mock_action, mock_bg, mock_llm
):
    mock_llm.return_value = MagicMock()
    mode_config = _make_mode_config(raw_safety_sandbox={})
    _load_mode_components(mode_config, _make_system_config())
    assert mode_config.safety_sandbox is None


@patch("src.runtime.config.load_llm")
@patch("src.runtime.config.load_background")
@patch("src.runtime.config.load_action")
@patch("src.runtime.config.load_simulator")
@patch("src.runtime.config.load_input")
def test_load_mode_components_safety_sandbox_import_error(
    mock_input, mock_sim, mock_action, mock_bg, mock_llm
):
    mock_llm.return_value = MagicMock()
    mode_config = _make_mode_config(raw_safety_sandbox={"enabled": True})
    with patch.dict("sys.modules", {"providers.safety_sandbox_provider": None}):
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError("mocked"))
                if name == "providers.safety_sandbox_provider"
                else __import__(name, *a, **kw)
            ),
        ):
            _load_mode_components(mode_config, _make_system_config())
    assert mode_config.safety_sandbox is None


@patch("src.runtime.config.load_llm")
@patch("src.runtime.config.load_background")
@patch("src.runtime.config.load_action")
@patch("src.runtime.config.load_simulator")
@patch("src.runtime.config.load_input")
def test_load_mode_components_safety_sandbox_loaded_but_disabled(
    mock_input, mock_sim, mock_action, mock_bg, mock_llm
):
    mock_llm.return_value = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.enabled = False
    mode_config = _make_mode_config(raw_safety_sandbox={"enabled": False})
    with patch.dict(
        "sys.modules",
        {
            "providers.safety_sandbox_provider": MagicMock(
                SafetySandboxProvider=MagicMock(return_value=mock_sandbox)
            )
        },
    ):
        _load_mode_components(mode_config, _make_system_config())
    assert mode_config.safety_sandbox is mock_sandbox


@patch("src.runtime.config.load_llm")
@patch("src.runtime.config.load_background")
@patch("src.runtime.config.load_action")
@patch("src.runtime.config.load_simulator")
@patch("src.runtime.config.load_input")
def test_load_mode_components_safety_sandbox_enabled_providers_init(
    mock_input, mock_sim, mock_action, mock_bg, mock_llm
):
    mock_llm.return_value = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.enabled = True
    mock_robot_state, mock_env_model, mock_teleops = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    mode_config = _make_mode_config(raw_safety_sandbox={"enabled": True})
    with patch.dict(
        "sys.modules",
        _make_provider_modules(
            mock_sandbox, mock_robot_state, mock_env_model, mock_teleops
        ),
    ):
        _load_mode_components(mode_config, _make_system_config())
    mock_robot_state.register_providers.assert_called_once()
    mock_robot_state.start.assert_called_once()
    mock_env_model.register_providers.assert_called_once()
    mock_env_model.start.assert_called_once()


@patch("src.runtime.config.load_llm")
@patch("src.runtime.config.load_background")
@patch("src.runtime.config.load_action")
@patch("src.runtime.config.load_simulator")
@patch("src.runtime.config.load_input")
def test_load_mode_components_providers_init_exception(
    mock_input, mock_sim, mock_action, mock_bg, mock_llm
):
    mock_llm.return_value = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.enabled = True
    mode_config = _make_mode_config(raw_safety_sandbox={"enabled": True})
    with patch.dict(
        "sys.modules",
        {
            "providers.safety_sandbox_provider": MagicMock(
                SafetySandboxProvider=MagicMock(return_value=mock_sandbox)
            ),
            "providers.robot_state_provider": MagicMock(
                RobotStateProvider=MagicMock(side_effect=Exception("fail"))
            ),
            "providers.environment_model_provider": MagicMock(),
            "providers.teleops_status_provider": MagicMock(),
            "providers.unitree_go2_odom_provider": MagicMock(),
            "providers.unitree_go2_rplidar_provider": MagicMock(),
        },
    ):
        _load_mode_components(mode_config, _make_system_config())


@patch("src.runtime.config.load_llm")
@patch("src.runtime.config.load_background")
@patch("src.runtime.config.load_action")
@patch("src.runtime.config.load_simulator")
@patch("src.runtime.config.load_input")
def test_load_mode_components_no_llm_raises(
    mock_input, mock_sim, mock_action, mock_bg, mock_llm
):
    mode_config = _make_mode_config()
    mode_config._raw_llm = None
    system_config = _make_system_config()
    system_config.global_cortex_llm = None
    with pytest.raises(ValueError, match="No LLM configuration found for mode"):
        _load_mode_components(mode_config, system_config)


@patch("src.runtime.config.load_llm")
@patch("src.runtime.config.load_background")
@patch("src.runtime.config.load_action")
@patch("src.runtime.config.load_simulator")
@patch("src.runtime.config.load_input")
def test_load_mode_components_odom_and_lidar_from_backgrounds(
    mock_input, mock_sim, mock_action, mock_bg, mock_llm
):
    mock_llm.return_value = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.enabled = True
    mock_robot_state, mock_env_model, mock_teleops = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    bg = MagicMock()
    bg.unitree_go2_odom_provider = MagicMock()
    bg.unitree_go2_state_provider = MagicMock()
    bg.unitree_go2_amcl_provider = MagicMock()
    bg.unitree_go2_rplidar_provider = MagicMock()
    mock_bg.return_value = bg

    mode_config = _make_mode_config(raw_safety_sandbox={"enabled": True})
    mode_config._raw_backgrounds = [{"type": "FakeBg", "config": {}}]

    with patch.dict(
        "sys.modules",
        _make_provider_modules(
            mock_sandbox, mock_robot_state, mock_env_model, mock_teleops
        ),
    ):
        _load_mode_components(mode_config, _make_system_config())

    kwargs = mock_robot_state.register_providers.call_args[1]
    assert kwargs["odom"] == bg.unitree_go2_odom_provider
    assert kwargs["lidar"] == bg.unitree_go2_rplidar_provider
    assert kwargs["state_prov"] == bg.unitree_go2_state_provider
    assert kwargs["amcl"] == bg.unitree_go2_amcl_provider


@patch("src.runtime.config.load_llm")
@patch("src.runtime.config.load_background")
@patch("src.runtime.config.load_action")
@patch("src.runtime.config.load_simulator")
@patch("src.runtime.config.load_input")
def test_load_mode_components_isinstance_match_from_inputs(
    mock_input, mock_sim, mock_action, mock_bg, mock_llm
):
    mock_llm.return_value = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.enabled = True
    mock_robot_state, mock_env_model, mock_teleops = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    class FakeOdomProvider:
        pass

    class FakeLidarProvider:
        pass

    odom_instance = FakeOdomProvider()
    lidar_instance = FakeLidarProvider()

    inp = MagicMock()
    inp.odom = odom_instance
    inp.lidar = lidar_instance
    mock_input.return_value = inp

    mode_config = _make_mode_config(raw_safety_sandbox={"enabled": True})
    mode_config._raw_inputs = [{"type": "FakeInput", "config": {}}]

    with patch.dict(
        "sys.modules",
        _make_provider_modules(
            mock_sandbox,
            mock_robot_state,
            mock_env_model,
            mock_teleops,
            odom_cls=FakeOdomProvider,
            lidar_cls=FakeLidarProvider,
        ),
    ):
        _load_mode_components(mode_config, _make_system_config())

    kwargs = mock_robot_state.register_providers.call_args[1]
    assert kwargs["odom"] is odom_instance
    assert kwargs["lidar"] is lidar_instance


def test_mode_config_to_dict_success():
    from src.runtime.config import TransitionRule, TransitionType, mode_config_to_dict

    mode = ModeConfig(
        version="1.0.1",
        name="test_mode",
        display_name="Test Mode",
        description="Test",
        system_prompt_base="You are a test.",
        hertz=2.0,
        timeout_seconds=30.0,
        remember_locations=True,
        save_interactions=True,
        _raw_inputs=[{"type": "camera"}],
        _raw_llm={"type": "OpenAILLM"},
        _raw_simulators=[],
        _raw_actions=[],
        _raw_backgrounds=[],
        _raw_lifecycle_hooks=[],
        _raw_safety_sandbox={"enabled": True},
    )
    system = ModeSystemConfig(
        version="1.0.1",
        name="test_system",
        default_mode="test_mode",
        allow_manual_switching=False,
        mode_memory_enabled=False,
        api_key="key123",
        robot_ip="10.0.0.1",
        URID="urid1",
        unitree_ethernet="eth0",
        system_governance="gov",
        system_prompt_examples="examples",
        global_cortex_llm={"type": "OpenAILLM"},
        _raw_global_lifecycle_hooks=[],
    )
    system.modes["test_mode"] = mode
    system.transition_rules.append(
        TransitionRule(
            from_mode="test_mode",
            to_mode="other_mode",
            transition_type=TransitionType.INPUT_TRIGGERED,
            trigger_keywords=["go"],
            priority=1,
            cooldown_seconds=0.0,
        )
    )

    result = mode_config_to_dict(system)
    assert result["name"] == "test_system"
    assert result["default_mode"] == "test_mode"
    assert result["api_key"] == "key123"
    assert result["modes"]["test_mode"]["safety_sandbox"] == {"enabled": True}
    assert result["transition_rules"][0]["from_mode"] == "test_mode"


def test_mode_config_to_dict_exception_returns_empty():
    from src.runtime.config import mode_config_to_dict

    bad_config = MagicMock()
    bad_config.modes.items.side_effect = Exception("boom")
    assert mode_config_to_dict(bad_config) == {}


@patch("src.runtime.config.load_llm")
@patch("src.runtime.config.load_background")
@patch("src.runtime.config.load_action")
@patch("src.runtime.config.load_simulator")
@patch("src.runtime.config.load_input")
def test_load_mode_components_odom_and_lidar_from_inputs(
    mock_input, mock_sim, mock_action, mock_bg, mock_llm
):
    """agent_inputs with odom/lidar → passed to register_providers via hasattr+isinstance."""
    mock_llm.return_value = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.enabled = True
    mock_robot_state, mock_env_model, mock_teleops = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    mock_odom_cls = MagicMock()
    mock_lidar_cls = MagicMock()
    mock_odom_instance = MagicMock(spec=mock_odom_cls)
    mock_lidar_instance = MagicMock(spec=mock_lidar_cls)

    inp = MagicMock(spec=["odom", "lidar"])
    inp.odom = mock_odom_instance
    inp.lidar = mock_lidar_instance

    mode_config = _make_mode_config(raw_safety_sandbox={"enabled": True})
    mode_config.agent_inputs = [inp]

    mock_odom_mod = MagicMock()
    mock_odom_mod.UnitreeGo2OdomProvider = mock_odom_cls
    mock_lidar_mod = MagicMock()
    mock_lidar_mod.UnitreeGo2RPLidarProvider = mock_lidar_cls

    with patch.dict(
        "sys.modules",
        {
            "providers.safety_sandbox_provider": MagicMock(
                SafetySandboxProvider=MagicMock(return_value=mock_sandbox)
            ),
            "providers.robot_state_provider": MagicMock(
                RobotStateProvider=MagicMock(return_value=mock_robot_state)
            ),
            "providers.environment_model_provider": MagicMock(
                EnvironmentModelProvider=MagicMock(return_value=mock_env_model)
            ),
            "providers.teleops_status_provider": MagicMock(
                TeleopsStatusProvider=MagicMock(return_value=mock_teleops)
            ),
            "providers.unitree_go2_odom_provider": mock_odom_mod,
            "providers.unitree_go2_rplidar_provider": mock_lidar_mod,
        },
    ):
        _load_mode_components(mode_config, _make_system_config())

    mock_robot_state.start.assert_called_once()
    mock_env_model.start.assert_called_once()


@patch("src.runtime.config.load_llm")
@patch("src.runtime.config.load_background")
@patch("src.runtime.config.load_action")
@patch("src.runtime.config.load_simulator")
@patch("src.runtime.config.load_input")
def test_load_mode_components_state_and_amcl_from_backgrounds(
    mock_input, mock_sim, mock_action, mock_bg, mock_llm
):
    """Backgrounds with state_provider and amcl_provider are picked up."""
    mock_llm.return_value = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.enabled = True
    mock_robot_state, mock_env_model, mock_teleops = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    bg = MagicMock()
    bg.unitree_go2_odom_provider = MagicMock()
    bg.unitree_go2_state_provider = MagicMock()
    bg.unitree_go2_amcl_provider = MagicMock()
    bg.unitree_go2_rplidar_provider = MagicMock()
    mock_bg.return_value = bg

    mode_config = _make_mode_config(raw_safety_sandbox={"enabled": True})
    mode_config._raw_backgrounds = [{"type": "FakeBg", "config": {}}]

    with patch.dict(
        "sys.modules",
        _make_provider_modules(
            mock_sandbox, mock_robot_state, mock_env_model, mock_teleops
        ),
    ):
        _load_mode_components(mode_config, _make_system_config())

    kwargs = mock_robot_state.register_providers.call_args[1]
    assert kwargs["state_prov"] == bg.unitree_go2_state_provider
    assert kwargs["amcl"] == bg.unitree_go2_amcl_provider
