import importlib
import inspect
import os
from typing import Type

import pytest

from simulators.base import Simulator, SimulatorConfig


def get_all_simulator_classes():
    """
    Dynamically discover all simulator plugin classes.
    Returns a list of simulator classes that inherit from the base Simulator class.
    """
    plugins_dir = os.path.join("src", "simulators", "plugins")
    plugin_files = [f[:-3] for f in os.listdir(plugins_dir) if f.endswith(".py")]
    simulator_classes = []

    for plugin in plugin_files:
        module = importlib.import_module(f"simulators.plugins.{plugin}")
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Simulator) and obj != Simulator:
                simulator_classes.append(obj)

    return simulator_classes


@pytest.mark.parametrize("simulator_class", get_all_simulator_classes())
def test_init_signature(simulator_class: Type[Simulator]):
    """Verify __init__ signature matches base class"""
    base_params = set(Simulator.__init__.__annotations__.keys())
    impl_params = set(simulator_class.__init__.__annotations__.keys())
    assert (
        base_params == impl_params
    ), f"{simulator_class.__name__} __init__ signature mismatch"


@pytest.mark.parametrize("simulator_class", get_all_simulator_classes())
def test_tick_signature(simulator_class: Type[Simulator]):
    """Verify tick method signature matches base class"""
    base_params = set(Simulator.tick.__annotations__.keys())
    impl_params = set(simulator_class.tick.__annotations__.keys())
    assert (
        base_params == impl_params
    ), f"{simulator_class.__name__} tick signature mismatch"


@pytest.mark.parametrize("simulator_class", get_all_simulator_classes())
def test_sim_signature(simulator_class: Type[Simulator]):
    """Verify sim method signature matches base class"""
    base_params = set(Simulator.sim.__annotations__.keys())
    impl_params = set(simulator_class.sim.__annotations__.keys())
    assert (
        base_params == impl_params
    ), f"{simulator_class.__name__} sim signature mismatch"


@pytest.mark.parametrize("simulator_class", get_all_simulator_classes())
def test_return_type_annotations(simulator_class: Type[Simulator]):
    """Verify return type annotations match base class"""
    methods_to_check = ["tick", "sim"]

    for method_name in methods_to_check:
        base_return = Simulator.__dict__[method_name].__annotations__.get("return")
        impl_return = simulator_class.__dict__[method_name].__annotations__.get(
            "return"
        )

        if base_return is not None:  # Only check if base class specifies return type
            assert impl_return == base_return, (
                f"{simulator_class.__name__}.{method_name} return type mismatch. "
                f"Expected {base_return}, got {impl_return}"
            )


@pytest.mark.parametrize("simulator_class", get_all_simulator_classes())
def test_docstring_exists(simulator_class: Type[Simulator]):
    """Verify all simulator classes and their methods have docstrings"""
    assert (
        simulator_class.__doc__ is not None
    ), f"{simulator_class.__name__} missing class docstring"

    methods_to_check = ["tick", "sim"]
    for method_name in methods_to_check:
        method = getattr(simulator_class, method_name)
        assert (
            method.__doc__ is not None
        ), f"{simulator_class.__name__}.{method_name} missing docstring"


@pytest.mark.parametrize("simulator_class", get_all_simulator_classes())
def test_simulator_instantiation(simulator_class: Type[Simulator]):
    """Test that each simulator can be instantiated with config"""
    config = SimulatorConfig(name=simulator_class.__name__)
    simulator = simulator_class(config)
    assert simulator.name == simulator_class.__name__
    assert simulator.config == config
