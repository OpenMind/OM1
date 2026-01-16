from typing import Type

import pytest

from llm import LLM


def get_all_llm_classes():
    import importlib
    import inspect
    import os

    plugins_dir = os.path.join("src", "llm", "plugins")
    plugin_files = [f[:-3] for f in os.listdir(plugins_dir) if f.endswith(".py")]

    llm_classes = []
    for plugin in plugin_files:
        module = importlib.import_module(f"llm.plugins.{plugin}")
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, LLM) and obj != LLM:
                llm_classes.append(obj)
    return llm_classes


@pytest.mark.parametrize("llm_class", get_all_llm_classes())
def test_init_signature(llm_class: Type[LLM]):
    # Verify __init__ signature matches base class
    base_params = set(LLM.__init__.__annotations__.keys())
    impl_params = set(llm_class.__init__.__annotations__.keys())
    assert (
        base_params == impl_params
    ), f"{llm_class.__name__} __init__ signature mismatch"


@pytest.mark.parametrize("llm_class", get_all_llm_classes())
def test_ask_signature(llm_class: Type[LLM]):
    # Verify ask method signature matches base class
    base_params = set(LLM.ask.__annotations__.keys())
    impl_params = set(llm_class.ask.__annotations__.keys())
    assert base_params == impl_params, f"{llm_class.__name__} ask signature mismatch"
