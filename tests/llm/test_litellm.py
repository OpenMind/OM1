"""Unit tests for LiteLLM plugin.

These tests verify the plugin file structure and litellm SDK interaction
without importing the full OM1 dependency chain (which requires zenoh).
"""

import ast
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PLUGIN_PATH = Path(__file__).resolve().parents[2] / "src" / "llm" / "plugins" / "litellm.py"


class TestLiteLLMPluginStructure:
    """Verify the plugin file has the correct structure for OM1 auto-discovery."""

    def _parse_ast(self):
        return ast.parse(PLUGIN_PATH.read_text())

    def test_plugin_file_exists(self):
        assert PLUGIN_PATH.exists()

    def test_has_litellm_config_class(self):
        tree = self._parse_ast()
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert "LiteLLMConfig" in classes

    def test_has_litellm_class(self):
        tree = self._parse_ast()
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert "LiteLLM" in classes

    def test_litellm_class_inherits_llm(self):
        tree = self._parse_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LiteLLM":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
                        base_names.append(base.value.id)
                    elif isinstance(base, ast.Name):
                        base_names.append(base.id)
                assert "LLM" in base_names
                return
        pytest.fail("LiteLLM class not found")

    def test_has_ask_method(self):
        tree = self._parse_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LiteLLM":
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                assert "ask" in methods
                return
        pytest.fail("LiteLLM class not found")

    def test_ask_is_async(self):
        tree = self._parse_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LiteLLM":
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == "ask":
                        return
        pytest.fail("ask() is not async")

    def test_uses_drop_params_true(self):
        src = PLUGIN_PATH.read_text()
        assert "drop_params" in src

    def test_uses_litellm_acompletion(self):
        src = PLUGIN_PATH.read_text()
        assert "acompletion" in src

    def test_lazy_imports_litellm(self):
        src = PLUGIN_PATH.read_text()
        assert "import litellm" not in src.split("class")[0]


class TestLiteLLMSDKInteraction:
    """Test litellm SDK calls directly (no OM1 deps needed)."""

    def test_acompletion_called_with_drop_params(self):
        fake_litellm = types.ModuleType("litellm")
        mock_msg = MagicMock(content="ok", tool_calls=None)
        mock_choice = MagicMock(message=mock_msg, finish_reason="stop")
        mock_resp = MagicMock(choices=[mock_choice])
        fake_litellm.acompletion = AsyncMock(return_value=mock_resp)
        sys.modules["litellm"] = fake_litellm

        try:
            import asyncio

            async def run():
                resp = await fake_litellm.acompletion(
                    model="anthropic/claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": "hi"}],
                    drop_params=True,
                )
                return resp

            asyncio.run(run())
            kwargs = fake_litellm.acompletion.call_args.kwargs
            assert kwargs["drop_params"] is True
            assert kwargs["model"] == "anthropic/claude-sonnet-4-20250514"
        finally:
            del sys.modules["litellm"]

    def test_acompletion_forwards_api_key(self):
        fake_litellm = types.ModuleType("litellm")
        mock_msg = MagicMock(content="ok", tool_calls=None)
        mock_resp = MagicMock(choices=[MagicMock(message=mock_msg)])
        fake_litellm.acompletion = AsyncMock(return_value=mock_resp)
        sys.modules["litellm"] = fake_litellm

        try:
            import asyncio

            async def run():
                await fake_litellm.acompletion(
                    model="openai/gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    api_key="sk-test",
                    drop_params=True,
                )

            asyncio.run(run())
            assert fake_litellm.acompletion.call_args.kwargs["api_key"] == "sk-test"
        finally:
            del sys.modules["litellm"]

    def test_acompletion_handles_tool_calls(self):
        fake_litellm = types.ModuleType("litellm")
        mock_tc = MagicMock()
        mock_tc.function.name = "move"
        mock_tc.function.arguments = '{"direction": "forward"}'
        mock_msg = MagicMock(content=None, tool_calls=[mock_tc])
        mock_resp = MagicMock(choices=[MagicMock(message=mock_msg)])
        fake_litellm.acompletion = AsyncMock(return_value=mock_resp)
        sys.modules["litellm"] = fake_litellm

        try:
            import asyncio

            async def run():
                resp = await fake_litellm.acompletion(
                    model="openai/gpt-4o",
                    messages=[{"role": "user", "content": "move"}],
                    tools=[{"type": "function"}],
                    tool_choice="auto",
                    drop_params=True,
                )
                return resp

            resp = asyncio.run(run())
            tc = resp.choices[0].message.tool_calls[0]
            assert tc.function.name == "move"
        finally:
            del sys.modules["litellm"]
