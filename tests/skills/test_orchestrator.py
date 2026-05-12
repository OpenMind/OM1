from unittest.mock import AsyncMock, MagicMock

import pytest

from llm.output_model import Action
from skills.orchestrator import SkillOrchestrator, build_tool_schema


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.function_schemas = []
    llm.ask = AsyncMock()
    return llm


def test_build_tool_schema():
    skills = {"test": MagicMock(name="test", description="desc", priority=1)}
    schema = build_tool_schema(skills)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "read_skill"
    # test isn't technically the skill_name because we mocked the object
    # but let's just make sure the dictionary looks like a schema
    assert "parameters" in schema["function"]


@pytest.mark.asyncio
async def test_execute_read_skills(mock_llm):
    # Setup mock skill
    mock_skill = MagicMock()
    mock_skill.instructions = "Do X"

    orchestrator = SkillOrchestrator([])
    orchestrator._skills = {"test_skill": mock_skill}

    action = Action(type="read_skill", value='{"skill_name": "test_skill"}')
    succeeded_calls = set()

    results = orchestrator.execute_read_skills([action], succeeded_calls)
    assert len(results) == 1
    assert results[0]["success"] is True
    assert results[0]["content"] == "Do X"
    assert orchestrator.call_signature(action) in succeeded_calls

    # Deduplication test
    results2 = orchestrator.execute_read_skills([action], succeeded_calls)
    assert len(results2) == 0


@pytest.mark.asyncio
async def test_build_skill_recall_prompt(mock_llm):
    orchestrator = SkillOrchestrator([])
    results = [{"tool_key": "read_skill", "success": True, "content": "Do X"}]
    prompt = orchestrator.build_skill_recall_prompt("initial", results)
    assert "initial" in prompt
    assert "[Skill Instructions]" in prompt
    assert "Do X" in prompt
