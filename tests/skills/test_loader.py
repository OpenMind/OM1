import tempfile
from pathlib import Path

from skills.loader import SkillLoader, _parse_frontmatter


def test_parse_frontmatter():
    content = """---
name: test-skill
description: "A test skill"
requires_tools:
  - mcp_tool_1
max_rounds: 3
priority: 5
---

# Test Body
This is the body.
"""
    meta, body = _parse_frontmatter(content)
    assert meta["name"] == "test-skill"
    assert meta["description"] == "A test skill"
    assert meta["requires_tools"] == ["mcp_tool_1"]
    assert meta["max_rounds"] == "3"
    assert meta["priority"] == "5"
    assert "This is the body." in body


def test_skill_loader():
    with tempfile.TemporaryDirectory() as temp_dir:
        skill_dir = Path(temp_dir) / "test_skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("---\nname: test_skill\ndescription: desc\nmax_rounds: 2\n---\nbody")

        loader = SkillLoader(temp_dir)
        assert len(loader.skills) == 1
        skill = loader.skills["test_skill"]
        assert skill.name == "test_skill"
        assert skill.description == "desc"
        assert skill.max_rounds == 2
        assert skill.instructions == "body"
