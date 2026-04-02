import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SkillEntry:
    """A parsed skill definition.

    Parameters
    ----------
    name : str
        Unique skill identifier.
    description : str
        Short description shown in the prompt catalog.
    instructions : str
        Full Markdown body.
    source_path : Path
        Absolute path to the ``SKILL.md`` file.
    requires_tools : list[str]
        tools required by this skill.
    max_rounds : int
        Max execution rounds for this skill.
    priority : int
        Priority among all skills.
    """

    name: str
    description: str
    instructions: str
    source_path: Path
    requires_tools: List[str] = field(default_factory=list)
    max_rounds: int = 8
    priority: int = 10


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def _parse_frontmatter(content: str) -> tuple:
    """Split a SKILL.md file into metadata and body.

    Uses a simple line-based parser that handles flat key: value pairs
    and list items. No external dependencies required.

    Returns
    -------
    tuple[dict, str]
        Parsed metadata dictionary and the Markdown body.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    yaml_block = match.group(1)
    body = match.group(2).strip()

    meta: Dict = {}
    current_key: Optional[str] = None
    current_list: Optional[List[str]] = None

    for line in yaml_block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- ") and current_key:
            value = stripped[2:].strip().strip('"').strip("'")
            if current_list is not None:
                current_list.append(value)
            continue

        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if value:
                meta[key] = value
                current_key = key
                current_list = None
            else:
                current_key = key
                current_list = []
                meta[key] = current_list

    return meta, body


class SkillLoader:
    """Scan and parse SKILL.md files from a skills directory.

    Parameters
    ----------
    skills_dir : str
        Root directory containing skill subdirectories.
    """

    def __init__(self, skills_dir: str) -> None:
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, SkillEntry] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Scan and parse all SKILL.md files."""
        if not self.skills_dir.exists():
            logging.warning(f"Skills directory does not exist: {self.skills_dir}")
            return

        count = 0
        for child in sorted(self.skills_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.exists():
                continue
            try:
                entry = self._parse_skill(skill_file)
                self.skills[entry.name] = entry
                count += 1
                logging.info(f"Loaded skill: {entry.name} ({skill_file})")
            except Exception as exc:
                logging.error(f"Failed to load skill from {skill_file}: {exc}")

        logging.info(f"SkillLoader: {count} skill(s) loaded from {self.skills_dir}")

    def _parse_skill(self, path: Path) -> SkillEntry:
        """Parse a single SKILL.md file into a SkillEntry."""
        content = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(content)

        name = meta.get("name")
        if not name:
            name = path.parent.name

        description = meta.get("description", "")

        requires_tools = meta.get("requires_tools", [])
        if not isinstance(requires_tools, list):
            requires_tools = []

        max_rounds = meta.get("max_rounds", 8)
        try:
            max_rounds = int(max_rounds)
        except (ValueError, TypeError):
            max_rounds = 8

        priority = meta.get("priority", 10)
        try:
            priority = int(priority)
        except (ValueError, TypeError):
            priority = 10

        return SkillEntry(
            name=name,
            description=description,
            instructions=body,
            source_path=path,
            requires_tools=requires_tools,
            max_rounds=max_rounds,
            priority=priority,
        )

    def reload(self) -> None:
        """Re-scan and reload all skills from disk."""
        self.skills.clear()
        self._load_all()
