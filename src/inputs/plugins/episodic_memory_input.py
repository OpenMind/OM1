import time
from typing import List, Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.episodic_memory_provider import EpisodicMemoryProvider
from providers.io_provider import IOProvider


class EpisodicMemoryInputConfig(SensorConfig):
    """Configuration for EpisodicMemoryInput. No extra fields needed."""

    pass


class EpisodicMemoryInput(FuserInput[EpisodicMemoryInputConfig, Optional[str]]):
    """
    Input plugin that injects relevant past episodic memories into the LLM prompt.

    Follows the same pattern as MockInput:
      _poll()                   -> get raw value (voice query string)
      _raw_to_text()            -> convert to Optional[Message]
      raw_to_text()             -> append to self.messages
      formatted_latest_buffer() -> format + add_input to IOProvider + clear buffer
    """

    def __init__(self, config: EpisodicMemoryInputConfig):
        super().__init__(config)
        self.descriptor_for_LLM = "EpisodicMemory"
        self.messages: List[Message] = []
        self.io_provider = IOProvider()
        self.provider = EpisodicMemoryProvider()

    async def _poll(self) -> Optional[str]:
        """Return the latest voice input if it arrived this tick, else None."""
        voice_input = self.io_provider.get_input("Voice")
        if (
            voice_input
            and voice_input.input
            and self.io_provider.tick_counter == voice_input.tick
        ):
            return voice_input.input.strip()
        return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """Recall relevant past episodes and format them into a Message."""
        if not raw_input:
            return None

        episodes = await self.provider.recall(raw_input, top_k=3)
        if not episodes:
            return None

        formatted = self._format_episodes(episodes)
        if not formatted:
            return None

        return Message(timestamp=time.time(), message=formatted)

    async def raw_to_text(self, raw_input: Optional[str]) -> None:
        """Convert raw poll result to a Message and append to buffer."""
        if raw_input is None:
            return

        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and clear the latest buffer, and register with IOProvider."""
        if not self.messages:
            return None

        latest = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n"
            f"// START\n"
            f"{latest.message}\n"
            f"// END\n"
        )

        self.io_provider.add_input(self.descriptor_for_LLM, latest.message, time.time())

        self.messages = []
        return result

    def _format_episodes(self, episodes: list) -> Optional[str]:
        """Format a list of episode dicts into a human-readable summary."""
        if not episodes:
            return None

        now = time.time()
        lines: List[str] = []

        for ep in episodes:
            ts = ep.get("timestamp", now)
            mode = ep.get("mode", "unknown")
            voice = ep.get("voice_input", "")
            actions = ep.get("actions", [])

            delta = now - ts
            if delta < 60:
                age = "just now"
            elif delta < 3600:
                age = f"{int(delta / 60)}m ago"
            elif delta < 86400:
                age = f"{int(delta / 3600)}h ago"
            else:
                age = f"{int(delta / 86400)}d ago"

            acts = (
                ", ".join(
                    f"{a.get('type', '')}('{a.get('value', '')}')"
                    for a in actions
                    if a.get("type") and a.get("value")
                )
                or "no actions"
            )

            lines.append(f"[{age}, {mode}] Said: '{voice}'\n  Robot did: {acts}")

        return "\n".join(lines)
