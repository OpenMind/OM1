import asyncio
import logging
import time
from typing import ClassVar, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider


class ScheduledCronInputConfig(SensorConfig):
    """Configuration for the ScheduledCronInput plugin."""

    input_name: str = Field(default="User Async Task", description="Label shown to the LLM for this input")


class ScheduledCronInput(FuserInput[ScheduledCronInputConfig, Optional[str]]):
    """
    Programmatic input plugin that lets other runtime components (e.g.
    ExecuteCronJobProvider) inject a natural-language command directly into
    the fuser pipeline so that it is processed by the LLM on the next tick.

        ScheduledCronInput.inject("turn on the living-room lights")
    """

    # Reference to the active instance so inject() can write directly to
    # self.messages without going through the poll loop, ensuring the message
    # is visible on the very next cortex tick.
    _instance: ClassVar[Optional["ScheduledCronInput"]] = None

    # Set to True when formatted_latest_buffer() returns a message this tick,
    # so cortex can filter schedule_cron_job from the LLM response and prevent
    # cron-triggered ticks from re-registering the same job.
    cron_triggered: bool = False

    def __init__(self, config: ScheduledCronInputConfig):
        super().__init__(config)
        self.messages: list[Message] = []
        self.descriptor_for_LLM = self.config.input_name
        self.io_provider = IOProvider()
        ScheduledCronInput._instance = self

    # ------------------------------------------------------------------
    # Public class-method API used by ExecuteCronJobProvider
    # ------------------------------------------------------------------

    @classmethod
    def inject(cls, message: str) -> None:
        """Inject *message* directly into the buffer and wake the cortex loop.

        Writes straight to the active instance's message buffer so the message
        is visible to formatted_latest_buffer() on the very next cortex tick.

        Parameters
        ----------
        message:
            The natural-language command or question to inject.
        """
        if cls._instance is None:
            logging.warning("ScheduledCronInput: inject() called before instance created, message dropped: %s", message)
            return
        cls._instance.messages.append(Message(timestamp=time.time(), message=message))
        logging.info("ScheduledCronInput: injected message: %s", message)
        SleepTickerProvider().skip_sleep = True

    # ------------------------------------------------------------------
    # FuserInput interface
    # ------------------------------------------------------------------

    async def _poll(self) -> Optional[str]:
        await asyncio.sleep(0.1)
        return None

    async def raw_to_text(self, raw_input: Optional[str]) -> None:
        """No-op: messages are injected directly via inject() and formatted_latest_buffer()."""
        pass

    def formatted_latest_buffer(self) -> Optional[str]:
        """Return the latest injected message formatted for the LLM, or None if empty."""
        if not self.messages:
            ScheduledCronInput.cron_triggered = False
            return None

        msg = self.messages[-1]
        result = f"\nINPUT: {self.descriptor_for_LLM}\n" f"// START\n{msg.message}\n// END\n"
        self.io_provider.add_input(self.descriptor_for_LLM, msg.message, time.time())
        self.messages = []
        ScheduledCronInput.cron_triggered = True
        return result
