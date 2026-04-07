import asyncio
import logging
import time
from queue import Empty, Queue
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from pydantic import Field


class ScheduledCronInputConfig(SensorConfig):
    """Configuration for the ScheduledCronInput plugin."""

    input_name: str = Field(
        default="User Async Task", description="Label shown to the LLM for this input"
    )


class ScheduledCronInput(FuserInput[ScheduledCronInputConfig, Optional[str]]):
    """
    Programmatic input plugin that lets other runtime components (e.g.
    ExecuteCronJobProvider) inject a natural-language command directly into
    the fuser pipeline so that it is processed by the LLM on the next tick.

    The shared queue is a class-level attribute so any code that holds
    a reference to the *class* can push messages without needing an
    instance reference:

        ScheduledCronInput.inject("turn on the living-room lights")
    """

    # Class-level queue shared across all instances and callers.
    _shared_queue: Queue = Queue()

    # Set to True when formatted_latest_buffer() returns a message this tick,
    # so cortex can filter schedule_cron_job from the LLM response and prevent
    # cron-triggered ticks from re-registering the same job.
    cron_triggered: bool = False

    def __init__(self, config: ScheduledCronInputConfig):
        super().__init__(config)
        self.messages: list[Message] = []
        self.descriptor_for_LLM = self.config.input_name
        self.io_provider = IOProvider()

    # ------------------------------------------------------------------
    # Public class-method API used by ExecuteCronJobProvider
    # ------------------------------------------------------------------

    @classmethod
    def inject(cls, message: str) -> None:
        """Enqueue *message* and wake the cortex loop immediately.

        Parameters
        ----------
        message:
            The natural-language command or question to inject.
        """
        cls._shared_queue.put(message)
        logging.info("ScheduledCronInput: injected message: %s", message)
        SleepTickerProvider().skip_sleep = True

    # ------------------------------------------------------------------
    # FuserInput interface
    # ------------------------------------------------------------------

    async def _poll(self) -> Optional[str]:
        await asyncio.sleep(0.1)
        try:
            return self._shared_queue.get_nowait()
        except Empty:
            return None

    async def raw_to_text(self, raw_input: Optional[str]) -> None:
        if raw_input is None:
            return
        self.messages.append(Message(timestamp=time.time(), message=raw_input))

    def formatted_latest_buffer(self) -> Optional[str]:
        if not self.messages:
            ScheduledCronInput.cron_triggered = False
            return None

        msg = self.messages[-1]
        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n"
            f"// START\n{msg.message}\n// END\n"
        )
        self.io_provider.add_input(
            self.descriptor_for_LLM, msg.message, time.time()
        )
        self.messages = []
        ScheduledCronInput.cron_triggered = True
        return result
