import asyncio
import logging
import os
from typing import List, Optional, Union

import json5

from actions.orchestrator import ActionOrchestrator
from backgrounds.orchestrator import BackgroundOrchestrator
from fuser import Fuser
from inputs.orchestrator import InputOrchestrator
from providers.config_provider import ConfigProvider
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from runtime.single_mode.config import RuntimeConfig, load_config
from simulators.orchestrator import SimulatorOrchestrator

from history_manager import history_from_config


class CortexRuntime:
    """
    Main runtime for single-mode OM1 agent.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        config_name: str,
        hot_reload: bool = True,
        check_interval: float = 60,
    ):
        self.config = config
        self.config_name = config_name
        self.hot_reload = hot_reload
        self.check_interval = check_interval

        self.fuser = Fuser(config)
        self.action_orchestrator = ActionOrchestrator(config)
        self.simulator_orchestrator = SimulatorOrchestrator(config)
        self.background_orchestrator = BackgroundOrchestrator(config)
        self.sleep_ticker_provider = SleepTickerProvider()
        self.io_provider = IOProvider()
        self.config_provider = ConfigProvider()

        self.last_modified: float = 0.0
        self.config_watcher_task: Optional[asyncio.Task] = None
        self.input_listener_task: Optional[asyncio.Task] = None
        self.simulator_task: Optional[Union[asyncio.Task, asyncio.Future]] = None
        self.action_task: Optional[Union[asyncio.Task, asyncio.Future]] = None
        self.background_task: Optional[Union[asyncio.Task, asyncio.Future]] = None
        self.cortex_loop_task: Optional[asyncio.Task] = None

        self._is_reloading = False

        # 🔑 Conversation history (loads from disk if exists)
        self.history = history_from_config(self.config.__dict__, self.config_name)

        if self.hot_reload:
            self.config_path = self._create_runtime_config_file()
            self.last_modified = self._get_file_mtime()

    # ---------------------------------------------------------------------
    # Config hot-reload helpers
    # ---------------------------------------------------------------------

    def _get_runtime_config_path(self) -> str:
        memory_folder_path = os.path.join(
            os.path.dirname(__file__), "../../../config", "memory"
        )
        os.makedirs(memory_folder_path, mode=0o755, exist_ok=True)
        return os.path.join(memory_folder_path, ".runtime.json5")

    def _create_runtime_config_file(self) -> str:
        runtime_config_path = self._get_runtime_config_path()
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../../../config",
            self.config_name + ".json5",
        )

        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    raw = json5.load(f)
                tmp_path = runtime_config_path + ".tmp"
                with open(tmp_path, "w") as wf:
                    json5.dump(raw, wf, indent=2)
                os.replace(tmp_path, runtime_config_path)
        except Exception as e:
            logging.error(f"Failed to create runtime config file: {e}")

        return runtime_config_path

    def _get_file_mtime(self) -> float:
        try:
            return os.path.getmtime(self.config_path)
        except OSError:
            return 0.0

    # ---------------------------------------------------------------------
    # Runtime loop
    # ---------------------------------------------------------------------

    async def run(self) -> None:
        try:
            if self.hot_reload:
                self.config_watcher_task = asyncio.create_task(
                    self._check_config_changes()
                )

            await self._start_orchestrators()
            self.cortex_loop_task = asyncio.create_task(self._run_cortex_loop())

            while True:
                await asyncio.sleep(1)

        finally:
            await self._cleanup_tasks()

    async def _check_config_changes(self) -> None:
        while True:
            await asyncio.sleep(self.check_interval)
            current_mtime = self._get_file_mtime()
            if current_mtime > self.last_modified:
                await self._reload_config()
                self.last_modified = current_mtime

    async def _reload_config(self) -> None:
        self._is_reloading = True
        try:
            new_config = load_config(
                self.config_name, config_source_path=self.config_path
            )
            await self._stop_current_orchestrators()

            self.config = new_config
            self.fuser = Fuser(new_config)
            self.action_orchestrator = ActionOrchestrator(new_config)
            self.simulator_orchestrator = SimulatorOrchestrator(new_config)
            self.background_orchestrator = BackgroundOrchestrator(new_config)

            # reload history settings (keeps file)
            self.history = history_from_config(self.config.__dict__, self.config_name)

            await self._start_orchestrators()
            self.cortex_loop_task = asyncio.create_task(self._run_cortex_loop())
        finally:
            self._is_reloading = False

    # ---------------------------------------------------------------------
    # Orchestrators
    # ---------------------------------------------------------------------

    async def _start_orchestrators(self) -> None:
        input_orchestrator = InputOrchestrator(self.config.agent_inputs)
        self.input_listener_task = asyncio.create_task(input_orchestrator.listen())

        self.simulator_task = self.simulator_orchestrator.start()
        self.action_task = self.action_orchestrator.start()
        self.background_task = self.background_orchestrator.start()

    async def _stop_current_orchestrators(self) -> None:
        tasks = [
            t
            for t in [
                self.cortex_loop_task,
                self.input_listener_task,
                self.simulator_task,
                self.action_task,
                self.background_task,
            ]
            if t and not t.done()
        ]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _cleanup_tasks(self) -> None:
        await self._stop_current_orchestrators()
        self.config_provider.stop()

    async def _run_cortex_loop(self) -> None:
        while True:
            if not self.sleep_ticker_provider.skip_sleep:
                await self.sleep_ticker_provider.sleep(1 / self.config.hertz)
            await self._tick()
            self.sleep_ticker_provider.skip_sleep = False

    # ---------------------------------------------------------------------
    # CORE TICK (FIXED)
    # ---------------------------------------------------------------------

    async def _tick(self) -> None:
        try:
            if self._is_reloading:
                return

            tick_num = self.io_provider.increment_tick()

            finished_promises, _ = await self.action_orchestrator.flush_promises()

            prompt = self.fuser.fuse(self.config.agent_inputs, finished_promises)
            if prompt is None:
                return

            # --------------------------------------------------
            # BUILD MESSAGES (OPENAI-SAFE, FLATTENED)
            # --------------------------------------------------
            messages = []

            system_prompt = getattr(self.config.cortex_llm, "system_prompt", None)
            if system_prompt:
                messages.append(
                    {"role": "system", "content": str(system_prompt)}
                )

            # 🔧 Flatten persisted history
            for msg in self.history.messages():
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                messages.append(
                    {
                        "role": msg.get("role", "assistant"),
                        "content": content,
                    }
                )

            # current user turn
            messages.append(
                {
                    "role": "user",
                    "content": str(prompt),
                }
            )

            # --------------------------------------------------
            # CALL LLM
            # --------------------------------------------------
            output = await self.config.cortex_llm.ask(messages)
            if output is None:
                return

            assistant_text = (
                output.raw if hasattr(output, "raw") else str(output)
            )

            # Persist assistant ONLY
            self.history.add(
                "assistant",
                assistant_text,
                {"tick": tick_num},
            )

            await self.simulator_orchestrator.promise(output.actions)
            await self.action_orchestrator.promise(output.actions)

        except Exception as e:
            logging.error(f"Error in cortex tick: {e}")
