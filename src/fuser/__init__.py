import logging
import time
import typing as T
from collections.abc import Sequence
from datetime import datetime

from actions import describe_action
from fuser.knowledge_base.retriever import KnowledgeBase
from inputs.base import Sensor
from providers.io_provider import IOProvider
from runtime.config import RuntimeConfig


class Fuser:
    """
    Combines multiple agent inputs into a single formatted prompt.

    Responsible for integrating system prompts, input streams, action descriptions,
    and command prompts into a coherent format for LLM processing.

    Parameters
    ----------
    config : RuntimeConfig
        Runtime configuration settings.
    io_provider : IOProvider
        Provider for handling I/O data and timing.
    """

    def __init__(self, config: RuntimeConfig):
        """
        Initialize the Fuser with runtime configuration.

        Parameters
        ----------
        config : RuntimeConfig
            Runtime configuration object.
        """
        self.config = config
        self.io_provider = IOProvider()

        self.knowledge_base = None
        self.kb_min_score = 0.0
        if config.knowledge_base:
            try:
                kb_config = dict(config.knowledge_base)
                self.kb_min_score = kb_config.get("min_score", 0.0)
                if self.kb_min_score > 0:
                    logging.info(
                        f"KnowledgeBase min_score threshold: {self.kb_min_score}"
                    )

                self.knowledge_base = KnowledgeBase(**kb_config)
                logging.info(
                    f"KnowledgeBase enabled with config: {config.knowledge_base}"
                )
            except Exception:
                logging.exception(
                    "Failed to initialize KnowledgeBase with provided config"
                )
                self.knowledge_base = None

    async def fuse(
        self, inputs: Sequence[Sensor], finished_promises: list[T.Any]
    ) -> T.Optional[str]:
        """
        Combine all inputs into a single formatted prompt string.

        Integrates system prompts, input buffers, action descriptions, and
        command prompts into a structured format for LLM processing.

        Parameters
        ----------
        inputs : Sequence[Sensor]
            Sequence of agent input objects containing latest input buffers.
        finished_promises : list[Any]
            List of completed promises from previous actions.

        Returns
        -------
        str or None
            Fused prompt string combining all inputs and context,
            or None if no inputs are available this tick.
        """
        # Record the timestamp of the input
        self.io_provider.fuser_start_time = time.time()

        input_strings = [input.formatted_latest_buffer() for input in inputs]
        logging.debug(f"InputMessageArray: {input_strings}")

        # Combine all inputs, memories, and configurations into a single prompt
        today = datetime.now().strftime("%B %-d, %Y")
        system_prompt = (
            "\nBASIC CONTEXT:\n"
            + self.config.system_prompt_base
            + f"\n\nToday is {today}.\n"
        )

        inputs_fused = "".join([s for s in input_strings if s is not None])

        # Query the knowledge base if configured and if there are inputs to query with
        kb_context = ""
        if self.knowledge_base and inputs_fused:
            try:
                query_text = None
                voice_input = self.io_provider.get_input("Voice")
                if (
                    voice_input
                    and voice_input.input
                    and self.io_provider.tick_counter == voice_input.tick
                ):
                    query_text = voice_input.input.strip()

                if query_text:
                    results = await self.knowledge_base.query(
                        query_text, top_k=3, min_score=self.kb_min_score
                    )
                    if results:
                        kb_context = self.knowledge_base.format_context(
                            results, max_chars=1500
                        )
                        logging.info(
                            f"Knowledge base: {len(results)} docs passed to LLM"
                        )
                    else:
                        logging.info(
                            "Knowledge base: 0 docs passed threshold, skipping context"
                        )
            except Exception as e:
                logging.error(f"Error querying knowledge base: {e}")

        # Add knowledge base context to inputs if available
        if kb_context:
            inputs_fused += f"\n\nKNOWLEDGE BASE:\n{kb_context}"

        # if we provide laws from blockchain, these override the locally stored rules
        # the rules are not provided in the system prompt, but as a separate INPUT,
        # since they are flowing from the outside world
        if self.config.system_governance and "Universal Laws" not in inputs_fused:
            system_prompt += "\nLAWS:\n" + self.config.system_governance

        if self.config.system_prompt_examples:
            system_prompt += "\n\nEXAMPLES:\n" + self.config.system_prompt_examples

        actions_fused = ""

        for action in self.config.agent_actions:
            desc = describe_action(
                action.name, action.llm_label, action.exclude_from_prompt
            )
            if desc:
                actions_fused += desc + "\n\n"

        question_prompt = "What will you do? Actions:"

        # this is the final prompt:
        # (1) a (typically) fixed overall system prompt with the agents, name, rules, and examples
        # (2) all the inputs (vision, sound, etc.)
        # (3) a (typically) fixed list of available actions
        # (4) a (typically) fixed system prompt requesting commands to be generated
        fused_prompt = f"{system_prompt}\n\nAVAILABLE INPUTS:\n{inputs_fused}\nAVAILABLE ACTIONS:\n\n{actions_fused}\n\n{question_prompt}"

        logging.debug(f"FINAL PROMPT: {fused_prompt}")

        # Record the global prompt, actions and inputs
        self.io_provider.set_fuser_system_prompt(f"{system_prompt}")
        self.io_provider.set_fuser_inputs(inputs_fused)

        # Record the timestamp of the output
        self.io_provider.fuser_end_time = time.time()

        return fused_prompt
