import json
import logging

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_frontier_exploration import (
    UnitreeGo2FrontierExplorationProvider,
)


class UnitreeGo2FrontierExplorationConfig(BackgroundConfig):
    """
    Configuration for Unitree Go2 Frontier Exploration Background.

    Parameters
    ----------
    topic : str
        Topic for exploration status.
    context_aware_text : str
        Context aware text as JSON string.
    """

    topic: str = Field(
        default="explore/status", description="Topic for exploration status"
    )
    context_aware_text: str = Field(
        default='{"exploration_done": true}',
        description="Context aware text as JSON string",
    )


class UnitreeGo2FrontierExploration(Background[UnitreeGo2FrontierExplorationConfig]):
    """
    Start Frontier Exploration from UnitreeGo2FrontierExplorationProvider.
    """

    def __init__(self, config: UnitreeGo2FrontierExplorationConfig):
        super().__init__(config)

        topic = self.config.topic
        context_aware_text = self.config.context_aware_text

        try:
            context_aware_text = json.loads(context_aware_text)
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Error decoding context_aware_text JSON: {e}")
            context_aware_text = {"exploration_done": True}

        self.unitree_go2_frontier_exploration_provider = (
            UnitreeGo2FrontierExplorationProvider(
                topic=topic,
                context_aware_text=context_aware_text,
            )
        )
        self.unitree_go2_frontier_exploration_provider.start()
        logging.info(
            "Unitree Go2 Frontier Exploration Provider initialized in background"
        )
