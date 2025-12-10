import json
import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_frontier_exploration import (
    UnitreeGo2FrontierExplorationProvider,
)


class UnitreeGo2FrontierExploration(Background):
    """
    Start Frontier Exploration from UnitreeGo2FrontierExplorationProvider.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        topic = getattr(
            self.config,
            "topic",
            "explore/status",
        )
        context_aware_text = getattr(
            self.config, "context_aware_text", json.dumps({"exploration_done": True})
        )

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
