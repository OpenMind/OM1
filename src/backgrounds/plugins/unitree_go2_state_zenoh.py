"""Background that keeps a Zenoh-routed Go2 SportModeState provider alive.

Surface-compatible with ``UnitreeGo2State``. The underlying provider
subscribes to Zenoh key ``sportmodestate``.

``/sportmodestate`` is only published by the real Go2 firmware, so when
running against a sim this background sits idle — downstream code treats
``state_code is None`` as "no info, proceed".

    backgrounds: [
        { type: "UnitreeGo2StateZenoh" },
    ]
"""

from __future__ import annotations

import logging

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_state_zenoh_provider import UnitreeGo2StateZenohProvider


class UnitreeGo2StateZenohConfig(BackgroundConfig):
    """Configuration for ``UnitreeGo2StateZenoh``.

    Parameters
    ----------
    topic : str
        Zenoh key to subscribe to. Default ``sportmodestate``.
    """

    topic: str = Field(
        default="sportmodestate",
        description="Zenoh key for Go2 SportModeState.",
    )


class UnitreeGo2StateZenoh(Background[UnitreeGo2StateZenohConfig]):
    """Zenoh-routed Go2 SportModeState background."""

    def __init__(self, config: UnitreeGo2StateZenohConfig):
        super().__init__(config)
        self.unitree_go2_state_provider = UnitreeGo2StateZenohProvider(topic=self.config.topic)
        logging.info("Unitree Go2 State Zenoh Provider initialized in background")
