import time
import typing as T

from pydantic import BaseModel, ConfigDict

ConfigType = T.TypeVar("ConfigType", bound="BackgroundConfig")


class BackgroundConfig(BaseModel):
    """
    Base configuration class for Background implementations.
    """

    model_config = ConfigDict(extra="allow")


class Background(T.Generic[ConfigType]):
    """
    Base class for background components.
    """

    def __init__(self, config: ConfigType):
        """
        Initialize background with configuration.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration object for the background
        """
        self.config = config
        self.name = getattr(config, "name", type(self).__name__)

    def run(self) -> None:
        """
        Run the background process.

        This method should be overridden by subclasses to implement specific behavior.
        """
        time.sleep(60)
