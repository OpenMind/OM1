import logging
import os
import re
from typing import Any, Union


class EnvLoader:
    """
    Load ${ENV_VAR} patterns in config.
    """

    @staticmethod
    def load_env_vars(
        config: Union[dict, list, str, Any],
    ) -> Union[dict, list, str, Any]:
        """
        Load environment variables into the configuration.

        Warns about any missing variables without defaults,
        then substitutes all ${ENV_VAR} patterns.

        Parameters
        ----------
        config : Union[dict, list, str, Any]
            Configuration value to process.

        Returns
        -------
        Union[dict, list, str, Any]
            Configuration with environment variables loaded.
        """
        return EnvLoader.search_env(config)

    @staticmethod
    def search_env(
        config: Union[dict, list, str, Any],
    ) -> Union[dict, list, str, Any]:
        """
        Search the configuration for ${ENV_VAR} patterns and load them.

        Recursively traverses dicts, lists, and strings.

        Parameters
        ----------
        config : Union[dict, list, str, Any]
            Configuration value to process.

        Returns
        -------
        Union[dict, list, str, Any]
            Configuration with environment variables loaded.
        """
        if config is None:
            return config

        if isinstance(config, dict):
            return {key: EnvLoader.search_env(value) for key, value in config.items()}

        if isinstance(config, list):
            return [EnvLoader.search_env(item) for item in config]

        if isinstance(config, str):
            return EnvLoader.load_value(config)

        return config

    @staticmethod
    def load_value(value: str) -> str:
        """
        Load environment variable values in a single string.

        Parameters
        ----------
        value : str
            String containing ${ENV_VAR} or ${ENV_VAR:-default} patterns.

        Returns
        -------
        str
            String with environment variables loaded.
        """
        pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"

        def replace_match(match):
            env_var = match.group(1)
            default_value = match.group(2)

            env_value = os.environ.get(env_var)

            if env_value is not None:
                return env_value
            elif default_value is not None:
                logging.debug(
                    f"Environment variable '{env_var}' not found, using default: '{default_value}'"
                )
                return default_value
            else:
                logging.warning(
                    f"Environment variable '{env_var}' with no default value not found "
                )
                return match.group(0)

        return re.sub(pattern, replace_match, value)


load_env_vars = EnvLoader.load_env_vars
