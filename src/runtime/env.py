import logging
import os
import re
from typing import Any, Union


class EnvLoader:
    """Load ${ENV_VAR} patterns in config."""

    @staticmethod
    def load_env_vars(config: dict) -> dict:
        """
        Load environment variables into the configuration.

        Substitutes all ${ENV_VAR} and ${ENV_VAR:-default} patterns.

        Parameters
        ----------
        config : dict
            The raw configuration dictionary to process.

        Returns
        -------
        dict
            Configuration with environment variables loaded.
        """
        return EnvLoader._process_load_value(config)  # type: ignore[return-value]

    @staticmethod
    def _process_load_value(
        config: Union[dict, list, str, Any],
    ) -> Union[dict, list, str, Any]:
        """
        Recursively process a config value, loading env vars in strings.

        Parameters
        ----------
        config : Union[dict, list, str, Any]
            A config value to process.

        Returns
        -------
        Union[dict, list, str, Any]
            The config value with environment variables loaded.
        """
        if config is None:
            return config

        if isinstance(config, dict):
            return {
                key: EnvLoader._process_load_value(value)
                for key, value in config.items()
            }

        if isinstance(config, list):
            return [EnvLoader._process_load_value(item) for item in config]

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
                logging.info(
                    f"Environment variable '{env_var}' not found, using default: '{default_value}'"
                )
                return default_value
            else:
                logging.warning(
                    f"Environment variable '{env_var}' not found and no default was provided"
                )
                return match.group(0)

        return re.sub(pattern, replace_match, value)


load_env_vars = EnvLoader.load_env_vars
