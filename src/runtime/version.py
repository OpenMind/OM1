import logging
from typing import Optional

latest_runtime_version = "v1.0.1"


def get_runtime_version() -> str:
    """
    Get the current runtime version.

    Returns
    -------
    str
        The current runtime version string.
    """
    return latest_runtime_version


def is_version_supported(version: Optional[str]) -> bool:
    """
    Check if the given version is supported.

    Parameters
    ----------
    version : Optional[str]
        The version string to check.

    Returns
    -------
    bool
        True if the version is supported, False otherwise.
    """
    if version is None:
        raise ValueError("Version cannot be None")

    supported_ver = latest_runtime_version.lstrip("v")
    input_ver = version.lstrip("v")

    try:
        supported_parts = [int(x) for x in supported_ver.split(".")]
        input_parts = [int(x) for x in input_ver.split(".")]

        while len(supported_parts) < 3:
            supported_parts.append(0)
        while len(input_parts) < 3:
            input_parts.append(0)

        if supported_parts[0] != input_parts[0]:
            raise ValueError(
                f"Major version mismatch: expected {supported_parts[0]}, "
                f"got {input_parts[0]}"
            )

        if supported_parts[1] != input_parts[1]:
            logging.warning(
                f"Version mismatch: expected minor version {supported_parts[1]}, "
                f"got {input_parts[1]}. This may cause compatibility issues."
            )

        return True

    except (ValueError, IndexError):
        raise ValueError("Invalid version format")


def verify_runtime_version(
    config_version: Optional[str], config_name: str = "configuration"
) -> bool:
    """
    Verify that the configuration version is compatible with the runtime version.

    Parameters
    ----------
    config_version : Optional[str]
        The version string from the configuration
    config_name : str
        Name of the configuration being loaded (for logging)

    Returns
    -------
    bool
        True if version is supported

    Raises
    ------
    ValueError
        If version is incompatible or invalid
    """
    runtime_version = get_runtime_version()

    logging.info(f"Loading {config_name} with version: {config_version}")
    logging.info(f"Runtime version: {runtime_version}")

    try:
        result = is_version_supported(config_version)
        if result:
            logging.info(f"{config_name} version is compatible with runtime")
        return result
    except ValueError as e:
        logging.error(f"Version compatibility check failed for {config_name}: {e}")
        raise ValueError(
            f"Configuration version '{config_version}' is incompatible with runtime version '{runtime_version}': {e}"
        )
    except Exception as e:
        logging.error(
            f"Unexpected error during version verification for {config_name}: {e}"
        )
        raise
