import sys
from unittest.mock import MagicMock


def pytest_configure():
    """
    Ensure heavy/remote dependencies are mocked at import time
    for all tests.
    """

    # Mock cdp SDK to prevent any real network/API usage.
    if "cdp" not in sys.modules:
        sys.modules["cdp"] = MagicMock()

    # Mock IO provider module to avoid importing zenoh or other heavy libs.
    if "providers.io_provider" not in sys.modules:
        sys.modules["providers.io_provider"] = MagicMock()

    # Some code paths may import using src.providers.io_provider
    sys.modules["src.providers.io_provider"] = sys.modules["providers.io_provider"]
