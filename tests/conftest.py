"""Global pytest configuration - mock heavy dependencies before import."""
import sys
from unittest.mock import MagicMock

# Mock zenoh before any provider imports it via providers/__init__.py
sys.modules['zenoh'] = MagicMock()
