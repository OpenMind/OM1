"""
Bootstrap module for CLI sys.path configuration.

This module MUST be imported before any other cli modules to ensure
correct module resolution. The sys.path manipulation is necessary because
om1-modules (external git dependency) installs a zenoh_msgs package that
conflicts with our local src/zenoh_msgs module.

Technical Details:
    - om1-modules package installs zenoh_msgs to site-packages
    - Our local src/zenoh_msgs has custom message definitions
    - Without this fix, Python imports the wrong zenoh_msgs
    - This module ensures src/ is at the beginning of sys.path
    - It also clears any incorrectly cached zenoh_msgs imports
"""

import os
import sys

# Ensure src directory is at the beginning of sys.path
# This prioritizes local modules over site-packages
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
elif sys.path[0] != _src_dir:
    sys.path.remove(_src_dir)
    sys.path.insert(0, _src_dir)

# Remove cached zenoh_msgs if it was loaded from wrong location
if "zenoh_msgs" in sys.modules:
    _zenoh_msgs = sys.modules["zenoh_msgs"]
    if hasattr(_zenoh_msgs, "__file__") and _zenoh_msgs.__file__:
        if "site-packages" in _zenoh_msgs.__file__:
            # Remove the wrongly cached module and its submodules
            _to_remove = [k for k in sys.modules if k.startswith("zenoh_msgs")]
            for k in _to_remove:
                del sys.modules[k]
