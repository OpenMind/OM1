cat > tests/conftest.py << 'EOF'
"""
Pytest configuration file for OM1 tests
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

pytest_plugins = []
EOF
