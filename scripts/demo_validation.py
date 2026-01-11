"""
Demo validation script for OM1 modules.
Quickly checks if the main components are running.
"""

from src import om1_module

def main():
    print("Running basic OM1 module check...")
    if om1_module.is_ready():
        print("✅ OM1 module is ready!")
    else:
        print("❌ OM1 module not ready. Check installation.")

if __name__ == "__main__":
    main()
