"""
Mini Module Health Checker for OM1
Checks if modules have required fields and basic functionality
Author: bebitco
"""

def check_module(module: dict) -> bool:
    """
    Checks if a module is valid
    Required fields: 'name', 'version', 'author'
    """
    required_fields = ['name', 'version', 'author']
    for field in required_fields:
        if field not in module:
            print(f"❌ Module missing field: {field}")
            return False
    print(f"✅ Module {module['name']} passed basic checks.")
    return True

def main():
    # Example modules
    modules = [
        {"name": "test_module_1", "version": "1.0", "author": "Alice"},
        {"name": "broken_module", "version": "0.1"},
    ]

    for mod in modules:
        check_module(mod)

if __name__ == "__main__":
    main()
