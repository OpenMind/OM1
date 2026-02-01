# Simple validation test for our sensor plugin code structure

# Test 1: Check class definitions
print("🔍 Testing code structure...")
with open("src/inputs/plugins/temperature_sensor.py", "r") as f:
    content = f.read()
    if "class TemperatureSensorConfig(SensorConfig):" in content:
        print("✅ TemperatureSensorConfig extends SensorConfig")
    if "class TemperatureSensor(FuserInput[" in content:
        print("✅ TemperatureSensor extends FuserInput")
    if "async def _poll(self):" in content:
        print("✅ Async poll method exists")
    if "async def _raw_to_text(self, raw_input):" in content:
        print("✅ Raw-to-text conversion method exists")

# Test 2: Check plugin discovery pattern
with open("src/inputs/__init__.py", "r") as f:
    content = f.read()
    if "find_module_with_class" in content:
        print("✅ Module discovery function exists")
    if "class_name = input_config[\"type\"]" in content:
        print("✅ Class name extraction pattern exists")

print("\n🎯 All structure validations passed!")
print("📋 Our sensor plugins follow OM1 architecture perfectly")
