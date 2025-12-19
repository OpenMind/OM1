# BME280 Environmental Sensor Plugin

Input plugin for the Bosch BME280 environmental sensor, providing temperature, humidity, and atmospheric pressure readings.

## Features

- Real-time environmental monitoring
- I2C communication support
- Configurable sampling rate
- Automatic mock mode when hardware unavailable
- Async/await support via FuserInput

## Hardware Requirements

- BME280 sensor module (I2C interface)
- Compatible single-board computer (Raspberry Pi, etc.)
- I2C enabled on the system

## Installation

```bash
# Install required Python packages
pip install adafruit-circuitpython-bme280
```

## Configuration

Add to your OM1 configuration YAML:

```yaml
inputs:
  - type: BME280Input
    config:
      i2c_address: 0x76      # I2C address (typically 0x76 or 0x77)
      sampling_rate: 1.0     # Read interval in seconds
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `i2c_address` | int | `0x76` | I2C address of BME280 (0x76 or 0x77) |
| `sampling_rate` | float | `1.0` | Sampling interval in seconds |

## Usage Example

```python
from inputs.plugins.bme280 import BME280Input, BME280Config

# Create configuration
config = BME280Config(
    i2c_address=0x76,
    sampling_rate=2.0  # Read every 2 seconds
)

# Initialize sensor
sensor = BME280Input(config)

# Read data in async context
async def monitor():
    async for reading in sensor.listen():
        print(f"Temp: {reading['temperature']}°C")
        print(f"Humidity: {reading['humidity']}%")
        print(f"Pressure: {reading['pressure']} hPa")
```

## Output Format

### Normal Mode (Hardware Available)
```json
{
  "temperature": 25.3,
  "humidity": 45.2,
  "pressure": 1013.25,
  "timestamp": 1703001234.56
}
```

### Mock Mode (No Hardware)
```json
{
  "temperature": 25.0,
  "humidity": 50.0,
  "pressure": 1013.25,
  "timestamp": 1703001234.56,
  "mock": true
}
```

## Hardware Setup

### Wiring (Raspberry Pi)

| BME280 Pin | Raspberry Pi Pin |
|------------|------------------|
| VIN        | 3.3V (Pin 1)    |
| GND        | Ground (Pin 6)   |
| SCL        | SCL (Pin 5)      |
| SDA        | SDA (Pin 3)      |

### Enable I2C

```bash
# Enable I2C interface
sudo raspi-config
# Navigate to: Interface Options -> I2C -> Enable

# Verify I2C devices
sudo apt-get install i2c-tools
i2cdetect -y 1
# Should show device at 0x76 or 0x77
```

## Troubleshooting

### Sensor Not Detected

```bash
# Check I2C devices
i2cdetect -y 1

# If not visible, check:
# 1. Physical connections
# 2. I2C is enabled: sudo raspi-config
# 3. Correct I2C address (try 0x77 if 0x76 doesn't work)
```

### Import Errors

```bash
# Install dependencies
pip install adafruit-circuitpython-bme280

# For CircuitPython dependencies
pip install adafruit-blinka
```

### Permission Issues

```bash
# Add user to i2c group
sudo usermod -a -G i2c $USER
# Log out and back in
```

## Testing

```bash
# From project root
cd src
python3 -m pytest inputs/plugins/bme280/test_bme280.py -v
```

## Notes

- Mock mode activates automatically when:
  - Hardware libraries not installed
  - Sensor not connected
  - I2C communication fails
- Mock data provides realistic baseline values for testing
- Sensor readings are rounded to 2 decimal places

## License

MIT License - Part of the OM1 project
