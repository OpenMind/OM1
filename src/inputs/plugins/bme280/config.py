from pydantic import Field

from inputs.base import SensorConfig


class BME280Config(SensorConfig):
    """
    Configuration for BME280 environmental sensor.

    Parameters
    ----------
    i2c_address : int
        I2C address of the BME280 sensor (default: 0x76)
    sampling_rate : float
        Sampling rate in seconds (default: 1.0)
    """

    i2c_address: int = Field(
        default=0x76, description="I2C address of the BME280 sensor"
    )
    sampling_rate: float = Field(default=1.0, description="Sampling rate in seconds")
