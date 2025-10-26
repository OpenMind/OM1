import random

class UltrasonicSensor:
    """Simulated Ultrasonic Distance Sensor"""
    def __init__(self):
        self.distance = 0.0

    def read(self):
        """Return a random distance between 0.1 and 3.0 meters."""
        self.distance = random.uniform(0.1, 3.0)
        return self.distance
