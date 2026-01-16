import osascript
import sounddevice as sd
import soundfile as sf

filename = "woof.wav"
# Extract data and sampling rate from file
data, fs = sf.read(filename, dtype="float32")

# Get all available devices
devices = sd.query_devices()
print(devices)

# Set the default output device to the first available output
# You might need to change the index based on your system
# You can also set it using a string, e.g., sd.default.device = 'Internal Speakers'
try:
    output_device_index = next(
        i for i, device in enumerate(devices) if device["name"] == "External Headphones"
    )
    sd.default.device = output_device_index
    print(f"Default output device set to: {devices[output_device_index]['name']}")
except StopIteration:
    print("No output device found.")

# Verify the change
print(
    f"Current default output device is: {sd.query_devices(sd.default.device)['name']}"
)

print("Current volume settings")
result = osascript.osascript("get volume settings")
print(result)

target_volume = 100
print(f"Setting volume to: {target_volume}")
vol = "set volume output volume " + str(target_volume)
osascript.osascript(vol)

result = osascript.osascript("get volume settings")
print(result)

# force unmute
osascript.osascript("set volume without output muted")

print("Confirm loud and unmuted:")
result = osascript.osascript("get volume settings")
print(result)

sd.play(data, fs)
status = sd.wait()  # Wait until file is done playing
