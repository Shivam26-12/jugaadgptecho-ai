import pyaudio

print("--- Finding All Available Microphones ---")
pa = pyaudio.PyAudio()

try:
    device_count = pa.get_device_count()
    print(f"Found {device_count} audio devices:")

    for i in range(device_count):
        device_info = pa.get_device_info_by_index(i)
        # Check if it's an input device
        if device_info.get('maxInputChannels') > 0:
            print(f"\n[Input Device ID {i}] - {device_info.get('name')}")
            print(f"  Rate: {int(device_info.get('defaultSampleRate'))} Hz")
            print(f"  Channels: {device_info.get('maxInputChannels')}")

except Exception as e:
    print(f"Error: {e}")

finally:
    pa.terminate()
    print("\n--- Check Complete ---")
    print("Look for your main microphone (e.g., 'Microphone Array (Realtek Audio)')")
    print("The 'Device ID' is the number you need.")