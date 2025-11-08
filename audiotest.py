import pyttsx3

print("Initializing text-to-speech engine...")
try:
    # Try to initialize the engine
    engine = pyttsx3.init()
except Exception as e:
    print(f"!!! ERROR: Failed to initialize engine: {e}")
    print("This might be a driver problem.")
    exit()

print("Engine initialized.")

# Get details of available voices
voices = engine.getProperty('voices')
if not voices:
    print("!!! ERROR: No voices found on your system. !!!")
    print("Your Windows SAPI5 drivers might be missing or broken.")
else:
    print(f"Found {len(voices)} voices.")
    # You can uncomment the lines below to see all voices
    # for i, voice in enumerate(voices):
    #     print(f"  Voice {i}: {voice.name}")

# Set properties before speaking
engine.setProperty('rate', 180)  # Speed of speech
engine.setProperty('volume', 1.0) # Volume (0.0 to 1.0)

# --- The Test ---
print("Attempting to speak 'Hello World'...")
say_this = "Hello world, this is a test."
engine.say(say_this)

try:
    # This command is CRITICAL. It tells the engine to run.
    engine.runAndWait()
    print("...Test complete.")
    
except Exception as e:
    print(f"!!! ERROR during speech: {e}")

print("Script finished.")