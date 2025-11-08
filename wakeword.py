import speech_recognition as sr

# --- THE TEST VARIABLE ---
# This is the line to change.
# Try these IDs from your 'check_mics.py' output, one by one:
# 1, 5, 9, 10, 0
TEST_MIC_ID = 15
# -------------------------

print("--- SpeechRecognition Test ---")
print(f"Testing with Device ID {TEST_MIC_ID}...")
print("This will test if Google's speech-to-text can hear you.")

r = sr.Recognizer()

# We are now forcing the script to use a specific microphone ID
try:
    with sr.Microphone(device_index=TEST_MIC_ID) as source:
        print("\nCalibrating for ambient noise... Please be quiet for 2 seconds...")
        # BUG FIX: Increased duration to 2 seconds for better noise calibration
        r.adjust_for_ambient_noise(source, duration=2)
        
        print(">>> SPEAK NOW! <<<")
        print("Say 'hello' or 'testing one two three'...")

        try:
            # BUG FIX: Increased timeout to 7 seconds to give you more time
            audio = r.listen(source, timeout=7, phrase_time_limit=7)
            
            print("\nProcessing audio...")
            
            # Use Google's free online API to recognize the speech
            text = r.recognize_google(audio).lower()
            
            print(f"========================================")
            print(f"SUCCESS! Device ID {TEST_MIC_ID} is working!")
            print(f"You said: '{text}'")
            print(f"========================================")

        except sr.UnknownValueError:
            print("----------------------------------------")
            print(f"ERROR: Could not understand audio on Device ID {TEST_MIC_ID}.")
            print("Your microphone is working, but the speech was not clear.")
            print("----------------------------------------")
        except sr.RequestError as e:
            print(f"----------------------------------------")
            print(f"ERROR: Could not connect to Google Speech service: {e}")
            print("Please check your internet connection.")
            print(f"----------------------------------------")
        except Exception as e:
            print(f"----------------------------------------")
            print(f"AN UNKNOWN ERROR OCCURRED: {e}")
            print(f"----------------------------------------")

except Exception as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"FATAL ERROR: Could not open microphone with Device ID {TEST_MIC_ID}.")
    print(f"Error details: {e}")
    print(f"This device is invalid. Please try a different ID.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")