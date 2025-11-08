"""
===================================================================
 ECHOSCOPE - Main AI Brain Software (v3.29 - Sensitivity Fix)
===================================================================
This script is the complete "brain" for your project.

BUG FIXES (v3.29):
- SENSITIVITY: Added 'sensitivities=[0.7]' to the
  pvporcupine.create() call (line 590) to make the
  "Vision" wake word much more sensitive. (0.5 is default).
- MIC ID: Set 'WORKING_MIC_ID = 18' (line 74).
- (Retains v3.22 fixes for Stereo-to-Mono conversion
  to prevent [Errno -9998] crash)
- (Retains v3.20 fixes for audio locking)
- (Retains v3.12 fixes for audio priority)
"""

# --- 1. IMPORT ALL LIBRARIES ---
import serial               # For Arduino
import threading            # To run tasks in the background (Speech, Wake Word)
import time
import os
import math
import struct
import webbrowser
import numpy as np          # For audio resampling

# --- AI & SENSORS ---
import cv2                  # For webcam
from ultralytics import YOLO # For object detection
import easyocr              # For text reading
import pvporcupine          # For "Vision" wake word
import pyaudio              # For microphone stream
import speech_recognition as sr # For voice commands

# --- AUDIO ---
import pyttsx3              # For text-to-speech

# --- GPS & MAPPING ---
from geopy.geocoders import Nominatim     # For "Where am I?" (reverse geocoding)
from geopy.distance import geodesic       # For navigation distance
from geographiclib.geodesic import Geodesic # For navigation bearing
import folium               # For "Show me the map"


# --- 2. CRITICAL SETUP & CONFIGURATION ---

# !!! This should be your Access Key from Picovoice !!!
PICOVOICE_ACCESS_KEY = "a03iV2E6f3yOxIylNa6mOZk/v2JJ64UQvl/lLJfDut4ZV0gpDZQvNA==" 

# --- This is your trained "Vision" wake word file ---
WAKE_WORD_FILE_PATH = "vision_en_windows_v3_0_0.ppn" 
# !!! ---------------------------------- !!!

# --- Arduino Configuration ---
ARDUINO_PORT = "COM5" # This is the port you confirmed
ARDUINO_BAUD = 115200

# --- Webcam Configuration ---
WEBCAM_INDEX = 1 # 0 is built-in, 1 is your new USB cam

# --- Audio Hardware Configuration (From our tests) ---
WORKING_MIC_ID = 18# <-- FIX v3.29: Set to your wired mic ID
MIC_CHANNELS = 1 # <-- FIX v3.22: We will accept 2-channel audio
MIC_SAMPLE_RATE = 16000 # The NATIVE rate of your mic
WAKE_WORD_SAMPLE_RATE = 16000 # The REQUIRED rate for Porcupine


# --- Navigation "Smart Compass" ---
# You can get these coordinates from Google Maps
SAVED_LOCATIONS = {
    "home": (25.4358, 81.8463),      # Example: Prayagraj, India
    "venue": (28.6139, 77.2090),     # Example: New Delhi, India
    "dropped pin": None             # We can set this later
}

# --- 3. GLOBAL VARIABLES (SYSTEM STATE) ---

# --- Hardware & AI Objects ---
arduino = None
porcupine = None
audio_stream = None 
pa = None
# speech_engine is no longer global, it's created in the 'say' thread
model = None
ocr_reader = None
cap = None
current_webcam_frame = None # Holds the latest frame for OCR

# --- Camera Geometry ---
# We will set these when the camera opens
screen_width = 640
screen_height = 480
x_zone_width = screen_width // 3  # For "left", "center", "right"
y_zone_width = screen_height // 3 # For "up", "middle", "down"

# --- State Flags (for prioritization) ---
audio_lock = threading.Lock() # Thread-safe way to manage audio
is_handling_command = False # Is the system busy with a voice command?
navigating_to = None        # e.g., "home" or "venue"
last_alert_time = 0         # To prevent alert spam
last_nav_time = 0           # To prevent navigation spam

# --- Sensor Data Storage ---
sensor_data = {
    'distances': {'L': 0, 'C': 0, 'R': 0},
    'gps': {'lat': 0.0, 'lng': 0.0}
}

# --- 4. CORE AUDIO & SPEECH FUNCTIONS ---

def say(message, priority='low'):
    """
    Speaks a message using pyttsx3 in a non-blocking thread.
    - priority='high': Will interrupt any current speech. (Obstacles, Command Responses)
    - priority='low': Will only play if the lock is free. (Navigation Updates)
    """
    
    def _speak():
        global audio_lock
        
        if priority == 'high':
            # High priority can interrupt.
            if audio_lock.locked():
                try:
                    audio_lock.release() 
                except RuntimeError:
                    pass 
            audio_lock.acquire() 

        elif priority == 'low':
            # Low priority cannot interrupt.
            if not audio_lock.acquire(blocking=False):
                return 

        # --- We now have the lock ---
        print(f"AUDIO (Priority: {priority}): {message}")

        try:
            # BUG FIX v3.11: Initialize engine *inside* the thread
            # This is the only robust, thread-safe way.
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)
            
            if priority == 'high':
                engine.stop() # Stop any previous queued speech
                
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech Error: {e}")
        finally:
            try:
                audio_lock.release() # Release the lock when done
            except RuntimeError:
                pass # Already released

    # Run speech in a separate thread so it doesn't block the main loop
    threading.Thread(target=_speak, daemon=True).start()

def handle_voice_command():
    """
    The main voice command handler.
    Triggered *after* "Vision" is heard.
    Listens for the full command and acts.
    """
    global is_handling_command, navigating_to, SAVED_LOCATIONS
    
    if is_handling_command:
        return # Already handling a command

    is_handling_command = True
    print("Wake word heard! Listening for command...")

    say("Yes?", priority='high') # This initial "Yes?" is low priority
    
    r = sr.Recognizer()
    
    # --- FIX: Force 'SpeechRecognition' to use your working mic ID 18 ---
    try:
        with sr.Microphone(device_index=WORKING_MIC_ID) as source: # Use ID 18
            print("Calibrating for ambient noise...")
            r.pause_threshold = 1.0 
            r.adjust_for_ambient_noise(source, duration=1)
            # Manually set a more sensitive threshold
            r.energy_threshold = 1000 
            print(f"Energy threshold set to {r.energy_threshold}")

            print(">>> SPEAK NOW! <<<")
            
            try:
                audio = r.listen(source, timeout=7, phrase_time_limit=6)
                command = r.recognize_google(audio).lower()
                print(f"User command: {command}")

                # --- Command 1: "Where am I?" ---
                if "where am i" in command:
                    say("Finding your location...", priority='high')
                    lat = sensor_data['gps']['lat']
                    lng = sensor_data['gps']['lng']
                    address = get_location_address(lat, lng)
                    say(address, priority='high')
                    # Open the map as requested
                    show_map_on_laptop(lat, lng)

                # --- Command 2: "Read This" ---
                elif "read this" in command or "read text" in command:
                    say("Scanning for text...", priority='low') # Low priority is fine here
                    if current_webcam_frame is not None:
                        # Use the latest frame from the main loop
                        text_to_speak = read_text_from_frame(current_webcam_frame)
                        say(text_to_speak, priority='high')
                    else:
                        say("Sorry, I can't see the camera feed.", priority='high')

                # --- Command 3: "Navigate Me To..." ---
                elif "navigate me to" in command:
                    destination = None
                    for loc in SAVED_LOCATIONS.keys():
                        if loc in command and loc != "dropped pin":
                            destination = loc
                            break
                    
                    if destination:
                        navigating_to = destination
                        say(f"Starting navigation to {destination}.", priority='high')
                        run_navigation_update() # Give one immediate update
                    else:
                        say("Sorry, I don't know that location.", priority='high')
                
                # --- Command 4: "Drop a Pin" ---
                elif "drop a pin" in command:
                    lat = sensor_data['gps']['lat']
                    lng = sensor_data['gps']['lng']
                    if lat != 0.0:
                        SAVED_LOCATIONS["dropped pin"] = (lat, lng)
                        say("Got it. Pin dropped.", priority='high')
                    else:
                        say("Sorry, I don't have a GPS lock to drop a pin.", priority='high')

                # --- Command 5: "Guide me to my pin" ---
                elif "guide me to my pin" in command or "navigate to my pin" in command:
                    if SAVED_LOCATIONS["dropped pin"]:
                        navigating_to = "dropped pin"
                        say("Starting navigation to your dropped pin.", priority='high')
                        run_navigation_update()
                    else:
                        say("You haven't dropped a pin yet.", priority='high')

                # --- Command 6: "Stop Navigation" ---
                elif "stop navigation" in command:
                    if navigating_to:
                        navigating_to = None
                        say("Navigation stopped.", priority='high')
                    else:
                        say("I wasn't navigating.", priority='high')
                
                # --- No Command Found ---
                else:
                    say("Sorry, I didn't understand that command.", priority='high')

            except sr.UnknownValueError:
                say("Sorry, I didn't catch that.", priority='high')
            except sr.RequestError as e:
                say("Could not connect to speech service.", priority='high')
            except Exception as e:
                print(f"Command handling error: {e}")
                say("Sorry, an error occurred.", priority='high')

    except Exception as e:
        print(f"FATAL ERROR: Could not open microphone with ID {WORKING_MIC_ID}. Error: {e}")
        say("Sorry, I have a problem with my microphone.", priority='high')
    finally:
        # Short pause to let the user hear the response
        time.sleep(2) 
        is_handling_command = False
        print("Listening for 'Vision' again...") 

# --- 5. BACKGROUND THREADS (WAKE WORD & ARDUINO) ---

def run_wake_word_detector():
    """
    Runs in a separate thread.
    Constantly listens to the mic stream for the wake word.
    
    FIX v3.22:
    - Opens mic at its native 44100Hz and in 2-channel (Stereo) mode.
    - Averages the stereo audio to mono.
    - Resamples the mono audio to 16000Hz.
    - Feeds the compatible audio to Porcupine.
    """
    global porcupine, audio_stream, pa, audio_lock
    
    try:
        pa = pyaudio.PyAudio()
        
        # Calculate the audio buffer size needed for the NATIVE rate
        porcupine_frame_ms = (porcupine.frame_length / WAKE_WORD_SAMPLE_RATE) * 1000.0
        NATIVE_FRAME_LENGTH = int((MIC_SAMPLE_RATE / 1000.0) * porcupine_frame_ms)

        audio_stream = pa.open(
            rate=MIC_SAMPLE_RATE,       # Open at the NATIVE rate (e.g., 44100)
            channels=MIC_CHANNELS,      # Open in STEREO (2 channels)
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=NATIVE_FRAME_LENGTH,
            input_device_index=WORKING_MIC_ID # <-- FIX v3.29: Set to 18
        )
        print(f"Wake word detector started... (Listening on Device {WORKING_MIC_ID} at {MIC_SAMPLE_RATE} Hz, {MIC_CHANNELS} channels)")

        while True:
            # --- FIX v3.20: Check the lock ---
            # If the speaker is talking, pause listening.
            if audio_lock.locked():
                time.sleep(0.5) 
                continue
            
            try:
                # Read a chunk of high-sample-rate, stereo audio
                pcm_native_stereo = audio_stream.read(NATIVE_FRAME_LENGTH, exception_on_overflow=False)
                
                # Convert from bytes to int16 numpy array
                audio_data_native_stereo = np.frombuffer(pcm_native_stereo, dtype=np.int16)
                
                # --- STEREO-TO-MONO FIX (v3.22) ---
                # Reshape to (n_samples, 2) and average the two channels
                # This fixes [Errno -9998] Invalid number of channels
                audio_data_native_mono = audio_data_native_stereo.reshape(-1, MIC_CHANNELS).mean(axis=1).astype(np.int16)
                # --- End Mono Fix ---

                
                # --- RESAMPLING FIX (v3.19) ---
                # Calculate the number of samples for the output (16kHz)
                # This fixes [Errno -9999] Unanticipated host error
                num_samples_16k = int(len(audio_data_native_mono) * WAKE_WORD_SAMPLE_RATE / MIC_SAMPLE_RATE)
                
                # Resample using linear interpolation
                audio_data_16k = np.interp(
                    np.linspace(0, len(audio_data_native_mono) - 1, num_samples_16k), # New 16k indices
                    np.arange(len(audio_data_native_mono)),                          # Old 44.1k indices
                    audio_data_native_mono                                           
                ).astype(np.int16)
                # --- End Resampling ---

            except Exception as e:
                print(f"Audio processing error in wake word thread: {e}")
                time.sleep(0.2)
                continue
            
            # Feed the *resampled, mono, 16kHz* audio to Porcupine
            keyword_index = porcupine.process(audio_data_16k)
            
            # -1 means no wake word. 0 means "Vision" was detected.
            if keyword_index == 0:
                # FIX: Launch command in a new thread
                if not is_handling_command:
                    threading.Thread(target=handle_voice_command, daemon=True).start()
                
    except KeyboardInterrupt:
        print("Stopping wake word detector.")
    except Exception as e:
        print(f"!!! FATAL ERROR in Wake Word thread: {e} !!!")
        if "Invalid sample rate" in str(e) or "Unanticipated host error" in str(e) or "Invalid number of channels" in str(e):
             print(f">>> HINT: Your microphone (ID {WORKING_MIC_ID}) is hardware-incompatible.")
             print(">>> Please try a different microphone.")
        elif "Invalid Input Device" in str(e):
            print(f">>> HINT: Microphone ID {WORKING_MIC_ID} was not found.")
            print(">>> Please re-run 'check_mics.py' and update 'WORKING_MIC_ID'.")
    finally:
        if porcupine: porcupine.delete()
        if audio_stream: audio_stream.close()
        if pa: pa.terminate()

def read_and_parse_arduino():
    """
    Runs in the main thread.
    Reads sensor data from Arduino and updates the global 'sensor_data'.
    """
    global sensor_data
    if arduino and arduino.in_waiting > 0:
        line = ""
        try:
            line = arduino.readline().decode('utf-8').rstrip()
            # Expected format: "D:L,C,R|G:LAT,LNG"
            
            parts = line.split('|')
            dist_part = parts[0].split(':')[1]
            gps_part = parts[1].split(':')[1]
            
            d = dist_part.split(',')
            sensor_data['distances'] = {'L': int(d[0]), 'C': int(d[1]), 'R': int(d[2])}
            
            g = gps_part.split(',')
            sensor_data['gps'] = {'lat': float(g[0]), 'lng': float(g[1])}
            
        except Exception as e:
            # print(f"Arduino Parse Error: {e}, Line: {line}")
            pass 

# --- 6. AI & GPS "SKILL" FUNCTIONS ---

def get_location_address(lat, lng):
    """Converts (lat, lng) into a human-readable street address."""
    if lat == 0.0 or lng == 0.0:
        return "Sorry, GPS signal not found."
    try:
        geolocator = Nominatim(user_agent="vision_hackathon_app") 
        coordinates = f"{lat}, {lng}"
        location = geolocator.reverse(coordinates, language="en")
        
        if location and "address" in location.raw:
            address = location.raw["address"]
            road = address.get("road", "")
            suburb = address.get("suburb", "")
            city = address.get("city", "")
            
            if road:
                return f"You are on or near {road}, in {city or suburb}."
            else:
                return f"You are in {suburb or city}."
        else:
            return "Current location address not found."
    except Exception as e:
        print(f"Geopy Error: {e}")
        return "Could not get address. Are you connected to the internet?"

def show_map_on_laptop(lat, lng):
    """Creates an interactive map.html file and opens it in the browser."""
    if lat == 0.0 or lng == 0.0:
        return 

    map_filename = "vision_map.html" 
    m = folium.Map(location=[lat, lng], zoom_start=18)
    folium.Marker([lat, lng], popup="Current Location").add_to(m)
    m.save(map_filename)
    
    webbrowser.open("file://" + os.path.realpath(map_filename))

def read_text_from_frame(frame):
    """Uses EasyOCR to find and read text from a single image frame."""
    try:
        results = ocr_reader.readtext(frame, detail=0, paragraph=True)
        
        if results:
            found_text = " ".join(results)
            print(f"OCR Found: {found_text}")
            return f"I see the text: {found_text}"
        else:
            return "I did not find any text."
    except Exception as e:
        print(f"OCR Error: {e}")
        return "Sorry, I had an error trying to read."

# --- 7. NAVIGATION & SENSOR FUSION (THE "BRAIN") ---

def run_navigation_update():
    """
    LOW-PRIORITY TASK.
    Runs in the main loop. Checks if we are navigating and
    provides a "Smart Compass" update.
    """
    global last_nav_time, navigating_to
    
    if navigating_to is None:
        return
        
    current_time = time.time()
    
    if is_handling_command:
        return
        
    if (current_time - last_nav_time < 10):
        return
        
    last_nav_time = current_time
    
    target_coords = SAVED_LOCATIONS[navigating_to]
    current_coords = (sensor_data['gps']['lat'], sensor_data['gps']['lng'])

    if current_coords[0] == 0.0:
        say("Trying to navigate, but I have no GPS signal.", priority='low')
        return

    # 1. Calculate Distance
    distance_m = geodesic(current_coords, target_coords).meters
    
    # 2. Check for Arrival
    if distance_m < 20: 
        say(f"You have arrived at {navigating_to}. Navigation stopped.", priority='high')
        navigating_to = None
        return

    # 3. Calculate Bearing (Compass Direction)
    geo_data = Geodesic.WGS84.Inverse(current_coords[0], current_coords[1], target_coords[0], target_coords[1])
    bearing = geo_data['azi1'] 
    
    direction = convert_bearing_to_direction(bearing)
    
    # 4. Format the final string
    if distance_m > 1000:
        distance_str = f"{distance_m / 1000:.1f} kilometers"
    else:
        distance_str = f"{distance_m:.0f} meters"

    # This is the ONLY low priority message.
    say(f"{navigating_to} is {distance_str} away, to your {direction}.", priority='low')


def convert_bearing_to_direction(bearing):
    """Converts a 0-360 degree bearing to a simple compass direction."""
    bearing = (bearing + 360) % 360
    
    if 337.5 <= bearing or bearing < 22.5:
        return "North"
    elif 22.5 <= bearing < 67.5:
        return "North-East"
    elif 67.5 <= bearing < 112.5:
        return "East"
    elif 112.5 <= bearing < 157.5:
        return "South-East"
    elif 157.5 <= bearing < 202.5:
        return "South"
    elif 202.5 <= bearing < 247.5:
        return "South-West"
    elif 247.5 <= bearing < 292.5:
        return "West"
    elif 292.5 <= bearing < 337.5:
        return "North-West"
        
def process_obstacle_alerts(detections):
    """
    HIGH-PRIORITY REFLEX.
    Fuses sonar distance with YOLO object name and 3x3 grid.
    """
    global last_alert_time
    
    # --- THIS IS THE FIX ---
    # If a command is being handled, DO NOT speak obstacle alerts.
    if is_handling_command:
        return
        
    current_time = time.time()
    if (current_time - last_alert_time < 3):
        return

    # --- Sensor Check ---
    # We check if sensors are even connected.
    if sensor_data is None or 'distances' not in sensor_data:
        # This can happen if the script starts faster than the Arduino
        return
        
    dist_L = sensor_data['distances']['L']
    dist_C = sensor_data['distances']['C']
    dist_R = sensor_data['distances']['R']
    
    OBSTACLE_THRESHOLD_CM = 200 
    
    alert_message = ""
    obstacle_found = False
    
    if 0 < dist_C < OBSTACLE_THRESHOLD_CM:
        obstacle_found = True
        x_zone, y_zone, obj_name = get_object_in_grid(detections, "center")
        
        if obj_name:
            alert_message = f"Obstacle, {y_zone}-{x_zone}, {obj_name}, {dist_C} centimeters."
        else:
            alert_message = f"Obstacle, {y_zone}-{x_zone}, {dist_C} centimeters."

    elif 0 < dist_L < OBSTACLE_THRESHOLD_CM:
        obstacle_found = True
        x_zone, y_zone, obj_name = get_object_in_grid(detections, "left")
        if obj_name:
            alert_message = f"Obstacle, {y_zone}-{x_zone}, {obj_name}, {dist_L} centimeters."
        else:
            alert_message = f"Obstacle, {y_zone}-{x_zone}, {dist_L} centimeters."

    elif 0 < dist_R < OBSTACLE_THRESHOLD_CM:
        obstacle_found = True
        x_zone, y_zone, obj_name = get_object_in_grid(detections, "right")
        if obj_name:
            alert_message = f"Obstacle, {y_zone}-{x_zone}, {obj_name}, {dist_R} centimeters."
        else:
            alert_message = f"ObstSacle, {y_zone}-{x_zone}, {dist_R} centimeters."
    
    if obstacle_found:
        say(alert_message, priority='low')
        last_alert_time = current_time

def get_object_in_grid(detections, sonar_zone):
    """
    Finds the main object in a specific 3x3 grid zone.
    Returns: (x_label, y_label, object_name)
    e.g., ("right", "up", "person")
    
    Returns obj_name=None if no relevant object is found.
    """
    
    best_obj_name = None
    best_obj_conf = 0.40 # Lowered confidence
    best_obj_x_label = sonar_zone
    best_obj_y_label = "ahead" 
    
    if sonar_zone == "left":
        x_min, x_max = 0, x_zone_width
    elif sonar_zone == "center":
        x_min, x_max = x_zone_width, x_zone_width * 2
    else: # "right"
        x_min, x_max = x_zone_width * 2, screen_width

    y_up_max = y_zone_width
    y_middle_max = y_zone_width * 2
    
    for r in detections:
        for box in r.boxes:
            obj_name = model.names[int(box.cls[0])]
            obj_conf = float(box.conf[0])
            
            if obj_conf > best_obj_conf: 
                x1, y1, x2, y2 = box.xyxy[0]
                obj_center_x = (x1 + x2) / 2
                obj_center_y = (y1 + y2) / 2
                
                if x_min < obj_center_x < x_max:
                    best_obj_conf = obj_conf
                    best_obj_name = obj_name
                    
                    if obj_center_y < y_up_max:
                        best_obj_y_label = "up"
                    elif obj_center_y < y_middle_max:
                        best_obj_y_label = "ahead"
                    else:
                        best_obj_y_label = "down"

    if best_obj_name is None:
        return (sonar_zone, "ahead", None) 
    else:
        return (best_obj_x_label, best_obj_y_label, best_obj_name)


# --- 8. MAIN APPLICATION ---

def main():
    global arduino, porcupine, model, ocr_reader, cap
    global screen_width, screen_height, x_zone_width, y_zone_width
    global current_webcam_frame, pa # <-- FIX v3.14: 'pa' needs to be global
    
    print("Starting Vision AI Brain...") 

    # --- 1. Initialize Hardware & AI ---
    try:
        # Connect to Arduino
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        arduino.flush()
        print(f"Connected to Arduino on {ARDUINO_PORT}.")
    except Exception as e:
        print(f"!!! FATAL ERROR: Cannot connect to Arduino on {ARDUINO_PORT} !!!")
        print(f"Error: {e}")
        print("Is it plugged in? Is the port correct? Is Serial Monitor closed?")
        return

    try:
        # Load YOLO Model
        print("Loading YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        print("YOLO model loaded.")
        
        # Load OCR Model
        print("Loading OCR model... (This may take a moment)")
        ocr_reader = easyocr.Reader(['en'], gpu=False) 
        print("OCR model loaded.")
        
        # Open Webcam
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            raise Exception(f"Cannot open webcam index {WEBCAM_INDEX}")
        
        # Get actual camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x_zone_width = screen_width // 3
        y_zone_width = screen_height // 3
        print(f"Webcam opened ({screen_width}x{screen_height}). Zones are {x_zone_width}x{y_zone_width}px.")

        # Load Porcupine Wake Word Engine
        porcupine = pvporcupine.create(
            access_key=PICOVOICE_ACCESS_KEY,
            keyword_paths=[WAKE_WORD_FILE_PATH]
        )
        print("Porcupine wake word engine loaded for 'Vision'.") 

    except Exception as e:
        print(f"!!! FATAL ERROR during initialization: {e} !!!")
        if "PICOVOICE_ACCESS_KEY" in str(e):
             print(">>> HINT: Did you forget to set your 'PICOVOICE_ACCESS_KEY' at the top of the script?")
        if "keyword_paths" in str(e):
             print(f">>> HINT: Did you download your '{WAKE_WORD_FILE_PATH}' file and put it in this folder?")
        return

    # --- 2. Start Background Threads ---
    wake_word_thread = threading.Thread(target=run_wake_word_detector, daemon=True)
    wake_word_thread.start()

    say("Vision online. Listening for 'Vision'.", priority='low') 

    # --- 3. Main AI Loop ---
    try:
        while True:
            # --- INPUTS ---
            # 1. Read Arduino
            read_and_parse_arduino()
            
            # 2. Read Webcam
            success, frame = cap.read()
            if not success:
                print("Webcam feed lost!")
                break
            
            # Save a copy for the OCR command to use
            current_webcam_frame = frame.copy()
            
            # 3. Run YOLO AI
            # BUG FIX v3.5: Expanded list of detectable classes
            # 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
            # 11=stop sign, 15=bench, 16=cat, 17=dog, 24=backpack, 26=handbag, 28=suitcase
            # 56=chair, 58=potted plant, 62=tv
            detections = list(model(frame, stream=True, verbose=False, classes=[0,1,2,3,5,7,11,15,16,17,24,26,28,56,58,62])) 
            
            # --- PROCESSING (THE "BRAIN") ---
            
            # Priority 1: High-Priority Obstacle Reflex
            process_obstacle_alerts(detections)
            
            # Priority 2: Low-Priority Navigation Updates
            run_navigation_update()

            # --- OUTPUTS ---
            # 4. Display annotated video (for your hackathon demo)
            try:
                # 'plot()' draws all the boxes and labels on the frame
                annotated_frame = detections[0].plot() 
                
                # Draw the 3x3 grid lines
                cv2.line(annotated_frame, (x_zone_width, 0), (x_zone_width, screen_height), (0, 255, 255), 1)
                cv2.line(annotated_frame, (x_zone_width * 2, 0), (x_zone_width * 2, screen_height), (0, 255, 255), 1)
                cv2.line(annotated_frame, (0, y_zone_width), (screen_width, y_zone_width), (0, 255, 255), 1)
                cv2.line(annotated_frame, (0, y_zone_width * 2), (screen_width, y_zone_width * 2), (0, 255, 255), 1)
                
                cv2.imshow("Vision AI - Press 'q' to quit", annotated_frame)
            except IndexError:
                # This happens if no objects are detected in 'detections'
                # Just show the normal, un-annotated frame
                cv2.imshow("Vision AI - Press 'q' to quit", frame) 
            except Exception as e:
                # Catch other potential display errors
                print(f"Display Error: {e}")
                cv2.imshow("Vision AI - Press 'q' to quit", frame) 

            # 5. Check for 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # --- 4. Cleanup ---
        print("Cleaning up... Vision offline.")
        if arduino: arduino.close()
        if cap: cap.release()
        cv2.destroyAllWindows()
        # The wake word thread will clean itself up.

# --- 9. RUN THE PROGRAM ---
if __name__ == "__main__":
    main()