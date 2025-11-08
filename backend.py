import time
import cv2  # You'll need OpenCV for the video stream
from flask import Flask, render_template, Response, jsonify

# --- This is where your REAL AI code would live ---
# You would start your Arduino connection and YOLO model here,
# perhaps in a separate thread, to continuously update
# global variables with the latest data.

# For this example, we'll just use mock data.
mock_data = {
    "location": {"lat": 40.7128, "lon": -74.0060},
    "objects": ["A person is 2 meters in front of you.", "A chair is on your left.", "A table is on your right."],
    "text": "This sign says STOP."
}
import random

def get_real_object_data():
    """
    <<< YOUR REAL AI CODE GOES HERE >>>
    This is a mock function. Replace it with your actual
    YOLO + Arduino distance logic.
    """
    # Example: read from Arduino, process with YOLO
    # For now, just return a random mock object
    return random.choice(mock_data["objects"])

def get_real_location_data():
    """
    <<< YOUR REAL AI CODE GOES HERE >>>
    This is a mock function. Replace it with your function
    that reads the latest GPS data from the Arduino.
    """
    # Example: return the latest lat/lon from your serial reader
    return mock_data["location"]

def get_real_text_data():
    """
    <<< YOUR REAL AI CODE GOES HERE >>>
    This is a mock function. Replace it with your
    OpenCV/Tesseract text-reading logic.
    """
    return mock_data["text"]

def generate_video_frames():
    """
    This function streams video from your camera.
    <<< REPLACE '0' WITH YOUR CAMERA SOURCE >>>
    If your AI script already has an OpenCV frame, you can
    just 'yield' that frame instead.
    """
    try:
        camera = cv2.VideoCapture(0) # Use 0 for built-in webcam
        if not camera.isOpened():
            raise "Cannot open camera"
            
        while True:
            success, frame = camera.read()
            if not success:
                # If the camera fails, send a black image
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Feed Offline", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # --- YOUR AI OVERLAY CODE ---
            # This is where you would draw your YOLO boxes on the 'frame'
            # e.g., frame = your_yolo_function(frame)
            # --- END AI OVERLAY ---

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in the special MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control frame rate (optional)
            time.sleep(0.03) # ~30 FPS
            
    except Exception as e:
        print(f"Camera error: {e}")
        # Clean up camera on error
        if 'camera' in locals() and camera.isOpened():
            camera.release()

# --- Flask Server Setup ---

app = Flask(__name__)

# This is the main route that serves your website
@app.route('/')
def index():
    """Serve the index.html page."""
    return render_template('index.html')

# --- API Endpoints ---
# These are the "phones" your JavaScript will call

@app.route('/api/objects')
def api_objects():
    """API endpoint to get object data."""
    object_description = get_real_object_data()
    return jsonify({"text": object_description})

@app.route('/api/location')
def api_location():
    """API endpoint to get location data."""
    location = get_real_location_data()
    text_to_speak = f"Your current location is latitude {location['lat']:.4f}, longitude {location['lon']:.4f}"
    return jsonify({"text": text_to_speak})

@app.route('/api/text')
def api_text():
    """API endpoint to read text."""
    text_result = get_real_text_data()
    return jsonify({"text": text_result})

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- How to Run ---
if __name__ == '__main__':
    # Setting threaded=True handles multiple requests (e.g., video + API)
    # Setting debug=True reloads the server when you change the code
    app.run(debug=True, threaded=True)