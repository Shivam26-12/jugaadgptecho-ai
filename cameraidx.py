import cv2
import time

print("--- Camera Index Test ---")
print("I will try to open cameras 0, 1, 2, and 3.")
print("A new window will open for each camera found.")
print("Press 'q' in each window to close it and test the next one.")

# We will test the first 4 possible camera IDs
for index in range(4):
    print(f"\nTrying to open camera at index: {index}...")
    
    # Try to capture from the current index
    cap = cv2.VideoCapture(index)
    
    if cap.isOpened():
        print(f"SUCCESS: Camera found at index {index}!")
        
        # Get and print the resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  > Resolution: {width} x {height}")
        
        window_name = f"Camera Index {index} (Press 'q' to close)"
        
        while True:
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame from camera {index}.")
                break
                
            # Display the frame
            cv2.imshow(window_name, frame)
            
            # Wait for 'q' key to be pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release the camera and destroy the window
        cap.release()
        cv2.destroyWindow(window_name)
        
    else:
        print(f"No camera found at index {index}.")

print("\n--- Test Complete ---")
cv2.destroyAllWindows()
