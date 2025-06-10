from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import re
import os
import base64
import json
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import time
import base64
from collections import defaultdict
from cv2 import TrackerCSRT_create, TrackerKCF_create

# Initialize Firebase Admin SDK
firebase_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

if not firebase_json:
    raise Exception("Missing FIREBASE_CREDENTIALS_JSON env variable")

cred = credentials.Certificate(json.loads(firebase_json))
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Use environment variables for RTSP credentials
RTSP_USER = os.getenv('RTSP_USER', 'admin')  # Default to 'admin' if not set
DEFAULT_PASSWORDS = ['admin123456', 'hik12345']  # List of allowed passwords


# Path to your YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Define the directory where detected items will be saved
SAVE_DIR = "captured_images"

# Global variable to store the connected CCTV IP address
connected_cctv_ip = None

def is_valid_ip(ip_address):
    """Validate the IP address format."""
    ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    return ip_pattern.match(ip_address) is not None

@app.route('/connect-cctv', methods=['POST'])
def connect_to_cctv():
    data = request.json
    print('Request Data:', data)  # Add this line to log incoming request data
    ip_address = data.get('ip_address')
    rtsp_user = data.get('rtsp_user')
    rtsp_pass = data.get('rtsp_pass')

    if not ip_address or not is_valid_ip(ip_address):
        return jsonify({'error': 'Invalid or missing IP address'}), 400
    if not rtsp_user or not rtsp_pass:
        return jsonify({'error': 'Missing RTSP credentials'}), 400

    rtsp_url = f"rtsp://{rtsp_user}:{rtsp_pass}@{ip_address}:554/ch=1?subtype=0"

    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({'error': f"Unable to connect to the CCTV at {ip_address}"}), 500

        ret, _ = cap.read()
        cap.release()

        if ret:
            connected_cctv_ip = ip_address  # Store the connected IP globally
            return jsonify({'message': f"Connected to {ip_address} and stream is active"}), 200
        else:
            return jsonify({'error': "Connected, but failed to verify the stream."}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/disconnect-cctv', methods=['POST'])
def disconnect_cctv():
    """Disconnect from a CCTV camera."""
    global connected_cctv_ip
    connected_cctv_ip = None  # Clear the stored IP address
    return jsonify({'message': "Disconnected from the CCTV"}), 200

@app.route('/view-cctv', methods=['GET'])
def view_cctv():
    """Fetch a single frame from the CCTV camera using the provided IP address."""
    ip_address = request.args.get('ip_address')
    
    if not ip_address:
        return jsonify({'error': 'No IP address provided.'}), 400

    for password in DEFAULT_PASSWORDS:
        rtsp_url = f"rtsp://{RTSP_USER}:{password}@{ip_address}:554/ch=1?subtype=0"
        try:
            cap = cv2.VideoCapture(rtsp_url)
            if cap.isOpened():
                # Successfully connected, capture a frame
                ret, frame = cap.read()
                cap.release()

                if ret:
                    # Convert the frame to JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    return Response(buffer.tobytes(), mimetype='image/jpeg')
                else:
                    cap.release()
                    return jsonify({'error': "Failed to capture a frame from the CCTV stream."}), 500
        except Exception as e:
            # Log exception and continue to try the next password
            print(f"Failed with password {password}: {e}")

    # If all passwords fail
    return jsonify({'error': f"Unable to connect to the CCTV at {ip_address} with available passwords."}), 500

@app.route('/start-detection', methods=['POST'])
def start_detection():
    """Stream video for 10 seconds, perform object detection, and save detected items with bounding boxes."""
    
    MIN_CONFIDENCE_PERCENTAGE = 10  # Set the minimum confidence percentage for detection

    data = request.get_json()
    print("Received data:", data)  # Debugging log

    if not data:
        return jsonify({'error': 'Request body is empty or not valid JSON.'}), 400

    ip_address = data.get('ip_address')   # Get IP address from the request body
    user_id = data.get('user_id')         # Get user ID from the request body
    item_name = data.get('item_name')     # Get item name from the request body
    camera_name = data.get('camera_name') # Get camera name from the request body
    rtsp_user = data.get('rtsp_user')
    rtsp_pass = data.get('rtsp_pass')

    if not ip_address:
        return jsonify({'error': 'No IP address provided.'}), 400
    if not user_id:
        return jsonify({'error': 'No user ID provided.'}), 400
    if not item_name:
        return jsonify({'error': 'No item name provided.'}), 400
    if not camera_name:
        return jsonify({'error': 'No camera name provided.'}), 400  # Check if camera_name is provided

    rtsp_url = f"rtsp://{rtsp_user}:{rtsp_pass}@{ip_address}:554/ch=1?subtype=0"
    print(f"User ID: {user_id}, Item Name: {item_name}, Camera Name: {camera_name}")  # Debugging log

    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"Unable to connect to the CCTV at {ip_address}")  # Debugging log
            return jsonify({'error': f"Unable to connect to the CCTV at {ip_address}"}), 500

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else 15
        max_frames = fps * 10  # Capture for 10 seconds
        frame_count = 0
        last_saved_timestamp = None  # Temporal filtering: to store the last save timestamp
        previous_detections = defaultdict(list)  # For spatial filtering: store previous detections (bounding boxes)
        
        # Create a detection document using timestamp as the document ID
        detection_ref = db.collection('users').document(user_id).collection('Detections')

        # Generate a single timestamp for the entire detection session
        detection_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        detection_session_ref = detection_ref.document(detection_timestamp)
        detection_session_ref.set({
        'createdAt': firestore.SERVER_TIMESTAMP  # Automatically assigns server timestamp
        })

        print(f"Detection session {detection_timestamp} created.")

        # Map the item name (e.g., "phone") to the model's class ID (e.g., 0)
        item_to_class_id = {
            'Glasses': 0,
            'Phone': 1,
            'Remote': 2,
            'Watch': 3,  # Assuming the class ID for "phone" is 0
            # You can add other item names to class ID mappings if needed
        }

        # Get the class ID for the requested item
        target_class_id = item_to_class_id.get(item_name)  # Convert to lowercase for case-insensitive matching
        if target_class_id is None:
            return jsonify({'error': 'Item name is not recognized in the model.'}), 400

        # Initialize object tracker
        tracker = TrackerCSRT_create()  # You can use other trackers like TrackerKCF_create

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame at count {frame_count}")  # Debugging log
                break

            results = model(frame)
            detections = results[0].boxes.data.tolist()

            detected_objects = False  # Flag to check if any object was detected in the frame
            current_detections = []  # Store the detections for this frame

            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection
                confidence_percentage = confidence * 100

                if confidence_percentage >= MIN_CONFIDENCE_PERCENTAGE and int(class_id) == target_class_id:
                    label = model.names[int(class_id)]
                    print(f"Detected {label} with confidence {confidence_percentage:.2f}%")

                    # Store detection as a tuple: (x1, y1, x2, y2)
                    current_detections.append((int(x1), int(y1), int(x2), int(y2)))

                    # Check if object has already been detected recently (Spatial Filtering)
                    is_duplicate = False
                    for prev_detection in previous_detections[frame_count - 1]:
                        if abs(prev_detection[0] - int(x1)) < 20 and abs(prev_detection[1] - int(y1)) < 20:
                            is_duplicate = True
                            break

                    if is_duplicate:
                        continue  # Skip saving if the object is considered duplicate

                    # Temporal Filtering: Only save if enough time has passed (1 second)
                    current_timestamp = time.time()
                    if last_saved_timestamp is None or current_timestamp - last_saved_timestamp >= 1:
                        # Draw bounding box on frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {confidence_percentage:.2f}%",
                                    (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Encode frame to base64
                        _, buffer = cv2.imencode('.jpg', frame)
                        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

                        # Save frame data directly to Firestore under the detection session document
                        image_data = {
                            'frame_data': img_base64,
                            'item_name': item_name,  # Save the item name in Firestore
                            'camera_name': camera_name,  # Save camera name in Firestore
                            'timestamp': firestore.SERVER_TIMESTAMP
                        }

                        # Add image data to the detection session document
                        detection_session_ref.collection('images').add(image_data)
                        print(f"Saved frame {frame_count} to Firestore under detection timestamp {detection_timestamp}")

                        # Update the last saved timestamp
                        last_saved_timestamp = current_timestamp

                    detected_objects = True  # Set flag to True when an object is detected

            # Save the current frame detections for spatial filtering in the next loop
            previous_detections[frame_count] = current_detections

            frame_count += 1

        print("Detection completed.")
        return jsonify({'message': "Detection completed and frames saved to Firestore under detection timestamp."}), 200

    except Exception as e:
        print(f"Error during detection: {e}")  # Debugging log
        return jsonify({'error': str(e)}), 500

    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
