from flask import Flask, Response, render_template, jsonify
import cv2
import threading
import time
import requests



import requests
from neweng import FaceDetectionSystem, RTSPVideoStream  

app = Flask(__name__)
POCKETBASE_URL = "http://127.0.0.1:8090/api/collections/faces/records"
# Paths
video_source = "rtsp://192.168.1.7:554/stream"
model_path = "yolov8n-face.pt"
person_path = "yolov8n.pt"
feature_path = "./datasets/face_features/feature"
arcface_model_path = "./face_recognition/arcface/weights/arcface_r100.pth"

# Initialize System
face_system = FaceDetectionSystem(model_path, feature_path, arcface_model_path, person_path)
stream = RTSPVideoStream(video_source).start()

def generate_frames():
    """Continuously capture and process frames for real-time streaming."""
    while True:
        ret, frame = stream.read()
        if not ret:
            continue

        # Process frame with face recognition
        processed_frame = face_system.process_frame(frame)

        # Convert to JPEG (Lower quality for speed)
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        # Send frame to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def fetch_all_faces():
    """Fetch all records from PocketBase by paginating through results."""
    all_faces = []
    page = 1
    per_page = 100  # Adjust this value if needed
    while True:
        response = requests.get(f"{POCKETBASE_URL}?page={page}&perPage={per_page}").json()
        if "items" in response:
            all_faces.extend(response["items"])
            if len(response["items"]) < per_page:
                break  # Stop when there are no more pages
        else:
            break
        page += 1  # Move to the next page

    return all_faces

@app.route('/get_faces')
def get_faces():
    """Return all detected faces from PocketBase."""
    faces = fetch_all_faces()
    return jsonify({"items": faces})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
