import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from collections import deque
from torchvision import transforms
import torch
from face_alignment.alignment import norm_crop
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
import logging
import warnings
logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RTSPVideoStream:
    def __init__(self, rtsp_url, buffer_size=30):
        self.rtsp_url = rtsp_url
        self.frame_buffer = deque(maxlen=buffer_size)
        self.stopped = False
        
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.last_frame_time = 0
        self.target_fps = 20
        self.frame_interval = 1.0 / self.target_fps

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading from stream. Reconnecting...")
                self._reconnect()
                continue
            
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_interval:
                self.frame_buffer.append(frame)
                self.last_frame_time = current_time

    def _reconnect(self):
        self.cap.release()
        time.sleep(1)
        self.cap = cv2.VideoCapture(self.rtsp_url)

    def read(self):
        if self.frame_buffer:
            return True, self.frame_buffer[-1]
        return False, None

    def stop(self):
        self.stopped = True
        self.cap.release()


class FaceDetectionSystem:
    def __init__(self, model_path, feature_path, arcface_model_path,person_path):
        """Initialize the face detection and recognition system."""
        # YOLO face detection model
        self.model = YOLO(model_path)
        self.model.conf = 0.3
        self.model.iou = 0.5
        self.person_model=YOLO(person_path,)
        
        # ArcFace model for recognition
        self.recognizer = iresnet_inference(
            model_name="r100", path=arcface_model_path, device=device
        )

        # Load precomputed face features
        self.images_names, self.images_embs = read_features(feature_path)

        # Track frame processing times
        self.frame_times = deque(maxlen=30)

    def calculate_fps(self):
        """Calculate current FPS."""
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                return len(self.frame_times) / time_diff
        return 0

    @torch.no_grad()
    def get_feature(self, face_image):
        """Extract features from a face image."""
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_tensor = preprocess(face_image).unsqueeze(0).to(device)
        emb_face = self.recognizer(face_tensor).cpu().numpy()
        return emb_face / np.linalg.norm(emb_face)

    def recognize_face(self, face_image):
        """Recognize the given face."""
        query_emb = self.get_feature(face_image)
        score, id_min = compare_encodings(query_emb, self.images_embs)
        name = self.images_names[id_min] if score[0] >= 0.25 else "Unknown"
        return name, score[0]
    #
    def process_frame(self, frame):
        """Detect persons first, then detect faces and recognize them."""
        self.frame_times.append(time.time())
        fps = self.calculate_fps()

        # Detect persons first
        person_results = self.person_model(frame, classes=[0])

        if person_results and len(person_results[0].boxes) > 0:
            for res in person_results:
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    personconf = float(box.conf[0])

                    # Draw person bounding box on frame (Blue)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Extract person region (cropped person)
                    cropped_person = frame[y1:y2, x1:x2]

                    # Only process face detection if confidence is high
                    if personconf >= 0.7:
                        results = self.model(cropped_person)

                        for result in results:
                            for box in result.boxes:
                                # Extract face bounding box (relative to cropped_person)
                                x1_face, y1_face, x2_face, y2_face = map(int, box.xyxy[0].cpu().numpy())
                                confidence = float(box.conf[0])

                                if confidence < 0.3:
                                    continue

                                # Ensure face bounding box is within cropped_person dimensions
                                h, w, _ = cropped_person.shape
                                x1_face = max(0, min(x1_face, w - 1))
                                x2_face = max(0, min(x2_face, w - 1))
                                y1_face = max(0, min(y1_face, h - 1))
                                y2_face = max(0, min(y2_face, h - 1))

                                # Crop face image for recognition
                                face_image = cropped_person[y1_face:y2_face, x1_face:x2_face]
                                if face_image.size == 0:
                                    continue

                                # Recognize face
                                name, score = self.recognize_face(face_image)

                                # Convert face bounding box to full-frame coordinates
                                face_x1, face_y1, face_x2, face_y2 = (
                                    x1 + x1_face, y1 + y1_face, x1 + x2_face, y1 + y2_face
                                )

                                # Ensure face box is within full frame
                                H, W, _ = frame.shape
                                face_x1 = max(0, min(face_x1, W - 1))
                                face_x2 = max(0, min(face_x2, W - 1))
                                face_y1 = max(0, min(face_y1, H - 1))
                                face_y2 = max(0, min(face_y2, H - 1))

                                # Draw face bounding box & label on full frame (Green)
                                label = f"{name} ({score:.2f})"
                                cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0), 2)
                                cv2.putText(frame, label, (face_x1, face_y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame



def main():
    # Paths
    video_source = "rtsp://192.168.1.7:554/stream"
    model_path = "yolov8n-face.pt"
    person_path="yolov8n.pt"
    feature_path = "./datasets/face_features/feature"
    arcface_model_path = "./face_recognition/arcface/weights/arcface_r100.pth"

    # Initialize system
    face_system = FaceDetectionSystem(model_path, feature_path, arcface_model_path,person_path)
    stream = RTSPVideoStream(video_source).start()

    try:
        while True:
            ret, frame = stream.read()
            if not ret:
                # print("Failed to read frame")
                continue

            # Process frame
            processed_frame = face_system.process_frame(frame)
            cv2.imshow("Face Detection and Recognition", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
