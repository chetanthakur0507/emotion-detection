import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response
import threading
import time

app = Flask(__name__)

# Load pretrained model
model = load_model('fer2013_mini_XCEPTION.hdf5', compile=False)

# Emotion labels and colors (BGR format)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 255, 0),    # Green
    'Fear': (0, 255, 255),     # Yellow
    'Happy': (0, 255, 255),    # Yellow
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (255, 0, 255), # Magenta
    'Neutral': (128, 128, 128) # Gray
}

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class CameraManager:
    """Handles webcam streaming and background predictive processing for ultra-smooth video."""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.faces_data = []
        self.lock = threading.Lock()
        self.running = True
        
        # Thread for smooth webcam capture
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Thread for heavy AI Prediction logic
        self.predict_thread = threading.Thread(target=self._predict_loop)
        self.predict_thread.daemon = True
        self.predict_thread.start()

    def _capture_loop(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                # Flip frame horizontally for a more natural mirror effect
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)

    def _predict_loop(self):
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.frame is not None:
                    frame_to_process = self.frame.copy()
                    
            if frame_to_process is not None:
                gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
                
                # Downscale by 50% to make Face Detection blazingly fast
                small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
                faces = face_cascade.detectMultiScale(small_gray, 1.3, 5)
                
                new_faces_data = []
                for (x, y, w, h) in faces:
                    # Restore coordinates to original bounds
                    x, y, w, h = x * 2, y * 2, w * 2, h * 2
                    
                    face = gray[y:y+h, x:x+w]
                    if face.shape[0] >= 48 and face.shape[1] >= 48:
                        face = cv2.resize(face, (48, 48))
                        face = face / 255.0
                        face = np.reshape(face, (1, 48, 48, 1))

                        prediction = model.predict(face, verbose=0)
                        emotion = emotion_labels[np.argmax(prediction)]
                        color = emotion_colors[emotion]

                        text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                        text_width, text_height = text_size[0]
                        
                        new_faces_data.append((x, y, w, h, emotion, color, text_width, text_height))
                
                with self.lock:
                    self.faces_data = new_faces_data
                    
            # Avoid hogging the CPU completely
            time.sleep(0.02)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()
            faces_data = list(self.faces_data)
            
        # Draw bounding boxes from latest prediction data
        for (x, y, w, h, emotion, color, text_width, text_height) in faces_data:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            bg_x1 = x
            bg_y1 = y - text_height - 15
            bg_x2 = x + text_width + 20
            bg_y2 = y - 5
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(frame, emotion, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

# Global camera instance
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = CameraManager()
    return camera

def gen_frames():
    cam = get_camera()
    while True:
        frame_bytes = cam.get_frame()
        if frame_bytes is not None:
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Provide a smooth ~30 FPS HTTP Stream
        time.sleep(0.033)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Running threaded so web server doesn't block
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)