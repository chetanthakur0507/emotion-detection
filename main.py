import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64

app = Flask(__name__)
# Enable CORS for socket communication to support mobile devices
socketio = SocketIO(app, cors_allowed_origins="*")

# Load pretrained model
model = load_model('fer2013_mini_XCEPTION.hdf5', compile=False)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# OpenCV Fast Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('process_image')
def handle_image(data):
    """
    Receives base64 image from the mobile/desktop browser,
    runs facial detection, and emits back coordinate sizes and stats.
    """
    try:
        header, encoded = data.split(",", 1)
        decoded = base64.b64decode(encoded)
        img_np = np.frombuffer(decoded, dtype=np.uint8)
        frame = cv2.imdecode(img_np, flags=cv2.IMREAD_COLOR)
        
        # No flip needed - front camera captures correctly, 
        # and CSS handles mirroring locally!

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Downscale logic isn't needed here if the client sends a small canvas (e.g. 640x480)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        response_data = {
            "faces": [],
            "stats": {
                "emotion": "Scanning...",
                "confidence": 0.0,
                "faces_count": len(faces)
            }
        }
        
        max_confidence = 0.0
        dominant_emotion = "No Face Detected" if len(faces) == 0 else "Scanning..."
        
        for (x, y, w, h) in faces:
            # Expand bounding box
            offset_x, offset_y = int(w * 0.1), int(h * 0.1)
            hx = max(0, x - offset_x)
            hy = max(0, y - offset_y)
            hw = min(gray.shape[1] - hx, w + offset_x * 2)
            hh = min(gray.shape[0] - hy, h + offset_y * 2)

            face_roi = gray[hy:hy+hh, hx:hx+hw]
            
            emotion_str = "Unknown"
            conf_val = 0.0
            
            if face_roi.shape[0] >= 48 and face_roi.shape[1] >= 48:
                face = cv2.resize(face_roi, (48, 48))
                face = face / 255.0
                face = np.reshape(face, (1, 48, 48, 1))

                prediction = model.predict(face, verbose=0)
                max_idx = np.argmax(prediction)
                emotion_str = emotion_labels[max_idx]
                conf_val = float(prediction[0][max_idx])
                
                if conf_val > max_confidence:
                    max_confidence = conf_val
                    dominant_emotion = emotion_str
                    
            response_data["faces"].append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "emotion": emotion_str
            })
            
        response_data["stats"]["emotion"] = dominant_emotion
        response_data["stats"]["confidence"] = max_confidence
        
        # Emit detection results instantly back to the sender
        emit('result', response_data)
        
    except Exception as e:
        print("Image processing error:", e)

if __name__ == '__main__':
    # Using 'adhoc' SSL context generates an on-the-fly 'https://' cert so Mobile Chrome allows webcam access.
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')