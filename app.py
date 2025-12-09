import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_FILENAME = "Final_Model.keras"
CLASSES_LIST = [
    "Clapping", 
    "Meet and Split", 
    "Sitting", 
    "Standing Still", 
    "Walking", 
    "Walking While Reading Book", 
    "Walking While Using Phone"
]

SEQUENCE_LENGTH = 20
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CONFIDENCE_THRESHOLD = 0.5 

# ==========================================
# 1. LOAD MODEL (Cached)
# ==========================================
@st.cache_resource
def load_model():
    try:
        # Loads the 3MB file directly from the repo
        model = tf.keras.models.load_model(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"Error loading model. Make sure '{MODEL_FILENAME}' is in the GitHub repo.")
        st.error(str(e))
        return None

model = load_model()

# ==========================================
# 2. VIDEO PROCESSING CLASS
# ==========================================
class ActivityDetector(VideoTransformerBase):
    def __init__(self):
        # We need these buffers to hold the last 20 frames
        self.frames_queue = []
        self.predictions_buffer = deque(maxlen=10)
        self.current_probs = np.zeros(len(CLASSES_LIST))
        self.frame_counter = 0

    def transform(self, frame):
        # Convert the frame from the webcam (AV format) to OpenCV (numpy)
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_counter += 1
        
        # --- LOGIC: Process 1 out of every 4 frames ---
        if self.frame_counter % 4 == 0:
            # Resize to 64x64 (same as training)
            resized = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized = resized / 255.0
            self.frames_queue.append(normalized)

            # Keep queue at max 20 frames
            if len(self.frames_queue) > SEQUENCE_LENGTH:
                self.frames_queue.pop(0)

            # --- PREDICT ---
            if len(self.frames_queue) == SEQUENCE_LENGTH and model is not None:
                input_data = np.expand_dims(np.array(self.frames_queue), axis=0)
                
                # Get prediction
                raw_probs = model.predict(input_data)[0]
                
                # Smooth the prediction
                self.predictions_buffer.append(raw_probs)
                self.current_probs = np.mean(self.predictions_buffer, axis=0)

        # --- DRAW UI ON VIDEO ---
        winner_idx = np.argmax(self.current_probs)
        label = CLASSES_LIST[winner_idx]
        prob = self.current_probs[winner_idx]
        
        # Colors: Green for high confidence, Yellow for low
        if prob > CONFIDENCE_THRESHOLD:
            color = (0, 255, 0)
            text = f"{label}: {prob*100:.0f}%"
        else:
            color = (0, 255, 255)
            text = "Analyzing..."

        # 1. Draw Text
        cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 2. Draw Confidence Bar
        cv2.rectangle(img, (20, 50), (220, 65), (50, 50, 50), -1) # Grey background
        bar_width = int(prob * 200)
        cv2.rectangle(img, (20, 50), (20 + bar_width, 65), color, -1) # Color fill

        return img

# ==========================================
# 3. STREAMLIT LAYOUT
# ==========================================
st.title("Human Activity Recognition 🏃‍♂️")
st.markdown("### Live Webcam Demo")
st.write("This app uses a specific TensorFlow model to detect 7 different actions.")

if model is not None:
    # This creates the webcam widget in the browser
    webrtc_streamer(
        key="activity-detection",
        video_transformer_factory=ActivityDetector,
        media_stream_constraints={"video": True, "audio": False}
    )
else:
    st.warning("⚠️ Model not loaded. Please check your file structure.")