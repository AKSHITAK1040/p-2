import streamlit as st
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import mediapipe as mp
import time
import threading

# Label list A–Z
labels = [chr(i) for i in range(65, 91)]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="asl_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = 64

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

st.set_page_config(page_title="ASL A–Z Detection")
st.title("🧠 ASL A–Z Fingerspelling (Webcam)")
st.caption("Show an ASL sign in front of your webcam to detect the letter live.")

# Global variable to store prediction safely
last_prediction = {"label": "", "time": 0}
prediction_lock = threading.Lock()

def run_inference(img):
    global last_prediction
    try:
        input_tensor = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        pred_index = np.argmax(output)
        pred_label = labels[pred_index]
        with prediction_lock:
            last_prediction = {"label": pred_label, "time": time.time()}
    except Exception as e:
        print("Inference error:", e)

class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_frame_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        h, w, _ = img.shape
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
                y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20

                x_min, y_min = max(x_min, 0), max(y_min, 0)
                x_max, y_max = min(x_max, w), min(y_max, h)

                hand_img = img[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    continue

                try:
                    hand_img_resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))

                    # Start inference in a separate thread
                    threading.Thread(target=run_inference, args=(hand_img_resized,), daemon=True).start()

                    # Draw box
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                except Exception as e:
                    print("Resize or crop error:", e)

        # Draw prediction text
        with prediction_lock:
            label = last_prediction["label"]
            if label:
                cv2.putText(img, f"{label}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start webcam stream
webrtc_streamer(key="asl-live", video_processor_factory=ASLProcessor)
