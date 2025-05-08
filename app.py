import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model and label encoder
def load_model_and_encoder(model_path, encoder_path):
    model = tf.keras.models.load_model(model_path)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

# Function to extract landmarks from hand landmarks
def extract_landmark_array(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append(landmark.x)
        landmarks.append(landmark.y)
        landmarks.append(landmark.z)
    return np.array(landmarks).flatten()  # Flatten the list to 1D array

# Function to predict sign from landmarks
def predict_sign(landmarks, model, label_encoder):
    # Reshape landmarks to match the model's input shape
    landmarks = np.expand_dims(landmarks, axis=0)
    prediction_probabilities = model.predict(landmarks)
    
    predicted_index = np.argmax(prediction_probabilities, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = np.max(prediction_probabilities)
    
    return predicted_label, confidence

# Load model and label encoder
model_path = "Model/best_model.keras"
encoder_path = "Model/label_encoder.pkl"
model, label_encoder = load_model_and_encoder(model_path, encoder_path)

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Streamlit page setup
st.set_page_config(page_title="Real-Time Sign Language Recognition", layout="centered")
st.title("🤟 Real-Time Sign Language Recognition")
st.write("This app detects hand gestures using your webcam and predicts ASL signs.")

# Confidence threshold slider
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.01)

# Streamlit video display placeholder
frame_placeholder = st.empty()
prediction_text = st.empty()

# Webcam control checkbox
run = st.checkbox('Start Webcam')

cap = None
if run:
    cap = cv2.VideoCapture(0)

while run and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Webcam not found or disconnected.")
        break

    # Flip and process the frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks and make prediction
            landmarks_array = extract_landmark_array(hand_landmarks)
            prediction, confidence = predict_sign(landmarks_array, model, label_encoder)

            if confidence >= confidence_threshold:
                text = f"{prediction} ({confidence:.2%})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                prediction_text.markdown(f"### ✋ Detected: **{prediction}**\nConfidence: **{confidence:.2%}**")
            else:
                cv2.putText(frame, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                prediction_text.markdown("### 🤔 Detecting...")

    # Convert frame back to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

if cap:
    cap.release()
