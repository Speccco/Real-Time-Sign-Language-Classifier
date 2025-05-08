import gradio as gr
import cv2
import mediapipe as mp
from model.utils import extract_landmark_array, predict_sign, load_model_and_encoder

# Load model and label encoder
model, label_encoder = load_model_and_encoder("model/best_model.keras", "model/label_encoder.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_frame(frame, confidence_threshold):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = extract_landmark_array(hand_landmarks)
            prediction, confidence = predict_sign(landmarks, model, label_encoder)

            if confidence >= confidence_threshold:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"{prediction} ({confidence:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Gradio interface with confidence slider
gr.Interface(
    fn=process_frame,
    inputs=[
        gr.Image(source="webcam", streaming=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.01, label="Confidence Threshold")
    ],
    outputs=gr.Image()
).launch()
