📷 Real-Time Sign Language Classifier
This app uses MediaPipe to detect hand landmarks and a trained neural network (best_model.keras) to classify American Sign Language (ASL) gestures in real time.

How It Works
🖐️ Detects your hand using the webcam
📌 Extracts hand landmarks (x, y, z points)
🧠 Passes them to a trained model
🗣️ Predicts the sign label (e.g., "Hello", "Thanks")

Features
Real-time webcam-based inference

Adjustable confidence threshold

Lightweight and responsive UI using Streamlit

Model Info
Framework: TensorFlow

Input: 63-dim vector (21 landmarks × 3)

Output: One-hot encoded sign labels

Label encoding: Stored in label_encoder.pkl

Try It
👉 [Try the demo on Streamlit](https://real-time-sign-language-classifier.streamlit.app/)

👉 [Try the demo on Hugging Face](https://speccco-sign-language.hf.space/?__theme=dark)


