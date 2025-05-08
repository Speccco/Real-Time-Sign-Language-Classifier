# Real-Time Sign Language Classifier

This Space uses **MediaPipe** to detect hand landmarks and a trained **Neural Network** (`best_model.keras`) to classify American Sign Language (ASL) gestures in real-time.

### How It Works

- 🖐️ Detects your hand using the webcam
- 📌 Extracts hand landmarks (x, y, z points)
- 🧠 Passes them to a trained model
- 🗣️ Predicts the sign label (e.g., "Hello", "Thanks")

### Features

- Real-time webcam-based inference
- Adjustable confidence threshold
- Lightweight and responsive UI with Gradio

### Model Info

- Framework: TensorFlow
- Input: 63-dim vector (21 landmarks × 3)
- Output: One-hot encoded sign labels
- Label encoding saved in `label_encoder.pkl`

### Try it

1. Allow webcam access.
2. Perform a hand gesture.
3. Adjust confidence threshold as needed.
[Try the demo here!](https://speccco-sign-language.hf.space/?__theme=dark)
---

Built with ❤️ using [Gradio](https://gradio.app), [MediaPipe](https://mediapipe.dev), and [Hugging Face Spaces](https://huggingface.co/spaces).
