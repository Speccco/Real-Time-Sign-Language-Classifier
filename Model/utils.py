def predict_sign(landmarks, model, label_encoder):
    """
    Run inference on landmark array and decode the prediction using label encoder.
    Also return the model's confidence for that prediction.
    """
    input_array = np.expand_dims(landmarks, axis=0)
    prediction = model.predict(input_array)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(prediction))
    return predicted_label, confidence
