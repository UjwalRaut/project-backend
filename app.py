# from flask import Flask, request, jsonify
# import librosa
# import librosa.display
# import tensorflow as tf
# import numpy as np

# app = Flask(__name__)

# # Load your speech emotion recognition model here
# # Replace this with actual model loading and initialization
# def load_model():
#     model = tf.keras.models.load_model('project_model.h5')
#     return model

# model = load_model()

# # ... (loading the model and setting up the route)

# @app.route('/predict-emotion', methods=['POST'])
# def predict_emotion():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file provided'}), 400

#     audio_file = request.files['audio']
#     audio_data, _ = tf.audio.decode_wav(audio_file.read())
#     audio_data = np.squeeze(audio_data.numpy(), axis=-1)

#     # Perform preprocessing on audio_data if required
#     # For example, you can extract MFCCs with n_mfcc=40:
#     mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=40)
#     mfccs_processed = np.mean(mfccs.T, axis=0)
#     # Reshape to match the model's expected input shape
#     mfccs_processed = mfccs_processed.reshape((-1, 40, 1))

#     # Use the loaded model for prediction
#     predicted_emotion = model.predict(mfccs_processed)
#     # Convert NumPy array to a regular Python list
#     predicted_emotion_list = predicted_emotion.tolist()

#     return jsonify({'emotion': predicted_emotion_list})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module
import librosa
import librosa.display
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for your app

# Load your speech emotion recognition model here
# Replace this with actual model loading and initialization
def load_model():
    model = tf.keras.models.load_model('speech_model.h5')
    return model

model = load_model()

# Define the emotion labels
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Surprised',
    6: 'Sad'
}

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_data, _ = tf.audio.decode_wav(audio_file.read())
    audio_data = np.squeeze(audio_data.numpy(), axis=-1)

    # Perform preprocessing on audio_data if required
    # For example, you can extract MFCCs with n_mfcc=40:
    mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    # Reshape to match the model's expected input shape
    mfccs_processed = mfccs_processed.reshape((-1, 40, 1))

    # Use the loaded model for prediction
    predicted_emotion_probs = model.predict(mfccs_processed)
    predicted_emotion_index = np.argmax(predicted_emotion_probs)
    predicted_emotion_label = emotion_labels[predicted_emotion_index]

    # Sort the probabilities in descending order and get the next probable label
    sorted_probs = np.argsort(predicted_emotion_probs[0])[::-1]
    next_probable_index = sorted_probs[1]
    next_probable_label = emotion_labels[next_probable_index]

    return jsonify({
        'predicted_emotion_label': predicted_emotion_label,
        'next_probable_emotion': next_probable_label,
        'predicted_emotion_probabilities': predicted_emotion_probs[0].tolist()
    })

@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify({'message': 'Hello, World!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

