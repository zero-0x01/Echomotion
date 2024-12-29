import gevent.monkey
gevent.monkey.patch_all()

import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import joblib
from flask import Flask
from flask_socketio import SocketIO, emit
import logging
import io


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'Echomotion123' 

# Initialize SocketIO with Gevent
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")  

# Load Pre-trained Model, Preprocessing Pipeline, and Label Encoder
try:
    model = load_model('models/best_model.keras')
    preprocessing_pipeline = joblib.load('models/preprocessing_pipeline.joblib')
    label_classes = np.load('models/label_encoder_classes_turev.npy', allow_pickle=True)
    logger.info("Model, preprocessing pipeline, and label encoder loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e  

# Configuration (Ensure this matches your training configuration)
config = {
    'data': {
        'sample_rate': 16000,
        'n_mfcc': 16,  # Adjust this based on training
        'expected_feature_size': 52,  # Set expected features
        'feature_extraction': {
            'include_delta': True,
            'include_delta_delta': True,
            'pitch': True,
            'energy': True,
            'zcr': True,
            'spectral_centroid': True,  # Added to make additional features=4
            'spectral_contrast': False,
            'mel_spectrogram': False,
        }
    }
}

# Feature Extraction Function
def extract_features_from_audio(audio, sr, config):
    try:
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config['data']['n_mfcc'], n_fft=2048)
        logger.info(f"Extracted MFCCs: {mfccs.shape}")  # Should be (n_mfcc, t)

        # Include delta and delta-delta features
        if config['data']['feature_extraction']['include_delta']:
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs = np.vstack((mfccs, mfccs_delta))
            logger.info(f"Included Delta MFCCs: {mfccs_delta.shape}")
        if config['data']['feature_extraction']['include_delta_delta']:
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
            mfccs = np.vstack((mfccs, mfccs_delta2))
            logger.info(f"Included Delta-Delta MFCCs: {mfccs_delta2.shape}")
        
        # Mean of MFCCs across time frames
        mfccs_mean = np.mean(mfccs, axis=1)
        logger.info(f"MFCCs mean shape: {mfccs_mean.shape}")  # Should be (n_mfcc*3,)

        # Additional Features
        fe_config = config['data']['feature_extraction']
        additional_features = []
        
        if fe_config.get('pitch', False):
            pitch, _ = librosa.piptrack(y=audio, sr=sr)
            pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
            additional_features.append(pitch_mean)
            logger.info(f"Pitch mean: {pitch_mean}")
        
        if fe_config.get('energy', False):
            energy = np.sum(librosa.feature.rms(y=audio))
            additional_features.append(energy)
            logger.info(f"Energy: {energy}")
        
        if fe_config.get('zcr', False):
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
            additional_features.append(zcr)
            logger.info(f"ZCR: {zcr}")
        
        if fe_config.get('spectral_centroid', False):
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            additional_features.append(spectral_centroid)
            logger.info(f"Spectral Centroid: {spectral_centroid}")
        
        # Combine all features
        features = np.hstack((mfccs_mean, additional_features))
        logger.info(f"Combined feature vector shape: {features.shape}")  # Should be (52,)

        # Log feature vector size before scaling
        logger.info(f"Feature vector size before scaling: {len(features)}")

        # Define expected feature size
        expected_feature_size = config['data'].get('expected_feature_size', 52)  # Default to 52

        # Ensure the feature size matches the expected size by the StandardScaler
        actual_feature_size = len(features)

        if actual_feature_size != expected_feature_size:
            logger.warning(f"Expected {expected_feature_size} features, but got {actual_feature_size} features.")
            # Adjust features to match expected size
            if actual_feature_size > expected_feature_size:
                features = features[:expected_feature_size]
                logger.warning(f"Features truncated to {expected_feature_size} features.")
            else:
                padding = np.zeros(expected_feature_size - actual_feature_size)
                features = np.hstack((features, padding))
                logger.warning(f"Features padded to {expected_feature_size} features.")
            logger.info(f"Adjusted feature vector size: {len(features)}")

        return features
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

# Function to Process Audio and Perform Emotion Recognition
def process_audio(audio_bytes):
    try:
        # Convert bytes to NumPy array
        audio_buffer = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_buffer, dtype='float32')
        
        # Log audio length
        logger.info(f"Received audio with {len(audio)} samples.")
        
        # Check if audio has at least 2048 samples
        if len(audio) < 2048:
            logger.warning(f"Received audio data is too short: {len(audio)} samples. Skipping processing.")
            return None
        
        # Resample if necessary
        if sr != config['data']['sample_rate']:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=config['data']['sample_rate'])
            sr = config['data']['sample_rate']
            logger.info(f"Resampled audio to {sr} Hz.")
        
        # Extract Features
        features = extract_features_from_audio(audio, sr, config)
        if features is not None:
            # Preprocess Features
            features = preprocessing_pipeline.transform([features])

            # Predict Emotion Probabilities
            predictions = model.predict(features)[0]  # Get the first (and only) prediction

            # Map Probabilities to Emotions
            detected_emotions = {label_classes[i]: float(predictions[i]) for i in range(len(label_classes))}
            return detected_emotions
        else:
            logger.warning("Could not extract features from audio.")
            return None
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return None

# Test Server
@app.route('/')
def index():
    return "Emotion Recognition Server is running."

# Handle WebSocket Connections
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('response', {'message': 'Connected to Emotion Recognition Server'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

# Handle Incoming Audio Data from Frontend
@socketio.on('audio_data')
def handle_audio_data(data):
    logger.info('Received audio data from client.')
    try:
        if isinstance(data, list):
            # If data is a list, convert it to bytes
            audio_bytes = bytes(data)
        elif isinstance(data, (bytes, bytearray)):
            audio_bytes = data
        else:
            logger.error("Unsupported data format received.")
            emit('emotion_data', {'emotions': {}})
            return 'error'

        # Process Audio and Get Emotions
        emotions = process_audio(audio_bytes)
        if emotions:
            # Emit Emotions Back to Client
            emit('emotion_data', {'emotions': emotions})
            logger.info("Emotions emitted back to client.")
        else:
            emit('emotion_data', {'emotions': {}})
            logger.warning("No emotions detected. Emitted empty emotions.")
        
        return 'ok'  # Acknowledge the client
    except Exception as e:
        logger.error(f"Exception in handle_audio_data: {e}")
        emit('emotion_data', {'emotions': {}})
        return 'error'


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
