# emotion_recognition_final.py

import os
import librosa
import numpy as np
import yaml
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, hamming_loss, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Bidirectional, LSTM,
                                     Dense, Dropout, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Ensure necessary directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=config['logging']['file'],
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger()

# Disable oneDNN optimizations (if you are encountering any related warnings)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define emotion mapping (standardize label mapping across datasets)
emotion_map = {
    'Angry': 'Angry',
    'Sad': 'Sad',
    'Calm': 'Calm',
    'Happy': 'Happy',
    'Fearful': 'Fearful',
    'Surprised': 'Surprised',
    'Disgusted': 'Disgusted',
    'kizgin': 'Angry',
    'sakin': 'Calm',
    'normal2': 'Calm',
    'mutlu': 'Happy',
    'mutlu2': 'Happy',
    'uzgun': 'Sad',
    # Add more mappings if necessary
}

# Feature Extraction Function with Enhancements
def extract_features(file_path, config, scaler=None):
    try:
        y, sr = librosa.load(file_path, sr=config['data']['sample_rate'])
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config['data']['n_mfcc'])
        
        # Include delta and delta-delta MFCCs if specified
        if config['data']['feature_extraction']['include_delta']:
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs = np.vstack((mfccs, mfccs_delta))
        if config['data']['feature_extraction']['include_delta_delta']:
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
            mfccs = np.vstack((mfccs, mfccs_delta2))
        
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Initialize feature list
        features = [mfccs_mean]
        
        # Additional Features
        fe_config = config['data']['feature_extraction']
        if fe_config.get('pitch', False):
            pitch, mag = librosa.core.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
            features.append(pitch_mean)
        
        if fe_config.get('energy', False):
            energy = np.sum(librosa.feature.rms(y=y))
            features.append(energy)
        
        if fe_config.get('zcr', False):
            zcr = librosa.feature.zero_crossing_rate(y=y)
            zcr_mean = np.mean(zcr)
            features.append(zcr_mean)
        
        if fe_config.get('spectral_contrast', False):
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            features.extend(spectral_contrast_mean)
        
        if fe_config.get('mel_spectrogram', False):
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1)
            features.extend(mel_spectrogram_mean)
        
        # Combine all features into a single vector
        features = np.hstack(features)
        
        # Feature Scaling
        if scaler:
            features = scaler.transform([features])[0]
        else:
            scaler = StandardScaler()
            features = scaler.fit_transform([features])[0]
        
        return features, scaler
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None, None

# Data Augmentation Function (Updated)
def augment_data(y, sr):
    augmented_data = []
    try:
        # Example Augmentations: Add noise, time stretching, pitch shifting
        # Add Gaussian noise
        noise = np.random.randn(len(y))
        y_noise = y + 0.005 * noise
        augmented_data.append(y_noise)
        
        # Time stretching
        y_stretch = librosa.effects.time_stretch(y=y, rate=1.1)
        augmented_data.append(y_stretch)
        
        # Pitch shifting with keyword arguments
        y_pitch = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
        augmented_data.append(y_pitch)
        
        return augmented_data
    except Exception as e:
        logger.error(f"Error in data augmentation: {e}")
        return augmented_data

# Process the Turev-DB dataset (folder-based structure)
def process_turev_db(data_dir, config):
    features = []
    labels = []
    scaler = None  # To store the scaler after first feature extraction
    
    # Iterate through each emotion folder
    for emotion in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion)
        if os.path.isdir(emotion_path):
            for file_name in os.listdir(emotion_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(emotion_path, file_name)
                    # Extract features
                    feat, scaler = extract_features(file_path, config, scaler)
                    if feat is not None:
                        features.append(feat)
                        labels.append(emotion_map.get(emotion, 'Neutral'))  # Default to 'Neutral' if not mapped
                        
                        # Data Augmentation (Optional)
                        y, sr = librosa.load(file_path, sr=config['data']['sample_rate'])
                        augmented_samples = augment_data(y, sr)
                        for augmented_y in augmented_samples:
                            # Extract features from augmented data
                            # Note: Not saving augmented audio files
                            # Directly process the augmented signal
                            
                            # Extract MFCCs
                            mfccs = librosa.feature.mfcc(y=augmented_y, sr=sr, n_mfcc=config['data']['n_mfcc'])
                            if config['data']['feature_extraction']['include_delta']:
                                mfccs_delta = librosa.feature.delta(mfccs)
                                mfccs = np.vstack((mfccs, mfccs_delta))
                            if config['data']['feature_extraction']['include_delta_delta']:
                                mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
                                mfccs = np.vstack((mfccs, mfccs_delta2))
                            mfccs_mean = np.mean(mfccs, axis=1)
                            
                            augmented_features = [mfccs_mean]
                            
                            # Additional Features
                            fe_config = config['data']['feature_extraction']
                            if fe_config.get('pitch', False):
                                pitch, mag = librosa.core.piptrack(y=augmented_y, sr=sr)
                                pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
                                augmented_features.append(pitch_mean)
                            
                            if fe_config.get('energy', False):
                                energy = np.sum(librosa.feature.rms(y=augmented_y))
                                augmented_features.append(energy)
                            
                            if fe_config.get('zcr', False):
                                zcr = librosa.feature.zero_crossing_rate(y=augmented_y)
                                zcr_mean = np.mean(zcr)
                                augmented_features.append(zcr_mean)
                            
                            if fe_config.get('spectral_contrast', False):
                                spectral_contrast = librosa.feature.spectral_contrast(y=augmented_y, sr=sr)
                                spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
                                augmented_features.extend(spectral_contrast_mean)
                            
                            if fe_config.get('mel_spectrogram', False):
                                mel_spectrogram = librosa.feature.melspectrogram(y=augmented_y, sr=sr)
                                mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1)
                                augmented_features.extend(mel_spectrogram_mean)
                            
                            # Combine all features into a single vector
                            augmented_features = np.hstack(augmented_features)
                            
                            # Feature Scaling
                            if scaler:
                                augmented_features = scaler.transform([augmented_features])[0]
                            else:
                                scaler = StandardScaler()
                                augmented_features = scaler.fit_transform([augmented_features])[0]
                            
                            features.append(augmented_features)
                            labels.append(emotion_map.get(emotion, 'Neutral'))
    
    return features, labels, scaler

# Load and Process Data
def load_data(config):
    data_dir = config['data']['dataset_path']
    features, labels, scaler = process_turev_db(data_dir, config)
    X = np.array(features)
    y = np.array(labels)
    logger.info(f"Extracted Features Shape: {X.shape}")
    logger.info(f"Labels Shape: {y.shape}")
    return X, y, scaler

# Encode Labels for Multi-Label Classification
def encode_labels(y):
    label_encoder = LabelEncoder()
    y_numeric = label_encoder.fit_transform(y)
    y_cat = to_categorical(y_numeric, num_classes=len(label_encoder.classes_))
    return y_cat, label_encoder

# Split Dataset
def split_dataset(X, y, config):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config['training']['test_size'],
        random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=config['training']['validation_split'],
        random_state=42, stratify=y_temp
    )
    logger.info(f"Training Set Shape: {X_train.shape}, {y_train.shape}")
    logger.info(f"Validation Set Shape: {X_val.shape}, {y_val.shape}")
    logger.info(f"Test Set Shape: {X_test.shape}, {y_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Build Model with Attention Mechanism
def build_model(input_shape, num_classes, config):
    inputs = Input(shape=input_shape)
    
    # Convolutional Layers with Regularization
    x = Conv1D(filters=config['model']['conv_layers'][0]['filters'],
               kernel_size=config['model']['conv_layers'][0]['kernel_size'],
               activation=config['model']['conv_layers'][0]['activation'],
               kernel_regularizer=l2(0.001))(inputs)
    x = MaxPooling1D(pool_size=config['model']['conv_layers'][0]['pool_size'])(x)
    
    x = Conv1D(filters=config['model']['conv_layers'][1]['filters'],
               kernel_size=config['model']['conv_layers'][1]['kernel_size'],
               activation=config['model']['conv_layers'][1]['activation'],
               kernel_regularizer=l2(0.001))(x)
    x = MaxPooling1D(pool_size=config['model']['conv_layers'][1]['pool_size'])(x)
    
    # Bidirectional LSTM Layers
    x = Bidirectional(LSTM(config['model']['lstm_units'], return_sequences=True))(x)
    x = Bidirectional(LSTM(config['model']['lstm_units']))(x)
    
    # Attention Mechanism
    attention = Dense(64, activation='tanh')(x)
    attention = Dense(1, activation='softmax')(attention)
    attention = Flatten()(attention)
    context_vector = x * attention
    
    # Fully Connected Layers
    x = Dense(config['model']['dense_units'], activation='relu')(context_vector)
    x = Dropout(config['model']['dropout_rate'])(x)
    
    # Output Layer
    outputs = Dense(num_classes, activation=config['model']['output_activation'])(x)
    
    # Define Model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile Model
    optimizer = Adam(learning_rate=config['model']['optimizer']['learning_rate'])
    model.compile(optimizer=optimizer, loss=config['model']['loss'], metrics=['accuracy'])
    
    model.summary(print_fn=lambda x: logger.info(x))
    
    return model

# Train Model
def train_model(model, X_train, y_train, X_val, y_val, config, class_weight):
    # Callbacks
    early_stopping = EarlyStopping(
        monitor=config['training']['early_stopping']['monitor'],
        patience=config['training']['early_stopping']['patience'],
        restore_best_weights=config['training']['early_stopping']['restore_best_weights']
    )
    
    checkpoint = ModelCheckpoint(
        filepath='models/best_model.keras',  # Use .h5 extension
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        # Remove 'save_format' to ensure compatibility
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    return history

# Evaluate Model
def evaluate_model(model, X_test, y_test, label_encoder):
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss: {test_loss}")
    logger.info(f"Test Accuracy: {test_accuracy}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_true = y_test.argmax(axis=1)
    y_pred_classes = y_pred_binary.argmax(axis=1)
    
    # Classification Report
    report = classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_)
    logger.info("Classification Report:\n" + report)
    
    # Additional Metrics
    ham_loss = hamming_loss(y_test, y_pred_binary)
    f1_micro = f1_score(y_test, y_pred_binary, average='micro')
    f1_macro = f1_score(y_test, y_pred_binary, average='macro')
    
    logger.info(f"Hamming Loss: {ham_loss}")
    logger.info(f"F1 Score (Micro): {f1_micro}")
    logger.info(f"F1 Score (Macro): {f1_macro}")
    
    return test_loss, test_accuracy, report, ham_loss, f1_micro, f1_macro

# Save Label Encoder and Scaler
def save_preprocessing(label_encoder, scaler, config):
    # Save label encoder
    label_encoder_save_path = config['deployment']['label_encoder_save_path']
    np.save(label_encoder_save_path, label_encoder.classes_)
    logger.info(f"Label encoder classes saved to {label_encoder_save_path}")
    
    # Save scaler (preprocessing pipeline)
    preprocessing_save_path = config['deployment']['preprocessing_save_path']
    joblib.dump(scaler, preprocessing_save_path)
    logger.info(f"Preprocessing scaler saved to {preprocessing_save_path}")

# Main Function
def main():
    try:
        # Load and Process Data
        X, y, scaler = load_data(config)
        
        # Encode Labels
        y_cat, label_encoder = encode_labels(y)
        
        # Split Dataset
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y_cat, config)
        
        # Reshape for CNN input
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Update input shape in config
        input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
        config['model']['input_shape'] = input_shape
        
        # Build Model
        
        # Load Best Model
        model = load_model('models/best_model.keras')  # Load the best_model.h5
        
        # Evaluate Model
        evaluate_model(model, X_test_reshaped, y_test, label_encoder)
        
        # Save Label Encoder and Scaler
        save_preprocessing(label_encoder, scaler, config)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
