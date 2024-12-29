# emotion_recognition_final.py

import os
import librosa
import numpy as np
import yaml
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, hamming_loss, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Bidirectional, LSTM,
                                     Dense, Dropout, Flatten, Multiply, Permute, RepeatVector)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# Konfigürasyon dosyasını yükle
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Gerekli klasörlerin varlığını sağla
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Loglama ayarlarını yap
logging.basicConfig(
    filename=config['logging']['file'],
    level=getattr(logging, config['logging']['level'].upper(), logging.INFO),
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger()

# oneDNN optimizasyonlarını devre dışı bırak (ilgili uyarılar alıyorsanız)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Duygu haritasını tanımla (veri setleri arasında etiketleri standardize etmek için)
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
    # Gerekirse daha fazla haritalama ekleyin
}

# Özellik Çıkarım Fonksiyonu (Geliştirilmiş)
def extract_features(y, sr, config, scaler=None):
    try:
        # MFCC'leri çıkar
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config['data']['n_mfcc'])
        
        # Delta ve delta-delta MFCC'leri dahil et
        if config['data']['feature_extraction']['include_delta']:
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs = np.vstack((mfccs, mfccs_delta))
        if config['data']['feature_extraction']['include_delta_delta']:
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
            mfccs = np.vstack((mfccs, mfccs_delta2))
        
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Özellik listesini başlat
        features = [mfccs_mean]
        
        # Ekstra Özellikler
        additional_features = config['data']['feature_extraction']['additional_features']
        if additional_features.get('pitch', False):
            pitch, _ = librosa.core.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
            features.append(pitch_mean)
        
        if additional_features.get('energy', False):
            energy = np.sum(librosa.feature.rms(y=y))
            features.append(energy)
        
        if additional_features.get('zcr', False):
            zcr = librosa.feature.zero_crossing_rate(y=y)
            zcr_mean = np.mean(zcr)
            features.append(zcr_mean)
        
        if additional_features.get('spectral_contrast', False):
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            features.extend(spectral_contrast_mean)
        
        if additional_features.get('mel_spectrogram', False):
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1)
            features.extend(mel_spectrogram_mean)
        
        # Tüm özellikleri tek bir vektörde birleştir
        features = np.hstack(features)
        
        # Özellik Ölçeklendirme
        if scaler:
            features = scaler.transform([features])[0]
        else:
            scaler = StandardScaler()
            features = scaler.fit_transform([features])[0]
        
        return features, scaler
    except Exception as e:
        logger.error(f"Özellik çıkarımında hata: {e}")
        return None, None

# Veri Artırma Fonksiyonu (Güncellenmiş)
def augment_data(y, sr, config):
    augmented_data = []
    try:
        # Veri artırma ayarlarını al
        augmentation_config = config['data']['augmentation']
        
        # Gürültü ekleme
        if augmentation_config.get('add_noise', False):
            noise = np.random.randn(len(y))
            noise_amp = augmentation_config.get('noise_amplitude', 0.005)
            y_noise = y + noise_amp * noise
            augmented_data.append(y_noise)
        
        # Zaman esnetme
        if augmentation_config.get('time_stretch', False):
            rate = augmentation_config.get('stretch_rate', 1.1)
            y_stretch = librosa.effects.time_stretch(y=y, rate=rate)
            # Esnetilen sinyalin aynı uzunlukta olmasını sağla
            if len(y_stretch) > len(y):
                y_stretch = y_stretch[:len(y)]
            else:
                y_stretch = np.pad(y_stretch, (0, max(0, len(y) - len(y_stretch))), 'constant')
            augmented_data.append(y_stretch)
        
        # Pitch kaydırma
        if augmentation_config.get('pitch_shift', False):
            n_steps = augmentation_config.get('pitch_steps', 2)
            y_pitch = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
            augmented_data.append(y_pitch)
        
        # Diğer veri artırma tekniklerini ekleyebilirsiniz
        
        return augmented_data
    except Exception as e:
        logger.error(f"Veri artırmada hata: {e}")
        return augmented_data

# Turev-DB veri setini işleme fonksiyonu (klasör tabanlı yapı)
def process_turev_db(data_dir, config):
    features = []
    labels = []
    scaler = None  # İlk özellik çıkarımdan sonra ölçekleyici saklanacak
    
    # Her duygu klasörünü dolaş
    for emotion in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion)
        if os.path.isdir(emotion_path):
            for file_name in os.listdir(emotion_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(emotion_path, file_name)
                    try:
                        # Ses dosyasını yükle
                        y, sr = librosa.load(file_path, sr=config['data']['sample_rate'])
                        
                        # Orijinal veriden özellikleri çıkar
                        feat, scaler = extract_features(y, sr, config, scaler)
                        if feat is not None:
                            features.append(feat)
                            labels.append(emotion_map.get(emotion, 'Neutral'))  # Harita yoksa 'Neutral' olarak ata
                        
                        # Veri Artırma (Opsiyonel)
                        augmentation_config = config['data']['augmentation']
                        if augmentation_config.get('enable', False):
                            augmented_samples = augment_data(y, sr, config)
                            for augmented_y in augmented_samples:
                                feat_aug, scaler = extract_features(augmented_y, sr, config, scaler)
                                if feat_aug is not None:
                                    features.append(feat_aug)
                                    labels.append(emotion_map.get(emotion, 'Neutral'))
                    except Exception as e:
                        logger.error(f"{file_path} dosyasını işlerken hata: {e}")
    
    return features, labels, scaler

# Veriyi yükleme ve işleme fonksiyonu
def load_data(config):
    data_dir = config['data']['dataset_path']
    features, labels, scaler = process_turev_db(data_dir, config)
    X = np.array(features)
    y = np.array(labels)
    logger.info(f"Çıkarılan Özelliklerin Şekli: {X.shape}")
    logger.info(f"Etiketlerin Şekli: {y.shape}")
    return X, y, scaler

# Çoklu Sınıf Sınıflandırma için Etiketleri Kodlama
def encode_labels(y):
    label_encoder = LabelEncoder()
    y_numeric = label_encoder.fit_transform(y)
    y_cat = to_categorical(y_numeric, num_classes=len(label_encoder.classes_))
    return y_cat, label_encoder

# Veri Setini Bölme Fonksiyonu
def split_dataset(X, y, config):
    test_size = config['training']['test_size']
    validation_split = config['training']['validation_split']
    
    # Eğitim ve geçici seti ayır
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size,
        random_state=42, stratify=y
    )
    # Geçici seti doğrulama ve test setine ayır
    val_size = validation_split / (1 - test_size)  # Doğrulama boyutunu ayarla
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size,
        random_state=42, stratify=y_temp
    )
    logger.info(f"Eğitim Setinin Şekli: {X_train.shape}, {y_train.shape}")
    logger.info(f"Doğrulama Setinin Şekli: {X_val.shape}, {y_val.shape}")
    logger.info(f"Test Setinin Şekli: {X_test.shape}, {y_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Dikkat Katmanı Fonksiyonu (Opsiyonel)
def attention_3d_block(inputs, config):
    # Girdilerin şekli: (batch_size, time_steps, input_dim)
    time_steps = int(inputs.shape[1])
    input_dim = int(inputs.shape[2])
    
    # Dikkat mekanizması için yoğun katmanlar
    a = Dense(config['model']['attention_units'], activation='tanh')(inputs)
    a = Dense(1, activation='softmax')(a)
    a = Flatten()(a)
    a = RepeatVector(input_dim)(a)
    a = Permute([2, 1])(a)
    
    # Girdi ile dikkat ağırlıklarını çarp
    output_attention_mul = Multiply()([inputs, a])
    return output_attention_mul

# Model Oluşturma Fonksiyonu (Opsiyonel Dikkat Mekanizması ile)
def build_model(input_shape, num_classes, config):
    inputs = Input(shape=input_shape)
    
    # Konvolüsyon Katmanları ve Havuzlama
    x = inputs
    for conv_layer in config['model']['conv_layers']:
        x = Conv1D(filters=conv_layer['filters'],
                   kernel_size=conv_layer['kernel_size'],
                   activation=conv_layer['activation'],
                   kernel_regularizer=l2(conv_layer.get('l2', 0.001)),
                   padding='same')(x)
        x = MaxPooling1D(pool_size=conv_layer['pool_size'])(x)
    
    # Bidirectional LSTM Katmanları
    for lstm_layer in config['model']['lstm_layers']:
        x = Bidirectional(LSTM(lstm_layer['units'], return_sequences=lstm_layer.get('return_sequences', False)))(x)
    
    # Dikkat Mekanizması (Opsiyonel)
    if config['model']['use_attention']:
        x = attention_3d_block(x, config)
        x = Flatten()(x)
    
    # Tam Bağlantılı Katmanlar
    for dense_layer in config['model']['dense_layers']:
        x = Dense(dense_layer['units'], activation=dense_layer['activation'],
                  kernel_regularizer=l2(dense_layer.get('l2', 0.001)))(x)
        if 'dropout' in dense_layer:
            x = Dropout(dense_layer['dropout'])(x)
    
    # Çıktı Katmanı
    outputs = Dense(num_classes, activation=config['model']['output_activation'])(x)
    
    # Modeli Tanımla
    model = Model(inputs=inputs, outputs=outputs)
    
    # Modeli Derle
    optimizer = Adam(learning_rate=config['model']['optimizer']['learning_rate'])
    model.compile(optimizer=optimizer, loss=config['model']['loss'], metrics=['accuracy'])
    
    # Model özetini log dosyasına yaz
    model.summary(print_fn=lambda x: logger.info(x))
    
    return model

# Modeli Eğitme Fonksiyonu
def train_model(model, X_train, y_train, X_val, y_val, config, class_weight):
    # Erken Durdurma ve Model Kontrol Noktası Geri Çağırma Ayarları
    early_stopping = EarlyStopping(
        monitor=config['training']['early_stopping']['monitor'],
        patience=config['training']['early_stopping']['patience'],
        restore_best_weights=config['training']['early_stopping']['restore_best_weights']
    )
    
    checkpoint = ModelCheckpoint(
        filepath=config['deployment']['model_save_path'],
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Modeli Eğit
    history = model.fit(
        X_train, y_train,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Eğitim geçmişini log dosyasına yaz
    for epoch in range(len(history.history['loss'])):
        logger.info(
            f"Epoch {epoch+1}/{config['training']['epochs']}: "
            f"loss={history.history['loss'][epoch]:.4f}, "
            f"val_loss={history.history['val_loss'][epoch]:.4f}, "
            f"accuracy={history.history['accuracy'][epoch]:.4f}, "
            f"val_accuracy={history.history['val_accuracy'][epoch]:.4f}"
        )
    
    return history

# Modeli Değerlendirme Fonksiyonu
def evaluate_model(model, X_test, y_test, label_encoder):
    # Test verisi üzerinde modeli değerlendir
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Kayıp: {test_loss}")
    logger.info(f"Test Doğruluk: {test_accuracy}")
    
    # Tahminleri al
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Sınıflandırma Raporu
    report = classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_)
    logger.info("Sınıflandırma Raporu:\n" + report)
    
    # Ek Metrikler
    ham_loss = hamming_loss(y_test, to_categorical(y_pred_classes, num_classes=y_test.shape[1]))
    f1_micro = f1_score(y_test, to_categorical(y_pred_classes, num_classes=y_test.shape[1]), average='micro')
    f1_macro = f1_score(y_test, to_categorical(y_pred_classes, num_classes=y_test.shape[1]), average='macro')
    
    logger.info(f"Hamming Kayıp: {ham_loss}")
    logger.info(f"F1 Skoru (Micro): {f1_micro}")
    logger.info(f"F1 Skoru (Macro): {f1_macro}")
    
    # Karışıklık Matrisi
    cm = confusion_matrix(y_true, y_pred_classes)
    logger.info(f"Karışıklık Matrisi:\n{cm}")
    
    return test_loss, test_accuracy, report, ham_loss, f1_micro, f1_macro

# Ön İşlem Nesnelerini Kaydetme Fonksiyonu
def save_preprocessing(label_encoder, scaler, config):
    try:
        # Etiket kodlayıcıyı kaydet
        label_encoder_save_path = config['deployment']['label_encoder_save_path']
        np.save(label_encoder_save_path, label_encoder.classes_)
        logger.info(f"Etiket kodlayıcı sınıfları {label_encoder_save_path} yoluna kaydedildi.")
        
        # Ölçekleyiciyi kaydet
        preprocessing_save_path = config['deployment']['preprocessing_save_path']
        joblib.dump(scaler, preprocessing_save_path)
        logger.info(f"Ön işleme ölçekleyici {preprocessing_save_path} yoluna kaydedildi.")
    except Exception as e:
        logger.error(f"Ön işleme nesnelerini kaydederken hata: {e}")

# Sınıf Ağırlıklarını Hesaplama Fonksiyonu
def compute_weights(y_train):
    try:
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        logger.info(f"Hesaplanan sınıf ağırlıkları: {class_weight_dict}")
        return class_weight_dict
    except Exception as e:
        logger.error(f"Sınıf ağırlıklarını hesaplarken hata: {e}")
        return None

# Ana Fonksiyon
def main():
    try:
        # Veriyi yükle ve işle
        X, y, scaler = load_data(config)
        
        # Veri ve etiketlerin boş olup olmadığını kontrol et
        if len(X) == 0 or len(y) == 0:
            logger.error("Çıkarılan özellikler veya etiketler boş. Veri setinizi ve özellik çıkarımını kontrol edin.")
            return
        
        # Etiketleri kodla
        y_cat, label_encoder = encode_labels(y)
        
        # Veri setini böl
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y_cat, config)
        
        # CNN girişi için veriyi yeniden şekillendir
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Girdi şekli yapılandırmasını güncelle
        input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
        config['model']['input_shape'] = input_shape
        
        # Sınıf ağırlıklarını hesapla
        y_train_classes = np.argmax(y_train, axis=1)
        class_weight_dict = compute_weights(y_train_classes)
        
        # Modeli oluştur
        model = build_model(input_shape, y_cat.shape[1], config)
        
        # Modeli eğit
        history = train_model(model, X_train_reshaped, y_train, X_val_reshaped, y_val, config, class_weight=class_weight_dict)
        
        # En iyi modeli yükle
        best_model_path = config['deployment']['model_save_path']
        if os.path.exists(best_model_path):
            best_model = load_model(best_model_path)
            logger.info(f"En iyi model {best_model_path} yolundan yüklendi.")
        else:
            logger.error(f"En iyi model dosyası {best_model_path} yolunda bulunamadı. Değerlendirme atlanıyor.")
            return
        
        # Modeli değerlendir
        evaluate_model(best_model, X_test_reshaped, y_test, label_encoder)
        
        # Etiket kodlayıcı ve ölçekleyiciyi kaydet
        save_preprocessing(label_encoder, scaler, config)
        
    except Exception as e:
        logger.error(f"Ana fonksiyonda hata oluştu: {e}")

if __name__ == "__main__":
    main()
