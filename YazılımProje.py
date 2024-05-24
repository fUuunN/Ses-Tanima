import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np
import os
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import seaborn as sns
import warnings

# Uyarıları kapatma
warnings.filterwarnings("ignore")

# Ses dosyalarının bulunduğu ana dizinler
base_directory = "C:/Users/moonm/OneDrive/Masaüstü/seslerimiz/karışıkLüks/"
subdirectories = ["Ahmet", "Ali", "Can", "Ozlem"]

# Özellik çıkarma
def extract_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y=audio_data)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sr)
    
    features = np.concatenate((np.mean(mfccs.T, axis=0), 
                               np.mean(zcr.T, axis=0), 
                               np.mean(chroma.T, axis=0),
                               np.mean(spectral_contrast.T, axis=0),
                               np.mean(tonnetz.T, axis=0)))
    return features

# Ses dosyalarını ve etiketleri yükleme
X = []
Y = []
for subdirectory in subdirectories:
    audio_directory = os.path.join(base_directory, subdirectory)
    for root, _, files in os.walk(audio_directory):
        for filename in files:
            if filename.endswith(".wav"):  # Ses dosyası uzantısını kontrol edin
                file_path = os.path.join(root, filename)
                X.append(file_path)
                label = os.path.splitext(filename)[0].split('_')[0]  # Dosya adını etiket olarak al
                Y.append(label)

# Yüklenen dosya ve etiket sayısını kontrol et
print(f"Yüklenen dosya sayısı: {len(X)}")
print(f"Yüklenen etiket sayısı: {len(Y)}")

# MFCC özelliklerini çıkarma
mfcc_features = []
for file in X:
    try:
        audio_data, sr = librosa.load(file, sr=22050)
        audio_data = librosa.util.normalize(audio_data)  # Ses verilerini normalize etme
        mfcc_features.append(extract_features(audio_data, sr))  # Özellik çıkarma
    except Exception as e:
        print(f"Error processing {file}: {e}")

mfcc_features = np.array(mfcc_features)

# MFCC özellik sayısını kontrol et
print(f"MFCC özellik sayısı: {len(mfcc_features)}")

# Veriyi bölme
if len(mfcc_features) > 0 and len(Y) > 0:
    X_train, X_test, Y_train, Y_test = train_test_split(mfcc_features, Y, test_size=0.2, random_state=42)

    # Model optimizasyonu
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                               param_grid=param_grid, 
                               cv=5, 
                               n_jobs=-1, 
                               verbose=2)

    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_

    # Modeli kaydetme
    joblib.dump(best_model, "YazilimProjeModel.pkl")

    # Modeli test etme
    Y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
    print("F1 Score:", f1_score(Y_test, Y_pred, average='weighted'))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))

    # Tek bir ses dosyasını tahmin etme
    def predict_single_file(file_path, model):
        try:
            audio_data, sr = librosa.load(file_path, sr=22050)
            audio_data = librosa.util.normalize(audio_data)
            features = extract_features(audio_data, sr)
            features = features.reshape(1, -1)  # Yeniden şekillendir
            prediction = model.predict(features)
            return prediction[0]
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    # Her klasördeki ilk ses dosyasının tahmin sonuçlarının ve ses sinyalinin histogramını çizme
    sns.set(style="whitegrid")
    for subdirectory in subdirectories:
        audio_directory = os.path.join(base_directory, subdirectory)
        first_file_path = None
        for root, _, files in os.walk(audio_directory):
            for filename in files:
                if filename.endswith(".wav"):  # Ses dosyası uzantısını kontrol edin
                    first_file_path = os.path.join(root, filename)
                    break
            if first_file_path:
                break

        if first_file_path:
            print(f"Processing first file in {subdirectory}: {first_file_path}")
            try:
                audio_data, sr = librosa.load(first_file_path, sr=None)
                plt.figure(figsize=(10, 6))
                plt.hist(audio_data, bins=100, color='blue', alpha=0.7)
                plt.xlabel('Genlik')
                plt.ylabel('Frekans')
                plt.title(f"{subdirectory} Klasöründeki İlk Dosyanın Ses Sinyalinin Histogramı")
                plt.grid(True)
                plt.show()
            except Exception as e:
                print(f"Error loading {first_file_path}: {e}")
else:
    print("MFCC özellikleri çıkarılamadı veya ses dosyaları yüklenemedi.")

# Belirli bir ses dosyasını tahmin etme
def predict_specific_file(file_path, model):
    try:
        prediction = predict_single_file(file_path, model)
        print("Predicted Label:", prediction)
    except Exception as e:
        print(f"Error predicting {file_path}: {e}")

# Örnek olarak Ahmet_000.wav dosyasını tahmin et
predict_specific_file("C:/Users/moonm/OneDrive/Masaüstü/seslerimiz/karışıkLüks/Ali/ali_01.wav", best_model)

