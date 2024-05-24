import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import numpy as np
import os
import librosa
from sklearn.ensemble import RandomForestClassifier
import joblib
import pyaudio
import threading
import time
import speech_recognition as sr

# Model yükleme
model = joblib.load("ahmetModeli2.pkl")

# Tanıma motorunu başlatma
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Kişi listeleri
melih_list = []
ozlem_list = []
ali_list = []
can_list = []

# Tahmin fonksiyonu için özellik çıkarımı
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

# Kayıt parametreleri
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
CHUNK = 1024
DURATION = 3  # Saniye cinsinden tahmin aralığı

is_recording = False

def set_window_center(root):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}")
    root.state('zoomed')  # Tam ekran moduna geçiş

def recognize_realtime_speech():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data, language="tr-TR")  # Türkçe tanıma
            return text
        except sr.UnknownValueError:
            return "Anlaşılamadı"
        except sr.RequestError as e:
            return f"Sunucuya ulaşılamıyor; {e}"

def record_and_predict():
    global is_recording
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while is_recording:
        frames = []
        start_time = time.time()
        while time.time() - start_time < DURATION:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
        audio_data = librosa.util.normalize(audio_data)
        features = extract_features(audio_data, RATE).reshape(1, -1)
        if model.predict(features)[0]== "ali":
            prediction = "Ali"
        elif model.predict(features)[0]=="ozlem":
            prediction = "Özlem"
        elif model.predict(features)[0]=="can":
            prediction = "Can"
        elif model.predict(features)[0] =="melih":
            prediction = "Melih"
        
        update_ui(prediction)

        recognized_text = recognize_realtime_speech()
        if prediction == "Melih":
            melih_list.append(recognized_text)
        elif prediction == "Özlem":
            ozlem_list.append(recognized_text)
        elif prediction == "Ali":
            ali_list.append(recognized_text)
        elif prediction == "Can":
            can_list.append(recognized_text)

    stream.stop_stream()
    stream.close()
    p.terminate()
    reset_ui()  # Kayıt durduktan sonra UI'yi sıfırla
    show_results()  # Kayıt durduğunda sonuçları göster

def start_recording():
    global is_recording
    if is_recording:
        return
    is_recording = True
    threading.Thread(target=record_and_predict).start()

def stop_recording():
    global is_recording
    is_recording = False

def update_ui(prediction):
    for name, widgets in profile_widgets.items():
        frame, name_label, record_label = widgets
        if name == prediction:
            frame.config(borderwidth=2, relief="solid", style="Highlighted.TFrame")
            name_label.config(foreground="white", background="#1874cd")
            record_label.config(image=record_photo, background="#83FFFD")
        else:
            frame.config(borderwidth=0, relief="flat", style="Profile.TFrame")
            name_label.config(foreground="black", background="#00868b")
            record_label.config(image=empty_photo, background="#00868b")

def reset_ui():
    for name, widgets in profile_widgets.items():
        frame, name_label, record_label = widgets
        frame.config(borderwidth=0, relief="flat", style="Profile.TFrame")
        name_label.config(foreground="black", background="#00868b")
        record_label.config(image=empty_photo, background="#00868b")

def show_results():
    melih_word_count = sum(len(sentence.split()) for sentence in melih_list if sentence != "Anlaşılamadı")
    ozlem_word_count = sum(len(sentence.split()) for sentence in ozlem_list if sentence != "Anlaşılamadı")
    ali_word_count = sum(len(sentence.split()) for sentence in ali_list if sentence != "Anlaşılamadı")
    can_word_count = sum(len(sentence.split()) for sentence in can_list if sentence != "Anlaşılamadı")

    total_word_count = melih_word_count + ozlem_word_count + ali_word_count + can_word_count

    melih_percentage = (melih_word_count / total_word_count) * 100 if total_word_count != 0 else 0
    ozlem_percentage = (ozlem_word_count / total_word_count) * 100 if total_word_count != 0 else 0
    ali_percentage = (ali_word_count / total_word_count) * 100 if total_word_count != 0 else 0
    can_percentage = (can_word_count / total_word_count) * 100 if total_word_count != 0 else 0

    result_text = f"Melih ({melih_word_count} kelime, %{melih_percentage:.2f}):\n{' '.join(melih_list)}\n\n"
    result_text += f"Özlem ({ozlem_word_count} kelime, %{ozlem_percentage:.2f}):\n{' '.join(ozlem_list)}\n\n"
    result_text += f"Ali ({ali_word_count} kelime, %{ali_percentage:.2f}):\n{' '.join(ali_list)}\n\n"
    result_text += f"Can ({can_word_count} kelime, %{can_percentage:.2f}):\n{' '.join(can_list)}\n\n"
    result_text += f"Toplam kelime sayısı: {total_word_count}"

    result_window = tk.Toplevel(root)
    result_window.title("Sonuçlar")
    result_textbox = ScrolledText(result_window, wrap=tk.WORD, width=100, height=30)
    result_textbox.pack(expand=True, fill=tk.BOTH)
    result_textbox.insert(tk.END, result_text)
    result_textbox.config(state=tk.DISABLED)


root = tk.Tk()
root.title("Ses Tanımlama Projesi")
set_window_center(root)
root.configure(background="#00868b")

# Stil tanımlamaları
style = ttk.Style()
style.configure("Normal.TFrame")
style.configure("Highlighted.TFrame", background="#1874cd")
style.configure("Red.TFrame", background="#00868b")
style.configure("Profile.TFrame", background="#00868b")

profiles = [
    ("melih.jpg", "Melih"),
    ("ozlem.jpg", "Özlem"),
    ("ali.jpg", "Ali"),
    ("Can.jpg", "Can")
]

profile_frame = ttk.Frame(root, style="Red.TFrame")
profile_frame.pack(pady=150)  # Yukarıdan biraz boşluk bırakmak için pady değerini azalttık
profile_frame.config(borderwidth=2)

profile_widgets = {}

for path, name in profiles:
    sub_frame = ttk.Frame(profile_frame,style="Profile.TFrame")
    sub_frame.pack(side=tk.LEFT, padx=50, pady=40)

    img = Image.open(path).resize((150, 150))
    photo = ImageTk.PhotoImage(img)

    label = tk.Label(sub_frame, image=photo, background="white")
    label.image = photo
    label.pack()

    name_label = tk.Label(sub_frame, text=name, foreground="black", font=("Arial", 18, "bold"), background="#00868b")
    name_label.pack()

    empty_image = Image.new('RGBA', (45, 45), (255, 255, 255, 0))
    empty_photo = ImageTk.PhotoImage(empty_image)

    record_icon = Image.open("record_icon3.png").resize((100, 100))  # Boyutları 2 kat büyütüldü
    record_photo = ImageTk.PhotoImage(record_icon)

    record_label = tk.Label(sub_frame, image=empty_photo, background="#00868b")
    record_label.image = empty_photo
    record_label.pack()

    profile_widgets[name] = (sub_frame, name_label, record_label)

button_frame = ttk.Frame(root, width=800, height=50, style="Red.TFrame")
button_frame.pack(pady=20, expand=True)  # Ortalamak için expand eklenmiştir

record_img = Image.open("play-button.png").resize((100, 90))  # Boyutları 2 kat büyütüldü
stop_img = Image.open("pause-button.png").resize((100, 90))  # Boyutları 2 kat büyütüldü

record_icon = ImageTk.PhotoImage(record_img)
stop_icon = ImageTk.PhotoImage(stop_img)

record_button = ttk.Button(button_frame, image=record_icon, command=start_recording, cursor="hand2")
record_button.grid(row=0, column=0, padx=20,pady=100)

stop_button = ttk.Button(button_frame, image=stop_icon, command=stop_recording, cursor="hand2")
stop_button.grid(row=0, column=1, padx=20,pady=100)

button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)

root.mainloop()