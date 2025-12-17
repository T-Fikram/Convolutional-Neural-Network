# 🎭 Face Emotion Recognition (Deteksi Emosi Wajah)

Project ini merupakan aplikasi berbasis **Artificial Intelligence (AI)** untuk mendeteksi ekspresi emosi wajah manusia ke dalam **7 kategori emosi** menggunakan pendekatan **Deep Learning (Convolutional Neural Network / CNN)**.

Aplikasi dikembangkan menggunakan **PyTorch** dengan arsitektur **ResNet50** serta dilengkapi antarmuka web interaktif berbasis **Streamlit**.

---

## 📋 Deskripsi Singkat

Aplikasi ini dilatih menggunakan dataset **FER-2013 (Facial Expression Recognition 2013)**. Model mampu memprediksi emosi dari:

* Gambar wajah yang diunggah (upload file)
* Wajah yang diambil secara langsung melalui **kamera webcam**

### 🎯 Kelas Emosi yang Dideteksi

1. 😡 Angry (Marah)
2. 🤢 Disgust (Jijik)
3. 😨 Fear (Takut)
4. 😄 Happy (Senang)
5. 😐 Neutral (Netral)
6. 😢 Sad (Sedih)
7. 😲 Surprise (Terkejut)

---

## ✨ Fitur Utama

* **Dual Input Mode**
  Mendukung input melalui upload gambar (JPG/PNG) dan kamera webcam.

* **Real-time Prediction**
  Prediksi emosi dilakukan dengan cepat setelah gambar diterima.

* **Confidence Score**
  Menampilkan tingkat keyakinan model dalam bentuk persentase.

* **Visualisasi Probabilitas**
  Grafik batang (Bar Chart) untuk menampilkan probabilitas seluruh kelas emosi.

* **UI Interaktif**
  Antarmuka yang sederhana, bersih, dan responsif menggunakan Streamlit.

---

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python 3.10+
* **Deep Learning Framework:** PyTorch, Torchvision
* **Arsitektur Model:** ResNet50 (Fine-Tuned)
* **Web Interface:** Streamlit
* **Data Processing:** Pandas, NumPy, Pillow

---

## 📂 Struktur Folder

Pastikan struktur folder project Anda seperti berikut sebelum menjalankan aplikasi:

```text
📁 Project_FER2013/
│
├── 📄 app.py                  # Source code utama aplikasi Streamlit
├── 📄 best_emotion_model.pth  # Model hasil training (weights)
├── 📄 requirements.txt        # Daftar dependency Python
├── 📄 README.md               # Dokumentasi project
└── 📂 sample_images/          # (Opsional) Contoh gambar uji
```

---

## 🚀 Cara Instalasi & Menjalankan Aplikasi

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer lokal:

### 1️⃣ Clone atau Download Project

Unduh repository ini atau salin folder project ke komputer Anda.

### 2️⃣ Buat Virtual Environment (Disarankan)

Agar dependency tidak bertabrakan dengan project lain:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependency

Install semua library yang dibutuhkan:


pip install -r requirements.txt


### 4️⃣ Jalankan Aplikasi

Jalankan perintah berikut:


streamlit run app.py


Aplikasi akan terbuka otomatis di browser pada alamat:

http://localhost:8501


## 🧠 Detail Model (Technical Overview)

Model yang digunakan adalah **ResNet50** dengan pendekatan **Fine-Tuning**.

### 🔹 Arsitektur Model

* **Backbone:** ResNet50 (Pretrained ImageNet)
* **Custom Fully Connected Head:**

  * Dropout (mengurangi overfitting)
  * Linear Layer (1024 neuron)
  * Batch Normalization + ReLU
  * Linear Layer (512 neuron)
  * Output Layer (7 neuron sesuai jumlah kelas emosi)


Project ini dibuat untuk memenuhi tugas mata kuliah **Kecerdasan Buatan (Artificial Intelligence)**.


## 📝 Catatan

* Pastikan wajah terlihat **jelas**, **menghadap kamera**, dan memiliki **pencahayaan yang cukup** agar hasil prediksi lebih akurat.
* Bagian yang berada di dalam tanda kurung siku `[ ... ]` wajib disesuaikan dengan data asli Anda sebelum dipublikasikan.
