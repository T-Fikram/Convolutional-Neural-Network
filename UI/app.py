import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os        
import gdown

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Deteksi Emosi Wajah - FER 2013",
    page_icon="🎭",
    layout="centered"
)

# ==========================================
# 2. DEFINISI KELAS CUSTOM (WAJIB SAMA DENGAN NOTEBOOK)
# ==========================================
class EmotionResNet50(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(EmotionResNet50, self).__init__()
        
        # Load pretrained ResNet50 (weights=None karena kita load manual)
        self.backbone = models.resnet50(weights=None)
        
        # Replace the final fully connected layer (Sesuai notebook)
        in_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate-0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate-0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 3. SETUP VARIABEL & TRANSFORMASI
# ==========================================
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. FUNGSI MEMUAT MODEL
# ==========================================
@st.cache_resource
def load_model():
    # -----------------------------------------------------------
    # GANTI ID INI DENGAN ID FILE GOOGLE DRIVE ANDA
    # -----------------------------------------------------------
    
    file_id = '1FxaBs9H7YG6HJP0XIlHVQXtEr166o2r7'
    
    output_model_file = 'best_emotion_model.pth'
    
    # Cek apakah file sudah ada? Jika belum, download dulu
    if not os.path.exists(output_model_file):
        url = f'https://drive.google.com/uc?id={file_id}'
        st.warning("Sedang mengunduh model dari Google Drive... (Hanya sekali)")
        gdown.download(url, output_model_file, quiet=False)
        st.success("Model berhasil diunduh!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionResNet50(num_classes=len(CLASS_NAMES))
    
    try:
        checkpoint = torch.load(output_model_file, map_location=device)
        
        if isinstance(checkpoint, dict) and 'backbone.conv1.weight' in checkpoint:
             model.load_state_dict(checkpoint)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

    model = model.to(device)
    model.eval()
    return model, device
    
# ==========================================
# 5. FUNGSI PREDIKSI
# ==========================================
def predict_image(image, model, device):
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    image_tensor = data_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    top_p, top_class = probabilities.topk(1, dim=1)
    return top_class.item(), top_p.item(), probabilities.cpu().numpy()[0]

# ==========================================
# 6. UI UTAMA (MODIFIKASI: TAB INPUT)
# ==========================================
st.title("Analisis Ekspresi Wajah")
st.write("Gunakan **Upload Gambar** atau **Kamera Langsung** untuk mendeteksi emosi.")

model, device = load_model()

if model is not None:
    # Buat Tab agar UI rapi
    tab1, tab2 = st.tabs(["Upload File", "Ambil Foto"])

    # Variabel untuk menampung gambar dari salah satu sumber
    final_image = None
    
    # === TAB 1 UPLOAD FILE ===
    with tab1:
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            final_image = Image.open(uploaded_file)

    # === TAB 2 KAMERA ===
    with tab2:
        camera_input = st.camera_input("Ambil foto wajah Anda")
        if camera_input is not None:
            final_image = Image.open(camera_input)

    # === PROSES ANALISIS (Jika ada gambar) ===
    if final_image is not None:
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(final_image, caption='Input Gambar', use_container_width=True)
        
        with col2:
            st.subheader("Hasil Prediksi")
            
            # Tombol Analisis
            if st.button("Analisis Emosi 🔍", key="btn_predict"):
                with st.spinner('Sedang memproses...'):
                    class_idx, conf, all_probs = predict_image(final_image, model, device)
                    
                    predicted_label = CLASS_NAMES[class_idx]
                    
                    # Warna alert sesuai emosi (Fitur tambahan biar keren)
                    if predicted_label in ['happy', 'surprise']:
                        st.success(f"Emosi: **{predicted_label.upper()}**")
                    elif predicted_label in ['angry', 'disgust', 'fear', 'sad']:
                        st.error(f"Emosi: **{predicted_label.upper()}**")
                    else:
                        st.info(f"Emosi: **{predicted_label.upper()}**")
                        
                    st.write(f"Confidence: **{conf*100:.2f}%**")
                    
                    # Chart
                    probs_dict = {k: float(v) for k, v in zip(CLASS_NAMES, all_probs)}
                    st.bar_chart(probs_dict)

# Footer & Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3242/3242257.png", width=50)
    st.title("Info Project")
    
    st.info("""
    **Mata Kuliah:** Kecerdasan Buatan (AI)            

    **Kelompok :**  2 (dua)

    **Dataset:** FER 2013

    **Model:** Custom ResNet50
    """)
    
    st.caption("Dibuat dengan Streamlit & PyTorch")
