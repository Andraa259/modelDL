import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import base64

# ======================
# KONFIGURASI HALAMAN
# ======================
st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .title-text {
        font-size: 3rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0;
    }
    
    .subtitle-text {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-card-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }
    
    .result-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        animation: pulse 2s infinite;
    }
    
    .result-low {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .info-box {
        background-color: #dbeafe;
        border-left: 5px solid #3b82f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 50px;
        padding: 15px 40px;
        border: none;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stSlider > div > div > div {
        background: #667eea;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

feature_names = [
        'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
        'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic',
        'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma',
        'KidneyDisease', 'SkinCancer'
    ]

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, label_encoders

try:
    model, scaler, label_encoders = load_models()
    model_ready = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    model_ready = False

# ======================
# HEADER
# ======================
st.markdown('<h1 class="title-text">❤️ HeartGuard AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Sistem Prediksi Risiko Penyakit Jantung Berbasis Deep Learning</p>', unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown("## 🩺 Tentang Aplikasi")
    st.info("""
    Aplikasi ini menggunakan model **DeepResNet** yang dilatih dengan dataset BRFSS 2020.
    
    Model ini dapat membantu mendeteksi risiko penyakit jantung berdasarkan indikator kesehatan dan gaya hidup.
    """)
    
    st.markdown("## 📊 Statistik Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Data", value="319K")
    with col2:
        st.metric(label="Fitur", value="17")
    
    st.markdown("## ⚠️ Disclaimer")
    st.warning("""
    Aplikasi ini hanya sebagai alat skrining awal.
    Hasil prediksi bukan pengganti diagnosis dokter.
    Konsultasikan dengan tenaga medis profesional.
    """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Dibuat oleh")
    st.markdown("Kelompok 8 - Deep Learning")

# ======================
# TABS
# ======================
tab1, tab2 = st.tabs(["📝 Input Data", "📖 Panduan"])

# ======================
# TAB 1: INPUT DATA
# ======================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown("### 👤 Informasi Demografi")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sex = st.selectbox(
            "Jenis Kelamin",
            ["Male", "Female"],
            help="Pilih jenis kelamin Anda"
        )
        age_category = st.selectbox(
            "Kategori Usia",
            ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
             "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"],
            help="Pilih rentang usia Anda"
        )
    
    with col2:
        bmi = st.slider(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1,
            help="Indeks Massa Tubuh"
        )
        st.markdown(f"<p style='text-align:center; color:#667eea; font-weight:600;'>{bmi}</p>", unsafe_allow_html=True)
    
    with col3:
        sleep_time = st.slider(
            "Durasi Tidur (jam)",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5,
            help="Rata-rata jam tidur per hari"
        )
        st.markdown(f"<p style='text-align:center; color:#667eea; font-weight:600;'>{sleep_time} jam</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏥 Kondisi Kesehatan")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        stroke = st.radio(
            "Riwayat Stroke",
            ["Yes", "No"],
            horizontal=True
        )
        diabetic = st.selectbox(
            "Diabetes",
            ["Yes", "No", "No, borderline diabetes", "Yes (during pregnancy)"]
        )
        asthma = st.radio(
            "Asma",
            ["Yes", "No"],
            horizontal=True
        )
    
    with col5:
        diff_walking = st.radio(
            "Kesulitan Berjalan",
            ["Yes", "No"],
            horizontal=True
        )
        kidney_disease = st.radio(
            "Penyakit Ginjal",
            ["Yes", "No"],
            horizontal=True
        )
        skin_cancer = st.radio(
            "Kanker Kulit",
            ["Yes", "No"],
            horizontal=True
        )
    
    with col6:
        physical_health = st.slider(
            "Hari Fisik Tidak Sehat (30 hari terakhir)",
            min_value=0,
            max_value=30,
            value=0,
            help="Jumlah hari kondisi fisik Anda tidak baik"
        )
        mental_health = st.slider(
            "Hari Mental Tidak Sehat (30 hari terakhir)",
            min_value=0,
            max_value=30,
            value=0,
            help="Jumlah hari kondisi mental Anda tidak baik"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏃 Gaya Hidup")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        smoking = st.radio(
            "Perokok",
            ["Yes", "No"],
            horizontal=True,
            help="Apakah Anda perokok?"
        )
        physical_activity = st.radio(
            "Aktivitas Fisik",
            ["Yes", "No"],
            horizontal=True,
            help="Apakah Anda berolahraga dalam 30 hari terakhir?"
        )
    
    with col8:
        alcohol_drinking = st.radio(
            "Konsumsi Alkohol",
            ["Yes", "No"],
            horizontal=True,
            help="Apakah Anda mengonsumsi alkohol berlebihan?"
        )
        gen_health = st.selectbox(
            "Kesehatan Umum",
            ["Excellent", "Very good", "Good", "Fair", "Poor"],
            help="Bagaimana penilaian kesehatan Anda secara umum?"
        )
    
    with col9:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("https://img.icons8.com/color/96/heart-health.png", width=100)
        st.markdown("<p style='text-align:center; color:#64748b;'>Jaga kesehatan jantung Anda</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Tombol prediksi
    st.markdown("<br>", unsafe_allow_html=True)
    col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
    
    with col_pred2:
        predict_button = st.button("🔮 Prediksi Risiko Sekarang", use_container_width=True)
    
    # Proses prediksi
    if predict_button and model_ready:
        with st.spinner("🧠 Sedang menganalisis data Anda..."):
            data = {
                'BMI': bmi,
                'Smoking': smoking,
                'AlcoholDrinking': alcohol_drinking,
                'Stroke': stroke,
                'PhysicalHealth': physical_health,
                'MentalHealth': mental_health,
                'DiffWalking': diff_walking,
                'Sex': sex,
                'AgeCategory': age_category,
                'Diabetic': diabetic,
                'PhysicalActivity': physical_activity,
                'GenHealth': gen_health,
                'SleepTime': sleep_time,
                'Asthma': asthma,
                'KidneyDisease': kidney_disease,
                'SkinCancer': skin_cancer
            }
            
            input_data = []
            for feature in feature_names:
                value = data[feature]
                if feature in label_encoders:
                    value = label_encoders[feature].transform([value])[0]
                else:
                    value = float(value)
                input_data.append(value)
            
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            input_reshaped = input_scaled.reshape(1, len(feature_names), 1)
            
            prob = model.predict(input_reshaped, verbose=0)[0][0]
            result = "Yes" if prob > 0.5 else "No"
            probability = prob * 100
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if result == "Yes":
            st.markdown(
                f'<div class="result-high">'
                f'⚠️ Risiko Tinggi<br>'
                f'<span style="font-size:3rem;">{probability:.2f}%</span><br>'
                f'<span style="font-size:1rem; font-weight:400;">Anda berisiko terkena penyakit jantung</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("""
            **Rekomendasi:**
            - Segera konsultasikan dengan dokter atau kardiolog
            - Perbaiki pola makan dengan mengurangi garam, gula, dan lemak jenuh
            - Tingkatkan aktivitas fisik secara bertahap
            - Berhenti merokok dan batasi konsumsi alkohol
            - Kelola stres dan pastikan tidur cukup
            - Pantau tekanan darah dan kadar kolesterol secara rutin
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown(
                f'<div class="result-low">'
                f'✅ Risiko Rendah<br>'
                f'<span style="font-size:3rem;">{probability:.2f}%</span><br>'
                f'<span style="font-size:1rem; font-weight:400;">Risiko penyakit jantung Anda relatif rendah</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            **Pertahankan gaya hidup sehat:**
            - Terus konsumsi makanan sehat dan seimbang
            - Olahraga teratur minimal 30 menit per hari
            - Tidur 7-8 jam setiap malam
            - Hindari merokok dan konsumsi alkohol berlebihan
            - Lakukan pemeriksaan kesehatan rutin
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Progress bar visual
        st.markdown("### 📊 Visualisasi Risiko")
        st.progress(min(float(prob), 1.0))
        
        # Detail probabilitas
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(label="Probabilitas Risiko", value=f"{probability:.2f}%")
        with col_b:
            st.metric(label="Hasil Prediksi", value=result)
        with col_c:
            st.metric(label="Threshold", value="50%")
    
    elif predict_button and not model_ready:
        st.error("Model belum siak. Pastikan file model, scaler, dan label encoders sudah benar.")

# ======================
# TAB 2: PANDUAN
# ======================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 📚 Panduan Penggunaan")
    
    st.markdown("""
    ### Cara Menggunakan Aplikasi:
    1. Isi semua data kesehatan dan gaya hidup pada tab **Input Data**
    2. Pastikan data yang diisi sesuai dengan kondisi Anda
    3. Klik tombol **Prediksi Risiko Sekarang**
    4. Aplikasi akan menampilkan probabilitas risiko penyakit jantung
    
    ### Faktor Risiko yang Dinilai:
    - **Demografi**: jenis kelamin, usia, BMI
    - **Kondisi Kesehatan**: stroke, diabetes, asma, penyakit ginjal, kanker kulit
    - **Gaya Hidup**: merokok, alkohol, aktivitas fisik, durasi tidur
    
    ### Metode:
    Model ini menggunakan arsitektur **DeepResNet** dengan teknik penyeimbangan data **LoRAS** dan dijelaskan dengan **Explainable AI**.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## ⚠️ Disclaimer")
    st.markdown("""
    Aplikasi ini hanya alat bantu skrining awal dan tidak menggantikan diagnosis medis profesional.
    Hasil prediksi bersifat probabilitas dan harus dikonfirmasi oleh dokter atau tenaga medis.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown(
    '<p class="footer">'
    'Made by Kelompok 8 - Deep Learning | Universitas 17 Agustus 1945 Surabaya'
    '</p>',
    unsafe_allow_html=True
)
