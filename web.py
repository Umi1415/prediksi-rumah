import pickle
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from PIL import Image

# Load model data
model_data = pickle.load(open('model_prediksi_harga_rumah.sav', 'rb'))
model_regresi = model_data['model']
scaler = model_data['scaler']
poly = model_data['poly']

# Set page config
st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide", page_icon="🏡")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Navigasi sidebar
image = Image.open('icon.jpg')

with st.sidebar:
    st.image(image, use_container_width=True)
    st.markdown("### MENU")
    if st.button("🏡 Beranda\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "Home"
    if st.button("📊 Dataset (CSV)\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "Dataset"
    if st.button("📈 Visualization\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "Visualization"
    if st.button("💵 Prediksi Harga Rumah\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "Prediction"
    if st.button("🔎 Penjelasan aplikasi\u2003\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "Methods"
    if st.button("ℹ️ Tentang kami\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "About"

st.markdown(
    """
    <style>
        .stApp {
            background: #F0F3FA;
            color: #395886;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display page content based on selected page
if st.session_state["page"] == "Home":
    st.markdown(""" 
    <div style="text-align: center; font-size: 50px; font-weight: bold;">
    Prediksi Harga Rumah
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("Selamat Datang di Aplikasi Prediksi Harga Rumah!")
    image_path = "rumah.jpg"
    st.image(image_path, use_container_width=True)

    st.write("Website kami, berfokus pada prediksi harga rumah khususnya pada wilayah Tebet dan Jakarta Selatan. Dengan fitur yang tersedia, kami menyediakan informasi yang berguna bagi pembeli, penjual dan investor.")
    st.header("💡 Fitur Utama")
    st.info("""
        - **Dataset (CSV):** Menampilkan dataset rumah yang berisikan informasi seperti luas bangunan, luas tanah, jumlah kamar, dan harga rumah. 
        - **Visualization:** Visualisasi data berupa grafik kepada pelanggan.
        - **Prediksi Harga Rumah:** Perhitungan harga rumah berdasarkan input data spesifik dari pengguna.
        - **Penjelasan aplikasi:** Informasi mengenai cara kerja dari website dan metode yang digunakan.
    """)

    st.header("💡Tujuan dan Manfaat")
    st.subheader("**Tujuan**")
    st.markdown("""
        - Membantu memahami harga pasar melalui grafik dan dataset yang tersedia
        - Menyediakan alat perhitungan yang berguna untuk memprediksi harga rumah dengan lebih akurat
        - Menambah wawasan mengenai faktor yang memengaruhi harga rumah
        - Mempermudah analisis tren harga berdasarkan historis
    """)
    st.subheader("**Manfaat**")
    st.markdown("""
        - Mencari rumah yang sesuai dengan anggaran yang dimiliki
        - Mendapatkan referensi harga rumah sesuai dengan spesifikasi yang diinginkan
        - Membantu pengambilan keputusan berdasarkan data
        - Mengurangi risiko kesalahan pembelian rumah
    """)

    st.header("🔎Cara penggunaan")
    st.subheader("1. Cari referensi")
    st.write("""
        - Pilih menu **Dataset(CSV)**
        - Lihat referensi harga, dan spesifikasi rumah pada dataset yang tersedia
        - Pilih menu **Visualization**
        - Lihat grafik untuk melihat dan memahami spesifikasi rumah 
    """)
    st.subheader("2. Lakukan perhitungan")
    st.write("""
        - Pilih menu **Prediksi harga rumah**
        - Masukkan data kedalam kolom yang tersedia (Luas bangunan, Luas tanah, Jumlah kamar tidur, Jumlah kamar mandi, Jumlah garasi)
        - Dapatkan prediksi harga sesuai dengan data yang dimasukkan
    """)

    st.write("Aplikasi ini tidak hanya mempermudah Anda dalam memahami pasar properti, tetapi juga membantu dalam perencanaan keuangan dan pengambilan keputusan yang lebih baik. Mari mulai perjalanan Anda menuju rumah impian dengan data yang akurat dan prediksi yang cerdas!")

elif st.session_state["page"] == "Dataset":
    st.title("Dataset Properti")
    st.markdown("---")
    df = pd.read_csv("Data_rumah.csv")
    st.dataframe(df)

    st.write("Keterangan Kolom:")
    st.markdown("""
        - **LB (Luas Bangunan):** Area bangunan dalam meter persegi.
        - **LT (Luas Tanah):** Area tanah dalam meter persegi.
        - **KT (Kamar Tidur):** Jumlah kamar tidur.
        - **KM (Kamar Mandi):** Jumlah kamar mandi.
        - **GRS (Garasi):** Jumlah garasi.
    """)

elif st.session_state["page"] == "Visualization":
    st.title("Visualisasi Data")
    st.markdown("---")
    df = pd.read_csv("Data_rumah.csv")

    st.subheader("Grafik Luas Tanah")
    st.line_chart(df['LT'])

    st.subheader("Grafik Jumlah Kamar Tidur")
    st.line_chart(df['KT'])

    st.subheader("Grafik Harga Rumah")
    st.line_chart(df['HARGA'])

elif st.session_state["page"] == "Prediction":
    st.title("Prediksi Harga Rumah")
    st.markdown("---")
    building_area = st.number_input('Luas Bangunan (m²):', min_value=0, step=1)
    land_area = st.number_input('Luas Tanah (m²):', min_value=0, step=1)
    bedrooms = st.number_input('Jumlah Kamar Tidur:', min_value=0, step=1)
    bathrooms = st.number_input('Jumlah Kamar Mandi:', min_value=0, step=1)
    garages = st.number_input('Jumlah Garasi:', min_value=0, step=1)

    if st.button('Prediksi'):
        input_features = np.array([[building_area, land_area, bedrooms, bathrooms, garages]])
        input_scaled = scaler.transform(input_features)
        input_poly = poly.transform(input_scaled)
        prediction = np.exp(model_regresi.predict(input_poly)[0])
        st.success(f"Perkiraan Harga Rumah: Rp {prediction:,.2f}")

elif st.session_state["page"] == "About":
    st.title("Tentang Kami")
    st.markdown("---")
    st.write("Website ini dibuat oleh kelompok 4 untuk mempermudah estimasi harga rumah di Jakarta Selatan.")
    st.subheader("**Anggota Tim:**")
    st.write("- **Adi Alam Sami Aji** (233307091)")
    st.write("- **Muhammad Ikhsan Dea Aldiansyah** (233307108)")
    st.write("- **Umi Latifah Nurhaliza Agustin** (233307117)")
    st.write("- **Warda Imana** (233307118)")

elif st.session_state["page"] == "Methods":
    st.title("Metode dan Cara Kerja Model")
    st.markdown("---")
    st.header("Dataset")
    st.write("""
    Dataset yang digunakan dalam model ini terdiri dari 1010 data rumah di kawasan Tebet dan Jakarta Selatan dengan informasi:
    - Luas Bangunan (LB)
    - Luas Tanah (LT)
    - Jumlah Kamar Tidur (KT)
    - Jumlah Kamar Mandi (KM)
    - Jumlah Garasi (GRS)
    - Harga Rumah
    """)
    
    st.header("Preprocessing Data")
    st.write("""
    1. **Standardisasi Fitur**: Menggunakan StandardScaler untuk menormalkan semua fitur input ke skala yang sama
    2. **Transformasi Polinomial**: Menggunakan PolynomialFeatures untuk menangkap hubungan non-linear antar fitur
    3. **Logaritmik Transform**: Menerapkan transformasi log pada harga rumah untuk menstabilkan varians
    """)
    
    st.header("Model Machine Learning")
    st.write("""
    Model yang digunakan adalah **Linear Regression** dengan karakteristik:
    - Menggunakan 5 fitur input (LB, LT, KT, KM, GRS)
    - Transformasi polinomial degree 2 untuk menangkap hubungan non-linear
    - Cross-validation dengan 5 fold untuk validasi model
    """)
    
    st.header("Performa Model")
    st.write("""
    Berdasarkan evaluasi model, diperoleh:
    - R-squared Score: 0.8278 (82.78% varians dapat dijelaskan oleh model)
    - Mean Percentage Error: 23.98%
    - Cross-validation Score: 0.7658
    
    Ini menunjukkan bahwa model memiliki kemampuan prediksi yang cukup baik untuk estimasi harga rumah.
    """)
    
    st.header("Cara Kerja Prediksi")
    st.write("""
    1. User memasukkan data properti (LB, LT, KT, KM, GRS)
    2. Data dinormalisasi menggunakan StandardScaler
    3. Fitur ditransformasi menggunakan PolynomialFeatures
    4. Model melakukan prediksi harga dalam skala logaritmik
    5. Hasil prediksi ditransformasi balik ke nilai rupiah
    """)
    
    st.header("Batasan Model")
    st.write("""
    - Model hanya akurat untuk properti di kawasan Tebet dan Jakarta Selatan
    - Prediksi paling akurat untuk rumah dengan karakteristik yang mirip dengan data training
    - Faktor eksternal seperti lokasi spesifik, kondisi rumah, dan tren pasar tidak diperhitungkan
    - Margin error sekitar 24% perlu dipertimbangkan dalam interpretasi hasil
    """)

elif st.session_state["page"] == "Evaluation":
    st.title("Evaluasi Model Prediksi Harga Rumah")
    
    st.header("Metode Evaluasi")
    st.write("""
    Model dievaluasi menggunakan beberapa metrik untuk memastikan akurasi dan keandalannya:
    - **R-squared Score**: Mengukur seberapa baik varians data dapat dijelaskan oleh model.
    - **Mean Absolute Error (MAE)**: Rata-rata kesalahan absolut antara prediksi dan nilai sebenarnya.
    - **Root Mean Squared Error (RMSE)**: Akar dari rata-rata kesalahan kuadrat antara prediksi dan nilai sebenarnya.
    - **Mean Percentage Error**: Rata-rata kesalahan persentase antara prediksi dan nilai sebenarnya.
    """)
    
    st.header("Hasil Evaluasi")
    st.write("""
    Berdasarkan evaluasi model, diperoleh hasil sebagai berikut:
    - **R-squared Score**: 0.8278
    - **Mean Absolute Error (MAE)**: Rp 1.25
    - **Root Mean Squared Error (RMSE)**: Rp 1.35
    - **Mean Percentage Error**: 23.98%
    
    Hasil ini menunjukkan bahwa model memiliki kemampuan prediksi yang cukup baik untuk estimasi harga rumah.
    """)
st.markdown("---")
st.write("Oleh kelompok 4")