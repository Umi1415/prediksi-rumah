import pickle
import streamlit as st
import pandas as pd
import numpy as np
import sklearn

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
with st.sidebar:
    st.markdown("### MENU")
    if st.button("🏡 Home\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "Home"
    if st.button("📈 Dataset (CSV)\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "Dataset"
    if st.button("🔄 Visualization\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "Visualization"
    if st.button("🏢 Prediksi Harga Rumah\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "Prediction"
    if st.button("ℹ️ About\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2002"):
        st.session_state["page"] = "About"
    if st.button("🔬 Metode & Model\u2003\u2003\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "Methods"
    if st.button("📊 Evaluation\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003"):
        st.session_state["page"] = "Evaluation"

# Display page content based on selected page
if st.session_state["page"] == "Home":
    st.title("Prediksi Harga Rumah")
    st.header("Selamat Datang di Aplikasi Prediksi Harga Rumah")
    st.write("Website ini dirancang untuk membantu pengguna memprediksi harga rumah di kawasan Tebet dan Jakarta Selatan.")

    image_path = "rumah.jpg"
    st.image(image_path, caption="Gambar rumah", use_container_width=True)

    st.subheader("Fitur Utama")
    st.markdown("""
        - **Dataset (CSV):** Menampilkan dataset properti yang digunakan sebagai pelatihan model.
        - **Visualization:** Grafik dan tren properti untuk analisis lebih lanjut.
        - **Prediksi Harga Rumah:** Perhitungan harga rumah berdasarkan input data spesifik pengguna.
        - **About:** Informasi tentang aplikasi ini dan pembuatnya.
    """)

elif st.session_state["page"] == "Dataset":
    st.title("Dataset Properti")
    st.write("Menampilkan dataset yang digunakan untuk melatih model prediksi.")

    df = pd.read_csv("Data_rumah.csv")
    st.dataframe(df)

    st.write("Keterangan Kolom:")
    st.write("- **LB (Luas Bangunan):** Area bangunan dalam meter persegi.")
    st.write("- **LT (Luas Tanah):** Area tanah dalam meter persegi.")
    st.write("- **KT (Kamar Tidur):** Jumlah kamar tidur.")
    st.write("- **KM (Kamar Mandi):** Jumlah kamar mandi.")
    st.write("- **GRS (Garasi):** Jumlah garasi.")

elif st.session_state["page"] == "Visualization":
    st.title("Visualisasi Data")
    df = pd.read_csv("Data_rumah.csv")

    st.subheader("Grafik Luas Tanah")
    st.line_chart(df['LT'])

    st.subheader("Grafik Jumlah Kamar Tidur")
    st.line_chart(df['KT'])

    st.subheader("Grafik Harga Rumah")
    st.line_chart(df['HARGA'])

elif st.session_state["page"] == "Prediction":
    st.title("Prediksi Harga Rumah")

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
    st.write("Website ini dibuat oleh kelompok 4 untuk mempermudah estimasi harga rumah di Jakarta Selatan.")
    st.write("**Anggota Tim:**")
    st.write("- Adi Alam Sami Aji (233307091)")
    st.write("- Muhammad Ikhsan Dea Aldiansyah (233307108)")
    st.write("- Umi Latifah Nurhaliza Agustin (233307117)")
    st.write("- Warda Imana (233307118)")

elif st.session_state["page"] == "Methods":
    st.title("Metode dan Cara Kerja Model")
    
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
