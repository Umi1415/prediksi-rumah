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
