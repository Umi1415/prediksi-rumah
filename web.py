import pickle
import streamlit as st
import pandas as pd
import numpy as np
import scikit-learn

model_data = pickle.load(open('model_prediksi_harga_rumah.sav', 'rb'))
model_regresi = model_data['model']
scaler = model_data['scaler']
poly = model_data['poly']

st.title('Prediksi harga rumah di kawasan Tebet dan Jaksel dengan metode linear regression')

menu = st.sidebar.selectbox("Pilih Konten", ("Home", "Dataset (CSV)", "Visualization", "Prediksi Harga Rumah", "About"))

if menu == "Home":
    st.write("Website ini dirancang untuk membantu pengguna memprediksi harga rumah berdasarkan data spesifik properti yang mereka masukkan. Dengan memanfaatkan Linear Regression, aplikasi ini memberikan estimasi harga rumah secara akurat, berdasarkan dataset rumah di kawasan Tebet dan Jakarta Selatan.")
    image_path = "rumah.jpg"  
    st.image(image_path, caption="Gambar rumah", use_container_width=True)

    st.header("Fitur utama")
    st.header("Dataset(CSV)")
    st.write("Halaman ini digunakan untuk menampilkan dataset properti yang digunakan sebagai dasar untuk pelatihan model prediksi. Pengguna dapat melihat informasi seperti luas bangunan, luas tanah, jumlah kamar, dan harga rumah.")
    
    st.header("Visualization")
    st.write("Halaman ini berisikan visualisasi data yang menyajikan grafik dan tren properti")
    st.write("- Grafik luas tanah properti ") 
    st.write("- Grafik jumlah kamar tidur")
    st.write("- Grafik distribusi harga rumah.")
    
    st.header("Prediksi Harga Rumah")
    st.write("Halaman ini berisikan perhitungan dari harga rumah berdasarkan inputan dari pengguna")
    st.write("Keterangan:")
    st.write("- **LB (Luas Bangunan)**: Area of the building in square meters.")
    st.write("- **LT (Luas Tanah)**: Area of the land in square meters.")
    st.write("- **KT (Kamar Tidur)**: Number of bedrooms.")
    st.write("- **KM (Kamar Mandi)**: Number of bathrooms.")
    st.write("- **GRS (Garasi)**: Number of garages.")

elif menu == "Dataset (CSV)":
    st.header("Menampilkan Dataset (CSV)")
    df = pd.read_csv("Data_rumah.csv")
    st.dataframe(df)

    st.write("Keterangan:")
    st.write("- **LB (Luas Bangunan)**: Area of the building in square meters.")
    st.write("- **LT (Luas Tanah)**: Area of the land in square meters.")
    st.write("- **KT (Kamar Tidur)**: Number of bedrooms.")
    st.write("- **KM (Kamar Mandi)**: Number of bathrooms.")
    st.write("- **GRS (Garasi)**: Number of garages.")

elif menu == "Visualization":
    st.header("Menampilkan Grafik")
    df = pd.read_csv("Data_rumah.csv")

    st.write("Grafik Luas Tanah")
    chart_land_area = pd.DataFrame(df, columns=["LT"])  
    st.line_chart(chart_land_area)

    st.write("Grafik Jumlah Kamar")
    chart_bedrooms = pd.DataFrame(df, columns=["KT"])  
    st.line_chart(chart_bedrooms)

    st.write("Grafik Harga")
    chart_price = pd.DataFrame(df, columns=["HARGA"])  
    st.line_chart(chart_price)

elif menu == "Prediksi Harga Rumah":
    st.header("Masukkan Data untuk Prediksi harga rumah")

    building_area = st.number_input('Luas Bangunan (m²):', min_value=0)
    land_area = st.number_input('Luas Tanah (m²):', min_value=0)
    bedrooms = st.number_input('Jumlah Kamar Tidur:', min_value=0)
    bathrooms = st.number_input('Jumlah Kamar Mandi:', min_value=0)
    garages = st.number_input('Jumlah Garasi:', min_value=0)

    if st.button('Prediksi'):
            input_features = np.array([[building_area, land_area, bedrooms, bathrooms, garages]])
            
            input_scaled = scaler.transform(input_features)
            
            input_poly = poly.transform(input_scaled)
            
            prediction = np.exp(model_regresi.predict(input_poly)[0])
            
            harga_rumah_formatted = f"Harga Rumah: Rp {prediction:,.2f}"
            st.success(harga_rumah_formatted)

elif menu == "About":
    st.write("Website ini dibuat oleh kelompok 4 yang beranggotakan:")
    st.write("- **Adi Alam Sami Aji** (233307091)") 
    st.write("- **Muhammad Ikhsan Dea Aldiansyah** (233307108)")
    st.write("- **Umi Latifah Nurhaliza Agustin** (233307117)")
    st.write("- **Warda Imana** (233307118)")
