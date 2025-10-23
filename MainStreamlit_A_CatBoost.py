#STREAMLIT

import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open("best_gradient_boosting.pkl", "rb") as f:
        model = pickle.load(f)
    return model

rf_model = load_model()

st.set_page_config(page_title="Earthquake Alert Prediction", layout="centered")
st.title("🌋 Earthquake Alert Prediction")

st.markdown("""
Aplikasi ini memprediksi **warna peringatan (alert)** dari suatu gempa bumi 
menggunakan model Machine Learning terbaik: **Random Forest Classifier**.
""")

st.markdown("""
### Penjelasan Fitur Input
| Nama Fitur | Arti | Penjelasan Sederhana |
|-------------|------|----------------------|
| **Magnitude** | Kekuatan gempa (skala Richter) | Semakin besar nilainya, semakin kuat gempa. |
| **Depth (km)** | Kedalaman pusat gempa | Gempa dangkal (<70 km) biasanya lebih terasa di permukaan. |
| **CDI** | Community Determined Intensity | Seberapa kuat guncangan dirasakan oleh masyarakat (1–10). |
| **MMI** | Modified Mercalli Intensity | Skala intensitas fisik gempa terhadap bangunan dan lingkungan (1–10). |
| **SIG** | Significance | Nilai positif = signifikan, negatif = ringan. |
""")

st.markdown("---")

alert_labels = {
    0: "🟢 Green – Dampak sangat kecil atau tidak signifikan",
    1: "🟡 Yellow – Dampak sedang, potensi kerusakan kecil",
    2: "🟠 Orange – Dampak signifikan, kemungkinan kerusakan sedang-besar",
    3: "🔴 Red – Dampak parah, kerusakan besar dan potensi korban tinggi"
}

alert_colors = {
    0: "#00C853",  # green
    1: "#FFD600",  # yellow
    2: "#FF6D00",  # orange
    3: "#D50000",  # red
}

def highlight_alert(model_name, value):
    label = alert_labels.get(value, "Tidak diketahui")
    color = alert_colors.get(value, "#FFFFFF")
    html = f"""
    <div style="background-color:{color}; padding:15px; border-radius:10px; text-align:center; color:white; font-weight:bold; font-size:18px;">
        {model_name} Prediction: {label}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

st.subheader("🧮 Masukkan Data Gempa")

col1, col2 = st.columns(2)
with col1:
    magnitude = st.number_input("Magnitude", 0.0, 10.0, 6.5)
    cdi = st.number_input("CDI", 0.0, 10.0, 5.0)
    sig = st.number_input("SIG", -1000.0, 1000.0, 0.0)
with col2:
    depth = st.number_input("Depth (km)", 0.0, 700.0, 20.0)
    mmi = st.number_input("MMI", 0.0, 10.0, 5.0)

input_data = pd.DataFrame([[magnitude, depth, cdi, mmi, sig]],
                          columns=["magnitude", "depth", "cdi", "mmi", "sig"])

if st.button("🔍 Prediksi Alert"):
    pred = rf_model.predict(input_data)[0]
    st.success("✅ Prediksi berhasil dilakukan!")

    st.markdown("### 🌐 Hasil Prediksi")
    highlight_alert("Random Forest", pred)

    st.markdown("---")
    st.markdown("### 🧭 Arti Warna Alert")
    alert_df = pd.DataFrame({
        "Kode": list(alert_labels.keys()),
        "Arti": list(alert_labels.values())
    })
    st.table(alert_df)
