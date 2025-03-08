import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Fungsi untuk memuat model dan vectorizer
def load_model_and_vectorizer():
    with open("model.pkl", "rb") as model_file:  # Ganti dengan path model Anda
        models = pickle.load(model_file)  # Misalnya dict { "SVC": model_svc, "LR": model_lr, "NB": model_nb, "KNN": model_knn }
    with open("vectorizer.pkl", "rb") as vectorizer_file:  # Ganti dengan path vectorizer Anda
        vectorizer = pickle.load(vectorizer_file)
    return models, vectorizer

# Fungsi untuk prediksi sentimen
def predict_sentiment(text, vectorizer, model):
    text_vectorized = vectorizer.transform([text])  # Transformasi teks menjadi vektor fitur
    prediction = model.predict(text_vectorized)  # Prediksi menggunakan model
    confidence = np.max(model.predict_proba(text_vectorized))  # Prediksi confidence
    sentiment = "Positif" if prediction == 0 else "Negatif"
    return sentiment, confidence

# Fungsi utama untuk Streamlit
def main():
    # Set page config
    st.set_page_config(page_title="Analisis Sentimen Tweet", page_icon="🐦", layout="wide")
    
    # Tampilan awal
    st.title("Aplikasi Analisis Sentimen Tweet")
    st.write("Aplikasi ini menggunakan model machine learning untuk menganalisis sentimen dari teks yang Anda masukkan.")
    
    # Load model dan vectorizer
    models, vectorizer = load_model_and_vectorizer()

    # Sidebar untuk navigasi
    page = st.sidebar.selectbox("Navigasi", ["Beranda", "Prediksi Sentimen"])

    if page == "Beranda":
        st.header("Selamat Datang di Aplikasi Analisis Sentimen")
        st.write("""
            Aplikasi ini membantu Anda menganalisis sentimen dari teks menggunakan beberapa model machine learning.
            
            ### Cara Penggunaan:
            1. **Prediksi Sentimen**: Masukkan teks Anda untuk melihat prediksi sentimennya menggunakan model pilihan.
            """)

    elif page == "Prediksi Sentimen":
        st.header("Prediksi Sentimen Tweet")
        st.write("Masukkan teks dan lihat prediksi sentimennya.")

        # Pilih model untuk analisis
        model_option = st.selectbox("Pilih Model", ["SVC", "Logistic Regression", "Naive Bayes", "KNN"])

        # Input teks dari pengguna
        text_input = st.text_area("Masukkan teks tweet:", "", height=150)

        # Button untuk melakukan prediksi
        if st.button("Analisis Sentimen"):
            if text_input:
                with st.spinner("Menganalisis sentimen..."):
                    # Pilih model yang dipilih dari dropdown
                    model = models[model_option]
                    sentiment, confidence = predict_sentiment(text_input, vectorizer, model)
                
                # Menampilkan hasil
                sentiment_color = "green" if sentiment == "Positif" else "red"
                st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {sentiment_color}10; border: 1px solid {sentiment_color};">
                        <h3 style="color: {sentiment_color}; margin-top: 0;">Prediksi Sentimen: {sentiment}</h3>
                        <p>Tingkat kepercayaan: {confidence:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.error("Silakan masukkan teks terlebih dahulu.")

if __name__ == "__main__":
    main()
