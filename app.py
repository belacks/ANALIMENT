import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import joblib
import gdown
import os

# Judul aplikasi dengan sedikit styling
st.set_page_config(
    page_title="Analisis Sentimen Tweet",
    page_icon="🐦",
    layout="wide"
)

# Download NLTK resources - pindahkan fungsi ini ke awal aplikasi
# dan panggil secara langsung (bukan sebagai cache_resource)
def download_nltk_resources():
    try:
        # Tambahkan verbose=True untuk melihat proses download
        nltk.download('punkt', quiet=False)
        nltk.download('stopwords', quiet=False)
        # Pastikan wordnet diunduh dengan benar
        nltk.download('wordnet', quiet=False)
        # Jika ada resource lain yang diperlukan
        nltk.download('omw-1.4', quiet=False)  # Open Multilingual Wordnet
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")
        return False

# Panggil fungsi secara langsung saat aplikasi dimulai
nltk_resources_downloaded = download_nltk_resources()
if not nltk_resources_downloaded:
    st.error("⚠️ Gagal mengunduh resource NLTK yang diperlukan. Aplikasi mungkin tidak berfungsi dengan baik.")

# Fungsi untuk mengunduh file dari Google Drive
@st.cache_resource
def download_models():
    # Direktori untuk menyimpan model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # URL Google Drive untuk setiap model dan vectorizer
    file_ids = {
        'vectorizer.pkl': '16APGmUhdSNXIN4wXqEtRr_FL-f-aLR7J',
        'svc_model.pkl': '1ThZvVawKukcCAuPnE278IxTz86iuflmo',
        'lr_model.pkl': '1uMjCePvyPRUsnoq8Wkv65p8n551SIt0S',
        'nb_model.pkl': '1IodoKbn1eMSI0xlRnkYpY0msCDoGAEv6',
        'knn_model.pkl': '15SffKdmGpdPFhIcFr-ceFjZbEPIDq7aX'
    }
    
    for filename, file_id in file_ids.items():
        output_path = f'models/{filename}'
        if not os.path.exists(output_path):
            try:
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, output_path, quiet=False)
                st.success(f"Downloaded {filename}")
            except Exception as e:
                st.error(f"Error downloading {filename}: {e}")
                return False
    
    return True

# Load all models
@st.cache_resource
def load_models():
    models = {}
    try:
        # Load vectorizer
        models['vectorizer'] = joblib.load('models/vectorizer.pkl')
        
        # Load classification models
        models['SVC'] = joblib.load('models/svc_model.pkl')
        models['Logistic Regression'] = joblib.load('models/lr_model.pkl')
        models['Naive Bayes'] = joblib.load('models/nb_model.pkl')
        models['KNN'] = joblib.load('models/knn_model.pkl')
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Verifikasi bahwa stopwords dan lemmatizer dapat diakses
try:
    # Coba inisialisasi lemmatizer untuk memastikan wordnet tersedia
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("testing")
    
    # Coba akses stopwords untuk memastikan tersedia
    stop_words = set(stopwords.words('english'))
    st.success("✅ NLTK resources berhasil dimuat")
except Exception as e:
    st.error(f"⚠️ Error saat mengakses NLTK resources: {e}")

# Fungsi preprocessing untuk teks
def preprocess_text(text):
    # Pembersihan teks secara bertahap
    # Convert to string if not already
    text = str(text)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 3. Remove hashtags (#topic)
    text = re.sub(r'#\w+', '', text)
    
    # 4. Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # 5. Remove non-alphabetic characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 6. Strip extra whitespace
    text = text.strip()
    
    return text

# Fungsi untuk menghapus stopwords - menggunakan variabel global 
# yang sudah diinisialisasi di awal
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Fungsi untuk lemmatisasi - menggunakan variabel global
# yang sudah diinisialisasi di awal
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Fungsi untuk melakukan prediksi dengan preprocessing yang sesuai
def predict_sentiment(text, model_name, models):
    # 1. Preprocess teks - langkah dasar
    clean_text = preprocess_text(text)
    st.write(f"Setelah preprocessing dasar: '{clean_text}'")
    
    # 2. Hapus stopwords
    clean_text = remove_stopwords(clean_text)
    st.write(f"Setelah penghapusan stopwords: '{clean_text}'")
    
    # 3. Lakukan lemmatisasi
    clean_text = lemmatize_text(clean_text)
    st.write(f"Setelah lemmatisasi: '{clean_text}'")
    
    # Vectorize teks
    vectorizer = models['vectorizer']
    text_vectorized = vectorizer.transform([clean_text])
    
    # Prediksi dengan model yang dipilih
    model = models[model_name]
    prediction = model.predict(text_vectorized)[0]
    
    # Mencoba mendapatkan probabilitas jika model mendukung
    try:
        probabilities = model.predict_proba(text_vectorized)[0]
        prob_df = pd.DataFrame({
            'Sentimen': ['Negatif', 'Positif'],
            'Probabilitas': [probabilities[0], probabilities[1]]
        })
    except:
        prob_df = None
    
    return prediction, clean_text, prob_df

# Main App
def main():
    # App header dengan styling
    st.title('🐦 Analisis Sentimen Tweet')
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1DA1F2;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #657786;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="sub-header">Analisis sentimen teks menggunakan Machine Learning</p>', unsafe_allow_html=True)
    
    # Unduh model
    models_downloaded = download_models()
    
    if not models_downloaded:
        st.warning("Ada masalah dalam mengunduh model. Aplikasi mungkin tidak berfungsi dengan baik.")
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("Gagal memuat model. Silakan coba lagi nanti.")
        return
    
    # Sidebar untuk memilih model
    st.sidebar.title("Pengaturan")
    selected_model = st.sidebar.selectbox(
        "Pilih Model Klasifikasi:",
        ["SVC", "Logistic Regression", "Naive Bayes", "KNN"]
    )
    
    # Tambahkan checkbox untuk menampilkan debug info
    show_debug = st.sidebar.checkbox("Tampilkan detail preprocessing", value=True)
    
    # Tab untuk navigasi
    tab1, tab2, tab3 = st.tabs(["Prediksi Sentimen", "Contoh Input", "Tentang"])
    
    with tab1:
        st.header("Prediksi Sentimen")
        
        # Input teks untuk analisis
        text_input = st.text_area(
            "Masukkan teks untuk analisis sentimen:",
            "This movie was fantastic! The acting was top-notch and the story was very engaging."
        )
        
        if st.button('Prediksi Sentimen', key='predict_button'):
            try:
                with st.spinner('Memproses...'):
                    # Melakukan prediksi
                    prediction, clean_text, prob_df = predict_sentiment(text_input, selected_model, models)
                    
                    # Menampilkan hasil
                    sentiment = "Positif" if prediction == 1 else "Negatif"
                    
                    # Warna sesuai sentimen
                    if sentiment == "Positif":
                        st.success(f'Prediksi Sentimen: {sentiment}')
                    else:
                        st.error(f'Prediksi Sentimen: {sentiment}')
                    
                    # Menampilkan teks yang sudah dibersihkan
                    if show_debug:
                        st.subheader('Teks setelah preprocessing:')
                        st.info(clean_text)
                    
                    # Menampilkan probabilitas jika tersedia
                    if prob_df is not None:
                        st.subheader("Probabilitas:")
                        
                        # Split columns for visualization
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.dataframe(prob_df)
                        
                        with col2:
                            # Bar chart probabilitas
                            chart_data = prob_df.set_index('Sentimen')
                            st.bar_chart(chart_data)
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
                st.error("Jika masalah masih terjadi, coba muat ulang halaman web")
    
    with tab2:
        st.header("Contoh Input")
        
        # Example cards with different sentiment examples
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Contoh Positif")
            examples_positive = [
                "This movie was fantastic! The acting was top-notch and the story was very engaging.",
                "I absolutely loved the service. The staff was friendly and attentive.",
                "The product exceeded my expectations. It's worth every penny!",
                "Had a great time at the concert. The band's energy was infectious!"
            ]
            
            for i, example in enumerate(examples_positive):
                st.info(example)
                if st.button(f'Prediksi Contoh Positif {i+1}', key=f'pos_example_{i}'):
                    try:
                        with st.spinner('Memproses...'):
                            prediction, clean_text, prob_df = predict_sentiment(example, selected_model, models)
                            sentiment = "Positif" if prediction == 1 else "Negatif"
                            if sentiment == "Positif":
                                st.success(f'Prediksi Sentimen: {sentiment}')
                            else:
                                st.error(f'Prediksi Sentimen: {sentiment}')
                            
                            if show_debug:
                                st.info(f'Teks setelah preprocessing: {clean_text}')
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {str(e)}")
        
        with col2:
            st.subheader("Contoh Negatif")
            examples_negative = [
                "I was disappointed with the plot. It felt like a waste of time.",
                "The customer service was terrible. I waited for hours with no resolution.",
                "Don't buy this product. It broke after one use.",
                "The restaurant was overpriced and the food was bland."
            ]
            
            for i, example in enumerate(examples_negative):
                st.error(example)
                if st.button(f'Prediksi Contoh Negatif {i+1}', key=f'neg_example_{i}'):
                    try:
                        with st.spinner('Memproses...'):
                            prediction, clean_text, prob_df = predict_sentiment(example, selected_model, models)
                            sentiment = "Positif" if prediction == 1 else "Negatif"
                            if sentiment == "Positif":
                                st.success(f'Prediksi Sentimen: {sentiment}')
                            else:
                                st.error(f'Prediksi Sentimen: {sentiment}')
                            
                            if show_debug:
                                st.info(f'Teks setelah preprocessing: {clean_text}')
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {str(e)}")
    
    with tab3:
        st.header("Tentang Aplikasi")
        st.write("""
        ### Analisis Sentimen Tweet
        
        Aplikasi ini menggunakan model machine learning untuk memprediksi sentimen dari teks yang dimasukkan.
        
        #### Proses Analisis Sentimen:
        1. **Preprocessing**: 
           - Konversi ke lowercase
           - Hapus mention (@username)
           - Hapus hashtag (#topic)
           - Hapus URL
           - Hapus karakter non-alfabet
        2. **Penghapusan Stopwords**: Menghapus kata-kata umum yang tidak memberikan informasi sentimen
        3. **Lemmatisasi**: Mengubah kata ke bentuk dasarnya
        4. **Vektorisasi**: Teks diubah menjadi vektor fitur menggunakan TF-IDF
        5. **Klasifikasi**: Model machine learning memprediksi sentimen (positif atau negatif)
        
        #### Model yang Tersedia:
        - **SVC**: Support Vector Classifier, efektif untuk dimensi tinggi
        - **Logistic Regression**: Model klasifikasi linear yang sederhana dan cepat
        - **Naive Bayes**: Algoritma probabilistik berdasarkan teorema Bayes
        - **KNN**: K-Nearest Neighbors, algoritma berbasis jarak
        
        #### Dataset:
        Model dilatih menggunakan dataset tweet yang dilabeli dengan sentimen positif (1) dan negatif (0).
        """)
        
        # Informasi tentang developer
        st.subheader("Developer")
        st.write("Dibuat sebagai bagian dari tugas analisis sentimen.")

if __name__ == "__main__":
    main()
