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

# Setup NLTK data path
# Coba 3 pendekatan berbeda untuk NLTK data
nltk_data_paths = [
    os.path.join(os.getcwd(), 'nltk_data'),
    '/app/nltk_data',  # Khusus untuk Streamlit Cloud
    '/home/appuser/nltk_data'  # Khusus untuk Streamlit Cloud
]

for path in nltk_data_paths:
    if path not in nltk.data.path:
        nltk.data.path.append(path)

# Download NLTK resources - dipanggil setiap kali aplikasi dijalankan
def download_nltk_resources():
    try:
        # Pastikan seluruh data NLTK yang dibutuhkan diunduh
        resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f'corpora/{resource}')
                st.sidebar.success(f"✅ Resource '{resource}' sudah tersedia")
            except LookupError:
                st.sidebar.info(f"⏳ Mengunduh resource '{resource}'...")
                nltk.download(resource, quiet=False)
                st.sidebar.success(f"✅ Resource '{resource}' berhasil diunduh")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Error downloading NLTK resources: {e}")
        return False

# Fungsi untuk mengunduh file dari Google Drive
@st.cache_resource
def download_models():
    # Direktori untuk menyimpan model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # URL Google Drive untuk setiap model dan vectorizer
    file_ids = {
        'vectorizer.pkl': '16APGmUhdSNXIN4wXqEtRr_FL-f-aLR7J',
        'svc_model.pkl': '1qobmO92v-0FZaAJkUuYgtMjtVq02mpD3',
        'lr_model.pkl': '1aNfCuT_kzBSt6fHUrBmox2aB-tPCg2FY',
        'nb_model.pkl': '1cZ3Nokx4DeWpoulLQsrz5VwvlyXXaJrN',
        'knn_model.pkl': '15i9hv842HAUzZds-JbT133WRSuv7_ipm'
    }
    
    for filename, file_id in file_ids.items():
        output_path = f'models/{filename}'
        if not os.path.exists(output_path):
            try:
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, output_path, quiet=False)
                st.sidebar.success(f"✅ Downloaded {filename}")
            except Exception as e:
                st.sidebar.error(f"❌ Error downloading {filename}: {e}")
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

# Fungsi preprocessing untuk teks - PENTING: HARUS SAMA DENGAN YANG DI NOTEBOOK
def preprocess_text(text):
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
    return text.strip()

# Fungsi untuk menghapus stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Fungsi untuk lemmatisasi
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Alternatif: Fungsi stemming jika lemmatisasi bermasalah
def stem_text(text):
    stemmer = PorterStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Fungsi preprocessing lengkap seperti di notebook Jupyter
def full_preprocessing(text, use_stemming=False):
    # 1. Preprocess teks dasar
    text = preprocess_text(text)
    
    # 2. Hapus stopwords
    text = remove_stopwords(text)
    
    # 3. Lakukan lemmatisasi atau stemming
    if use_stemming:
        text = stem_text(text)
    else:
        text = lemmatize_text(text)
    
    return text

# Fungsi untuk melakukan prediksi
def predict_sentiment(text, model_name, models, use_stemming=False):
    # Preprocess teks
    clean_text = full_preprocessing(text, use_stemming)
    
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
    .stButton>button {
        background-color: #1DA1F2;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="sub-header">Analisis sentimen teks menggunakan Machine Learning</p>', unsafe_allow_html=True)
    
    # Sidebar untuk pengaturan
    st.sidebar.title("Pengaturan")
    
    # Download NLTK resources
    with st.sidebar.expander("Status NLTK Resources", expanded=False):
        nltk_resources_downloaded = download_nltk_resources()
    
    # Unduh model
    with st.sidebar.expander("Status Model", expanded=False):
        models_downloaded = download_models()
    
    if not models_downloaded:
        st.warning("❌ Ada masalah dalam mengunduh model. Aplikasi mungkin tidak berfungsi dengan baik.")
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("❌ Gagal memuat model. Silakan coba lagi nanti.")
        return
    
    # Pilihan model klasifikasi
    selected_model = st.sidebar.selectbox(
        "Pilih Model Klasifikasi:",
        ["SVC", "Logistic Regression", "Naive Bayes", "KNN"]
    )
    
    # Opsi preprocessing
    with st.sidebar.expander("Opsi Preprocessing", expanded=False):
        use_stemming = st.checkbox("Gunakan Stemming (bukan Lemmatization)", value=False)
        show_debug = st.checkbox("Tampilkan detail preprocessing", value=True)
    
    # Tab untuk navigasi
    tab1, tab2, tab3 = st.tabs(["Prediksi Sentimen", "Contoh Input", "Tentang"])
    
    with tab1:
        st.header("Prediksi Sentimen")
        
        # Input teks untuk analisis
        text_input = st.text_area(
            "Masukkan teks untuk analisis sentimen:",
            "This movie was fantastic! The acting was top-notch and the story was very engaging."
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            predict_button = st.button('Prediksi Sentimen', key='predict_button')
        
        if predict_button:
            try:
                with st.spinner('Memproses...'):
                    # Tahapan preprocessing untuk visibilitas
                    if show_debug:
                        text = text_input
                        st.subheader("Tahapan Preprocessing:")
                        
                        # Step 1: Preprocessing dasar
                        preprocessed = preprocess_text(text)
                        st.info(f"**Setelah preprocessing dasar:** '{preprocessed}'")
                        
                        # Step 2: Hapus stopwords
                        no_stopwords = remove_stopwords(preprocessed)
                        st.info(f"**Setelah penghapusan stopwords:** '{no_stopwords}'")
                        
                        # Step 3: Lemmatisasi/Stemming
                        if use_stemming:
                            final = stem_text(no_stopwords)
                            st.info(f"**Setelah stemming:** '{final}'")
                        else:
                            final = lemmatize_text(no_stopwords)
                            st.info(f"**Setelah lemmatisasi:** '{final}'")
                    
                    # Melakukan prediksi
                    prediction, clean_text, prob_df = predict_sentiment(text_input, selected_model, models, use_stemming)
                    
                    # Menampilkan hasil
                    sentiment = "Positif" if prediction == 1 else "Negatif"
                    
                    # Warna sesuai sentimen
                    if sentiment == "Positif":
                        st.success(f'Prediksi Sentimen: {sentiment}')
                    else:
                        st.error(f'Prediksi Sentimen: {sentiment}')
                    
                    # Menampilkan teks yang sudah dibersihkan
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
                st.error(f"❌ Terjadi kesalahan: {str(e)}")
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
                            prediction, clean_text, prob_df = predict_sentiment(example, selected_model, models, use_stemming)
                            sentiment = "Positif" if prediction == 1 else "Negatif"
                            if sentiment == "Positif":
                                st.success(f'Prediksi Sentimen: {sentiment}')
                            else:
                                st.error(f'Prediksi Sentimen: {sentiment}')
                            
                            if show_debug:
                                st.info(f'Teks setelah preprocessing: {clean_text}')
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan: {str(e)}")
        
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
                            prediction, clean_text, prob_df = predict_sentiment(example, selected_model, models, use_stemming)
                            sentiment = "Positif" if prediction == 1 else "Negatif"
                            if sentiment == "Positif":
                                st.success(f'Prediksi Sentimen: {sentiment}')
                            else:
                                st.error(f'Prediksi Sentimen: {sentiment}')
                            
                            if show_debug:
                                st.info(f'Teks setelah preprocessing: {clean_text}')
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan: {str(e)}")
    
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
        3. **Lemmatisasi/Stemming**: Mengubah kata ke bentuk dasarnya
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
        
        # Debugging info
        with st.expander("Informasi Debug (untuk pengembang)", expanded=False):
            st.write("### NLTK Data Path")
            for path in nltk.data.path:
                st.code(path)
            
            st.write("### Environment Info")
            st.code(f"Current Working Directory: {os.getcwd()}")
            
            st.write("### Vectorizer Info")
            try:
                vectorizer = models['vectorizer']
                st.code(f"Jumlah fitur: {len(vectorizer.get_feature_names_out())}")
                st.code(f"Contoh fitur: {vectorizer.get_feature_names_out()[:10]}")
            except:
                st.warning("Vectorizer tidak tersedia")

if __name__ == "__main__":
    main()
