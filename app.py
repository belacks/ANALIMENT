import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import joblib
import gdown
import os

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis of IMDB Movie Ratings",
    page_icon="📊",
    layout="wide"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab')

download_nltk_resources()

# Fungsi preprocessing teks
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Stopword removal
@st.cache_resource
def get_stopwords():
    return set(stopwords.words('english'))

stop_words = get_stopwords()

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Fungsi untuk preprocessing teks secara keseluruhan
def preprocess_text(text):
    cleaned = clean_text(text)
    cleaned = remove_stopwords(cleaned)
    cleaned = lemmatize_text(cleaned)
    return cleaned

# Load model dan vectorizer dari Google Drive
@st.cache_resource
def download_models():
    # URL files from Google Drive (Ganti dengan URL Google Drive Anda)
    model_urls = {
        'svm_model.joblib': 'https://drive.google.com/uc?export=download&id=1FYysyTuEd_iEbwGZKMncwjklv-H9LVZe',
        'nb_model.joblib': 'https://drive.google.com/uc?export=download&id=1reHMrMmy7ZIWtIRkm2TCQ6CAV7HbwIbV',
        'lr_model.joblib': 'https://drive.google.com/uc?export=download&id=1bADkbh-bLmUyenmFQVDHhGfDVxRmPFok',
        'knn_model.joblib': 'https://drive.google.com/uc?export=download&id=1IkPfxW7pq2nxR6ZUtSECcTXDuiPEHx8i',
        'tfidf_vectorizer.joblib': 'https://drive.google.com/uc?export=download&id=1ZQnFijrsIPik6ZTaZgKxHjuWCVuvf4ez'
    }
    
    # Buat folder models jika belum ada
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Download semua model
    for file_name, url in model_urls.items():
        output_path = os.path.join('models', file_name)
        if not os.path.exists(output_path):
            gdown.download(url, output_path, quiet=True)

# Load models dan vectorizer
@st.cache_resource
def load_models():
    download_models()
    
    # Load models dari drive
    svm_model = joblib.load('models/svm_model.joblib')
    nb_model = joblib.load('models/nb_model.joblib')
    lr_model = joblib.load('models/lr_model.joblib')
    knn_model = joblib.load('models/knn_model.joblib')
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    
    return svm_model, nb_model, lr_model, knn_model, tfidf_vectorizer

# Load models
svm_model, nb_model, lr_model, knn_model, tfidf_vectorizer = load_models()

# Fungsi untuk prediksi dengan model
def predict_sentiment(text, model, vectorizer):
    # Preprocess text
    cleaned = preprocess_text(text)
    cleaned = remove_stopwords(cleaned)
    cleaned = lemmatize_text(cleaned)
    
    # Vectorize
    text_vector = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    # Get sentiment label and confidence
    sentiment = 'Positif' if prediction == 1 else 'Negatif'
    confidence = probability[1] if sentiment == 'Positif' else probability[0]
    
    return sentiment, confidence

# Header aplikasi
st.title("Sentiment Analysis of IMDB Movie Reviews")
st.write("""
Aplikasi ini melakukan analisis sentimen pada teks menggunakan empat model machine learning: 
SVM, Naive Bayes, Logistic Regression, dan KNN.
""")

# Sidebar dengan informasi
with st.sidebar:
    st.header("Tentang Aplikasi")
    st.write("""
    Aplikasi ini menggunakan model machine learning untuk menganalisis sentimen dari teks.
    Data yang digunakan untuk pelatihan adalah dataset tweets dengan label sentimen.
    """)
    
    st.subheader("Langkah Preprocessing")
    st.write("""
    1. Case folding (lowercase)
    2. Menghapus URL, mention, dan hashtag
    3. Menghapus tanda baca dan angka
    4. Menghapus stopwords
    5. Lemmatization
    """)
    
    st.subheader("Model yang Digunakan")
    st.write("1. Support Vector Machine (SVM)")
    st.write("2. Naive Bayes")
    st.write("3. Logistic Regression")
    st.write("4. K-Nearest Neighbors (KNN)")

# Formulir input
st.header("Masukkan Teks untuk Analisis")

# Text area untuk input user
user_input = st.text_area("Masukkan teks yang akan dianalisis:", 
                         "This movie was fantastic! The acting was top-notch and the story was very engaging.",
                         height=150)

# Tombol untuk submit
if st.button("Analisis Sentimen"):
    # Tampilkan spinner saat memproses
    with st.spinner("Menganalisis sentimen..."):
        # Lakukan prediksi dengan semua model
        svm_sentiment, svm_conf = predict_sentiment(user_input, svm_model, tfidf_vectorizer)
        nb_sentiment, nb_conf = predict_sentiment(user_input, nb_model, tfidf_vectorizer)
        lr_sentiment, lr_conf = predict_sentiment(user_input, lr_model, tfidf_vectorizer)
        knn_sentiment, knn_conf = predict_sentiment(user_input, knn_model, tfidf_vectorizer)
        
        # Tampilkan teks yang sudah dipreprocess
        st.subheader("Teks Setelah Preprocessing:")
        preprocessed_text = preprocess_text(user_input)
        st.text(preprocessed_text)
        
        # Tampilkan hasil prediksi dalam bentuk tabel
        st.subheader("Hasil Prediksi:")
        
        # Buat 4 kolom untuk hasil dari 4 model
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("SVM", svm_sentiment, f"{svm_conf:.2%}")
            
        with col2:
            st.metric("Naive Bayes", nb_sentiment, f"{nb_conf:.2%}")
            
        with col3:
            st.metric("Logistic Regression", lr_sentiment, f"{lr_conf:.2%}")
            
        with col4:
            st.metric("KNN", knn_sentiment, f"{knn_conf:.2%}")
        
        # Visualisasi hasil
        st.subheader("Visualisasi Confidence:")
        
        # Buat data untuk bar chart
        models = ['SVM', 'Naive Bayes', 'Logistic Regression', 'KNN']
        confidence_values = [svm_conf, nb_conf, lr_conf, knn_conf]
        sentiments = [svm_sentiment, nb_sentiment, lr_sentiment, knn_sentiment]
        
        # Buat DataFrame untuk visualisasi
        viz_data = pd.DataFrame({
            'Model': models,
            'Confidence': confidence_values,
            'Sentiment': sentiments
        })
        
        # Atur warna berdasarkan sentimen
        colors = ['green' if s == 'Positif' else 'red' for s in sentiments]
        
        # Plot bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(viz_data['Model'], viz_data['Confidence'], color=colors)
        
        # Tambahkan label pada bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f"{sentiments[i]}\n{confidence_values[i]:.2%}",
                    ha='center', va='bottom', rotation=0)
        
        ax.set_ylim(0, 1.15)
        ax.set_title('Confidence Level per Model')
        ax.set_ylabel('Confidence')
        ax.set_xlabel('Model')
        
        # Tampilkan plot
        st.pyplot(fig)
        
        # Generate WordCloud dari input user
        st.subheader("WordCloud dari Teks:")
        
        # Buat WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             stopwords=STOPWORDS, max_words=100).generate(preprocessed_text)
        
        # Plot WordCloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        # Tampilkan WordCloud
        st.pyplot(fig)

# Contoh penggunaan
st.header("Contoh Penggunaan")

with st.expander("Contoh Teks dengan Sentimen Positif"):
    st.write("""
    **Input:** This movie was fantastic! The acting was top-notch and the story was very engaging.
    
    **Output:** Prediksi Sentimen: Positif
    """)

with st.expander("Contoh Teks dengan Sentimen Negatif"):
    st.write("""
    **Input:** I was disappointed with the plot. It felt like a waste of time.
    
    **Output:** Prediksi Sentimen: Negatif
    """)

# Footer
st.markdown("---")
st.caption("Aplikasi Analisis Sentimen IMDB Movie Reviews - Dibuat dengan Streamlit")
