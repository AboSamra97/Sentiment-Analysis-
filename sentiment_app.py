# sentiment_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import re
import nltk
import os
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- NLTK Setup ---
NLTK_DATA_PATH = os.path.join(os.path.expanduser("~"), "nltk_data")

# Create directory if not exists
os.makedirs(NLTK_DATA_PATH, exist_ok=True)

# Download resources if missing
try:
    nltk.data.find('tokenizers/punkt', paths=[NLTK_DATA_PATH])
    nltk.data.find('corpora/wordnet', paths=[NLTK_DATA_PATH])
    nltk.data.find('corpora/stopwords', paths=[NLTK_DATA_PATH])
except LookupError:
    try:
        nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)
        nltk.download('wordnet', download_dir=NLTK_DATA_PATH, quiet=True)
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH, quiet=True)
    except Exception as e:
        st.error(f"NLTK setup failed: {str(e)}")

# Update NLTK path
nltk.data.path.append(NLTK_DATA_PATH)

# --- Constants ---
MAX_SEQUENCE_LENGTH = 80  # Must match your training
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

# --- Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])
    words = nltk.word_tokenize(text)
    return ' '.join([LEMMATIZER.lemmatize(word) for word in words])

# --- Model Loading ---
@st.cache_resource
def load_components():
    try:
        model = tf.keras.models.load_model('lstm_sentiment_model.keras')
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading components: {e}")
        return None, None

# --- Prediction Function ---
def predict_sentiment(text):
    model, tokenizer = load_components()
    if not model or not tokenizer:
        return None, None
    
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, 
                          padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    return ("Positive 😊", prediction) if prediction >= 0.5 else ("Negative 😞", prediction)

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="🎥 Sentiment Analyzer",page_icon="🎥", layout="centered")
    
    st.title("🎥 IMDB Movie Reviews Sentiment Analyzer")
    st.markdown("Analyze sentiment using LSTM model")
    
    tweet = st.text_area("Enter your tweet:", height=150, 
                        placeholder="I really enjoyed this movie...")
    
    if st.button("Analyze", type="primary"):
        if not tweet.strip():
            st.error("Please enter some text")
        else:
            with st.spinner("Analyzing..."):
                sentiment, confidence = predict_sentiment(tweet)
                
                if sentiment:
    # Calculate confidence for the predicted class
                    confidence_score = confidence if "Positive" in sentiment else (1 - confidence)
    
                    color = "#1DA1F2" if "Positive" in sentiment else "#E0245E"
                    st.markdown(f'<h2 style="text-align: center; color: {color}">{sentiment}</h2>', 
                            unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align: center; font-size: 1.2rem;">Confidence: {confidence_score:.2%}</div>', 
                            unsafe_allow_html=True)
                    st.progress(float(confidence_score))
                else:
                    st.error("Analysis failed. Please check model files.")

    st.markdown("---")
    st.markdown("*AI-powered Sentiment Analysis*")

if __name__ == "__main__":
    main()
