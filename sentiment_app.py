import streamlit as st
import tensorflow as tf
import numpy as np
import re
import spacy
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# --- Configuration ---
MAX_NUM_WORDS = 15000
MAX_SEQUENCE_LENGTH = 80
MODEL_PATH = "lstm_sentiment_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

# --- NLP Setup ---
try:
    nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
except:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])

try:
    stop_words = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# --- Preprocessing Function ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

# --- Model Loading ---
@st.cache_resource
def load_components():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading components: {e}")
        return None, None

# --- Prediction Function ---
def predict_sentiment(text):
    model, tokenizer = load_components()
    if model is None or tokenizer is None:
        return None, None
    
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(
        sequence, 
        maxlen=MAX_SEQUENCE_LENGTH,
        padding='post', 
        truncating='post'
    )
    prediction = model.predict(padded_sequence)[0][0]
    return ("Positive 😊", prediction) if prediction >= 0.5 else ("Negative 😞", prediction)

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="🐦", layout="centered")
    
    # Custom CSS
    st.markdown("""
    <style>
        /* ... (your existing CSS styles) ... */
    </style>
    """, unsafe_allow_html=True)

    st.title("🐦 Tweet Sentiment Analyzer")
    tweet_text = st.text_area("Enter Tweet Text:", placeholder="Paste your tweet here...", height=150)

    if st.button("Analyze Sentiment", type="primary"):
        if not tweet_text.strip():
            st.error("Please enter some text")
        else:
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence = predict_sentiment(tweet_text)
                
                if sentiment:
                    color = "#1DA1F2" if "Positive" in sentiment else "#E0245E"
                    st.markdown(f'<h2 style="text-align: center; color: {color}">{sentiment}</h2>', unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align: center; font-size: 1.2rem;">Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
                    st.progress(float(confidence if "Positive" in sentiment else 1-confidence))
                else:
                    st.error("Analysis failed")

    st.markdown("---")
    st.markdown("*AI-powered Sentiment Analysis with LSTM*")

if __name__ == "__main__":
    main()
