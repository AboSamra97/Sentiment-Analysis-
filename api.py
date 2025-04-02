import streamlit as st
import tensorflow as tf
import numpy as np
import re
import spacy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pickle

# Download NLTK resources and load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

try:
    stop_words = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Constants from your training
MAX_SEQUENCE_LENGTH = 200  # Update with your actual sequence length

# Sentiment mapping
SENTIMENT_MAP = {
    0: 'Negative 😞',
    1: 'Positive 😊'
}

def preprocess_text(text):
    """Replicate your preprocessing pipeline"""
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Lemmatization with spaCy
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc])
    
    return text

@st.cache_resource
def load_components():
    try:
        model = tf.keras.models.load_model('lstm_sentiment_model (1).keras')
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading components: {e}")
        return None, None

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
    sentiment = 1 if prediction >= 0.5 else 0
    return sentiment, prediction

def main():
    st.set_page_config(
        page_title="Tweet Sentiment Analyzer",
        page_icon="🐦",
        layout="centered"
    )

    st.markdown("""
    <style>
        /* ... (keep your existing CSS) */
        .confidence {
            font-size: 1.2rem;
            color: #657786;
            text-align: center;
            margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🐦 Tweet Sentiment Analyzer")
    st.markdown("Analyze sentiment using advanced NLP processing!")

    tweet_text = st.text_area(
        "Enter Tweet Text:", 
        placeholder="Paste your tweet here...", 
        height=150
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not tweet_text.strip():
            st.error("Please enter some text")
        else:
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence = predict_sentiment(tweet_text)
                
                if sentiment is not None:
                    sentiment_label = SENTIMENT_MAP.get(sentiment, "Unknown")
                    confidence_pct = f"{confidence * 100:.2f}%"
                    color = "#1DA1F2" if sentiment == 1 else "#E0245E"
                    
                    st.markdown(f'<h2 style="text-align: center; color: {color}">{sentiment_label}</h2>', 
                               unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence">Confidence: {confidence_pct}</div>', 
                               unsafe_allow_html=True)
                    
                    # Visual confidence indicator
                    st.progress(float(confidence if sentiment == 1 else 1-confidence))
                else:
                    st.error("Failed to analyze sentiment")

    st.markdown("---")
    st.markdown("*AI-powered Sentiment Analysis with LSTM*")

if __name__ == "__main__":
    main()