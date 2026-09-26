import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

MAX_FEATURES = 10000
MAX_LEN = 500

# Get the folder where this main.py is located
BASE_DIR = Path(__file__).resolve().parent

# Model is in the same folder as main.py
MODEL_PATH = BASE_DIR / "simple_rnn_imdb_relu.h5"


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# ============================================================
# Load IMDb Word Index
# ============================================================

@st.cache_data
def load_word_index():
    return imdb.get_word_index()


word_index = load_word_index()


# ============================================================
# Load Trained RNN Model
# ============================================================

@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return load_model(MODEL_PATH)


model = load_trained_model()


# ============================================================
# Preprocess Text
# ============================================================

def preprocess_text(text):

    # Convert text to lowercase
    words = text.lower().split()

    # IMDb special token handling
    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    # Keep values inside vocabulary range
    encoded_review = [
        token if token < MAX_FEATURES else 2
        for token in encoded_review
    ]

    # Pad / truncate review to 500 tokens
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=MAX_LEN
    )

    return padded_review


# ============================================================
# Prediction
# ============================================================

def predict_sentiment(review):

    preprocessed_input = preprocess_text(review)

    prediction = model.predict(
        preprocessed_input,
        verbose=0
    )

    score = float(prediction[0][0])

    if score >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, score


# ============================================================
# Streamlit UI
# ============================================================

st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and the SimpleRNN model "
    "will classify it as positive or negative."
)

st.divider()

user_input = st.text_area(
    "Movie Review",
    placeholder="Example: This movie was absolutely fantastic..."
)

if st.button("Classify Review"):

    if user_input.strip():

        sentiment, score = predict_sentiment(user_input)

        st.subheader(f"Sentiment: {sentiment}")

        st.write(
            f"Prediction Score: **{score:.4f}**"
        )

        if sentiment == "Positive":
            st.success("The model predicts a positive review.")
        else:
            st.error("The model predicts a negative review.")

    else:

        st.warning(
            "Please enter a movie review before clicking Classify Review."
        )


# ============================================================
# Model Information
# ============================================================

with st.expander("Model Information"):

    st.write("**Model:** SimpleRNN")
    st.write("**Dataset:** IMDb Movie Reviews")
    st.write("**Vocabulary Size:** 10,000")
    st.write("**Maximum Review Length:** 500")
    st.write("**Embedding Dimension:** 10")
    st.write("**RNN Units:** 5")
    st.write("**Output:** Binary classification")
