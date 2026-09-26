import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# -----------------------------
# Configuration
# -----------------------------
MAX_FEATURES = 10000
MAX_LEN = 500
MODEL_PATH = "simple_rnn_imdb_relu.h5"


# -----------------------------
# Load IMDb word index
# -----------------------------
word_index = imdb.get_word_index()


# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)


model = load_trained_model()


# -----------------------------
# Preprocess user input
# -----------------------------
def preprocess_text(text):

    words = text.lower().split()

    encoded_review = [
        word_index[word] + 3 if word in word_index else 2
        for word in words
    ]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=MAX_LEN
    )

    return padded_review


# -----------------------------
# Prediction function
# -----------------------------
def predict_sentiment(review):

    preprocessed_input = preprocess_text(review)

    prediction = model.predict(
        preprocessed_input,
        verbose=0
    )

    score = float(prediction[0][0])

    sentiment = "Positive" if score > 0.5 else "Negative"

    return sentiment, score


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review to classify it as positive or negative."
)


# -----------------------------
# User input
# -----------------------------
user_input = st.text_area("Movie Review")


# -----------------------------
# Classification
# -----------------------------
if st.button("Classify"):

    if user_input.strip():

        sentiment, score = predict_sentiment(user_input)

        st.subheader(f"Sentiment: {sentiment}")

        st.write(
            f"Prediction Score: {score:.4f}"
        )

    else:

        st.warning("Please enter a movie review.")