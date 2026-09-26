
# IMDb Movie Review Sentiment Analysis using RNN

A Deep Learning project that predicts whether an IMDb movie review is **Positive** or **Negative** using a Simple Recurrent Neural Network (RNN).

## Quick Revision — Complete Workflow

```text
IMDb Movie Reviews
        ↓
Text Preprocessing
        ↓
Tokenization / Word Index
        ↓
Integer Encoding
        ↓
Sequence Padding
        ↓
Embedding Layer
        ↓
SimpleRNN
        ↓
Dense Layer
        ↓
Sigmoid Output
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Save Trained Model
        ↓
Streamlit Application
        ↓
User Review
        ↓
Same Preprocessing
        ↓
RNN Prediction
        ↓
Sentiment Score
        ↓
Positive / Negative
1. Dataset

The project uses the IMDb Movie Reviews dataset available through TensorFlow/Keras.

The task is binary sentiment classification.

0 → Negative
1 → Positive

The model uses a vocabulary of:

MAX_FEATURES = 10000

and a maximum sequence length of:

MAX_LEN = 500
2. Text Preprocessing

Raw movie reviews are text, so they must be converted into numerical sequences before being passed to the RNN.

Word Index

The IMDb dataset provides a word-to-integer mapping.

Word
 ↓
Integer ID

Example:

"movie" → numerical ID
"good"  → numerical ID
Unknown Words

Words that are not available in the IMDb vocabulary are represented using the unknown-word token.

Unknown Word
      ↓
   Token 2
Sequence Encoding

The words in a review are converted into a sequence of integer IDs.

Review
   ↓
Words
   ↓
Integer IDs
Padding

Different reviews have different lengths.

Review 1 → 100 tokens
Review 2 → 300 tokens
Review 3 → 500 tokens

The model requires a fixed sequence length.

MAX_LEN = 500

Therefore, shorter sequences are padded and longer sequences are truncated.

3. Embedding Layer

The integer sequences are passed into an Embedding layer.

Integer Sequence
       ↓
Embedding Layer
       ↓
Dense Vector Representation

The Embedding layer learns numerical representations of words during training.

4. RNN Architecture
Input Sequence
      ↓
Embedding Layer
      ↓
SimpleRNN
      ↓
Dense Layer
      ↓
Sigmoid
      ↓
Sentiment Score

The SimpleRNN processes the sequence step by step while maintaining information from previous time steps.

This allows the model to work with sequential text data.

5. Important RNN Concepts
Sequence

Text is sequential data where the order of words matters.

Word 1 → Word 2 → Word 3 → Word 4
Embedding

Converts integer word IDs into learned dense vector representations.

RNN

Processes sequential information while maintaining information from previous time steps.

Hidden State

The RNN carries information from previous time steps while processing the sequence.

Sigmoid

The final output produces a value between 0 and 1.

Score >= 0.5 → Positive
Score < 0.5  → Negative
6. Training

During training:

Input Sequence
      ↓
Embedding
      ↓
SimpleRNN
      ↓
Prediction
      ↓
Loss Calculation
      ↓
Backpropagation
      ↓
Weight Updates

This process is repeated over batches and epochs.

7. Model Evaluation

After training, the model is evaluated using the IMDb test dataset.

The main evaluation metrics are:

Test Loss
Test Accuracy

The current model achieved:

Test Loss:     0.9308807253837585
Test Accuracy: 0.5794000029563904

The current model is primarily a learning implementation and can be improved with further experimentation.

8. Saving the Model

The trained RNN model is saved so that it can be reused without retraining.

simple_rnn_imdb_relu.h5

The saved model is loaded by the Streamlit application.

9. Prediction Workflow
New Movie Review
        ↓
Text Preprocessing
        ↓
IMDb Word Encoding
        ↓
Sequence Padding
        ↓
Saved RNN Model
        ↓
model.predict()
        ↓
Sentiment Score
        ↓
Positive / Negative

The model is not retrained during prediction.

10. Streamlit Workflow
User Review
    ↓
Streamlit
    ↓
Text Preprocessing
    ↓
IMDb Encoding
    ↓
Sequence Padding
    ↓
Saved RNN Model
    ↓
Prediction
    ↓
Sentiment Score
    ↓
Positive / Negative
11. Project Structure
rnnclassification/
│
├── main.py
├── embedding.ipynb
├── RNN_README.md
├── requirements.txt
└── simple_rnn_imdb_relu.h5
12. Technologies
Python
TensorFlow
Keras
NumPy
Streamlit
Jupyter Notebook
13. Key Learning

The same preprocessing logic used during training must be followed during prediction.

Training:

IMDb Review
     ↓
Integer Encoding
     ↓
Padding
     ↓
Embedding
     ↓
SimpleRNN
Prediction:

New Review
     ↓
Same Integer Encoding
     ↓
Same Padding
     ↓
Saved RNN Model
     ↓
Prediction

The important idea is:

Training Preprocessing
          =
Prediction Preprocessing

Otherwise, the model may receive input in a different format from what it learned during training.
