IMDb Movie Review Sentiment Analysis using SimpleRNN

A Deep Learning project that uses an Embedding layer and SimpleRNN to classify IMDb movie reviews as positive or negative.

The trained model is integrated with a Streamlit application for interactive sentiment prediction.

Quick Revision — Complete Workflow

IMDb Dataset
     ↓
Integer Encoding
     ↓
Padding
     ↓
Embedding
     ↓
SimpleRNN
     ↓
Dense + Sigmoid
     ↓
Training
     ↓
Evaluation
     ↓
Save .h5 Model
     ↓
Load Model
     ↓
Streamlit
     ↓
User Review
     ↓
Preprocessing
     ↓
Prediction
     ↓
Positive / Negative

1. IMDb Dataset

The project uses the IMDb movie review dataset provided by Keras.

from tensorflow.keras.datasets import imdb

The vocabulary limit used in the project is:

max_features = 10000

The IMDb dataset contains 25,000 training reviews and 25,000 testing reviews.

2. Integer Encoding

A neural network cannot directly process raw text.

"This movie was excellent"
        ↓
Integer IDs
        ↓
[integer, integer, integer, ...]

IMDb provides a mapping between words and integer IDs.

3. Word Index

word_index = imdb.get_word_index()

reverse_word_index = {
    value: key for key, value in word_index.items()
}

The reverse mapping is useful for inspecting and decoding encoded reviews.

4. Padding

Reviews have different lengths.

The project uses:

max_len = 500

Padding is performed with:

sequence.pad_sequences(
    X_train,
    maxlen=max_len
)

After padding:

X_train → (25000, 500)
X_test  → (25000, 500)

Shorter sequences receive padding values and longer sequences are truncated to the selected maximum length.

5. Embedding Layer

Integer word IDs are converted into dense vectors.

The project uses:

Embedding(
    input_dim=10000,
    output_dim=10,
    input_length=500
)

Therefore one word ID becomes a 10-dimensional vector.

A sequence of 500 tokens becomes:

(500,) → (500, 10)

The embedding representations are learned during training.

6. SimpleRNN

The embedded sequence is passed to:

SimpleRNN(5)

The RNN processes the sequence step by step and maintains a hidden state carrying information from previous time steps.

The project uses 5 hidden units.

The output shape is:

(None, 5)

None represents the batch size.

7. Dense + Sigmoid

The RNN output is passed to:

Dense(1, activation='sigmoid')

There is one output neuron because this is binary classification.

Sigmoid produces a value between 0 and 1.

0.85 → Positive
0.20 → Negative

The project uses a threshold of 0.5 for the final classification.

8. Model Architecture

Input Review
     ↓
500 Token IDs
     ↓
Embedding
10000 vocabulary
10-dimensional vectors
     ↓
SimpleRNN
5 hidden units
     ↓
Dense
1 neuron
Sigmoid
     ↓
Prediction Probability

Model parameters:

Embedding → 100,000
SimpleRNN → 80
Dense     → 6
Total     → 100,086

9. Model Compilation

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

Adam

Adam is the optimizer. It updates model weights using gradients calculated from the loss.

Binary Crossentropy

Binary crossentropy is used for the two-class sentiment problem.

Accuracy

Accuracy measures the proportion of correctly classified predictions.

10. Training

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

Training flow:

Input
  ↓
Embedding
  ↓
SimpleRNN
  ↓
Dense + Sigmoid
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Adam updates weights

11. Evaluation

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

This produces test loss and test accuracy using the separate test dataset.

12. Saving the Model

model.save("simple_rnn_imdb_relu.h5")

The saved model can later be loaded without retraining:

model = load_model("simple_rnn_imdb_relu.h5")

13. New Review Preprocessing

A new review must be converted into the same type of representation used during training.

User Review
     ↓
Lowercase
     ↓
Split into words
     ↓
Word → IMDb ID
     ↓
Padding to 500
     ↓
Model

The preprocessing used during prediction must be compatible with the preprocessing used during training.

14. Prediction Function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(
        preprocessed_input
    )

    score = float(prediction[0][0])

    sentiment = (
        "Positive"
        if score > 0.5
        else "Negative"
    )

    return sentiment, score

The model returns a probability score between 0 and 1.

15. Streamlit Application

User enters movie review
          ↓
Streamlit
          ↓
preprocess_text()
          ↓
Padding
          ↓
Saved RNN Model
          ↓
model.predict()
          ↓
Prediction Score
          ↓
Positive / Negative

The Streamlit application is implemented in main.py.

16. Project Structure

rnnclassification/
│
├── embedding.ipynb
├── main.py
├── README.md
├── requirements.txt
│
├── simple_rnn_imdb_relu.h5
│
└── .gitignore

17. Technologies

Python

TensorFlow

Keras

NumPy

Streamlit

18. Important Concepts Learned

Integer Encoding

Converts words into numerical IDs.

Padding

Makes sequences the same length.

Embedding

Converts word IDs into dense vectors.

RNN

Processes sequential information while maintaining a hidden state.

Hidden State

Carries information from previous time steps.

Sigmoid

Produces a value between 0 and 1 for binary classification.

Binary Classification

The model chooses between Positive and Negative.

Loss

Measures the difference between prediction and actual label.

Optimizer

Updates model weights to reduce the loss.

19. Key Learning

The central idea is:

Text
 ↓
Numbers
 ↓
Vectors
 ↓
Sequence Processing
 ↓
Classification

The Embedding layer converts integer word IDs into learned vector representations, and the SimpleRNN processes those representations as a sequence.

The trained model can then be reused for inference through the Streamlit application.