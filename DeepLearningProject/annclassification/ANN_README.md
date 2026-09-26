# Customer Churn Prediction using ANN

A Deep Learning project that predicts whether a customer is likely to churn using an Artificial Neural Network (ANN).

## Quick Revision — Complete Workflow

```text
Raw Customer Dataset
        ↓
Data Preprocessing
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Train/Test Split
        ↓
ANN Model
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Save Model + Preprocessing Objects
        ↓
Streamlit Application
        ↓
User Input
        ↓
Same Preprocessing
        ↓
ANN Prediction
        ↓
Churn Probability
        ↓
Stay / Churn
```

## 1. Data Preprocessing

The raw customer dataset contains numerical and categorical features.

Categorical features must be converted into numerical representations before being given to a neural network.

### Gender

Gender is converted using label encoding.

```text
label_encoder_gender.pkl
```

### Geography

Geography contains multiple categories and is converted using one-hot encoding.

```text
onehot_encoder_geo.pkl
```

### Feature Scaling

Numerical features are scaled before being passed to the ANN.

```text
scaler.pkl
```

The same preprocessing objects used during training must be used during prediction.

## 2. Train/Test Split

The processed dataset is divided into training and testing data.

```text
Training Data
     ↓
Used to learn patterns

Testing Data
     ↓
Used to evaluate the trained model
```

The model should not learn from the test data.

## 3. ANN Architecture

```text
Input Features
      ↓
Input Layer
      ↓
Hidden Layer(s)
      ↓
Output Layer
      ↓
Churn Probability
```

The hidden layers learn relationships between customer characteristics and churn.

## 4. Training

During training:

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss Calculation
  ↓
Backpropagation
  ↓
Weight Updates
```

This process is repeated over batches and epochs.

## 5. Important ANN Concepts

### Forward Propagation
Input data moves through the neural network to produce a prediction.

### Loss
Loss measures the difference between the actual value and the model prediction.

### Backpropagation
The error is propagated backward so that the model can calculate how its weights should be adjusted.

### Optimizer
The optimizer updates the model's weights using the calculated gradients.

### Feature Scaling
Scaling puts numerical features into a suitable range for neural network training.

### Encoding
Categorical values are converted into numerical representations.

## 6. Saving the Model

The trained ANN model is saved so that it can be reused without retraining.

```text
churn_model.h5
```

The preprocessing objects are also saved:

```text
label_encoder_gender.pkl
onehot_encoder_geo.pkl
scaler.pkl
```

## 7. Prediction Workflow

```text
New Customer
     ↓
Gender Encoding
     ↓
Geography One-Hot Encoding
     ↓
Feature Scaling
     ↓
Saved ANN Model
     ↓
model.predict()
     ↓
Churn Probability
     ↓
Stay / Churn
```

The model is not retrained during prediction.

## 8. Streamlit Workflow

```text
User Input
    ↓
Streamlit
    ↓
Preprocessing
    ↓
Saved ANN Model
    ↓
Prediction
    ↓
Churn Probability
    ↓
Stay / Churn
```

## 9. Project Structure

```text
annclassification/
│
├── app.py
├── README.md
├── requirements.txt
│
├── churn_model.h5
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
└── scaler.pkl
```

## 10. Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit

## 11. Key Learning

Training and prediction must use compatible preprocessing.

```text
Training:
Raw Data → Encoding → Scaling → ANN
```

```text
Prediction:
New Data → Same Encoding → Same Scaling → Saved ANN
```
