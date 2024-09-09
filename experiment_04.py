"""
Project:    Advancements in Identifying Online Sexual Predators: A Critical Analysis of Machine Learning Methodologies
            Edge Hill University - CIS4114 - MSc Research & Development Project
Author:     Mr Ricardo Lopes (24305731@edgehill.ac.uk)
Created:    2024-08-30
Version:    1.0
Ownership:  Mr Ricardo Lopes (24305731@edgehill.ac.uk)
License:    MIT License

This script processes and analyses chat conversations from the PAN2012 Sexual Predator Identification dataset using an
LSTM-based deep learning approach. It performs the following steps:

1. Data Parsing and Labeling:
    - Parses the XML files containing chat conversations for both training and testing data.
    - Labels the messages as 'predator' or 'non-predator' based on a predefined list of predator IDs.

2. Data Preprocessing:
    - Fills missing text data with empty strings.
    - Applies text preprocessing to clean and standardise the messages (e.g., tokenization, stopword removal).

3. Feature Extraction (Bag of Words Vectorisation):
    - Converts the preprocessed text data into numerical features using BoW vectorisation.
    - Pads the sequences to ensure uniform input size for the LSTM model.

4. Model Training:
    - Trains an LSTM (Long Short-Term Memory) neural network on the training data.
    - Applies dropout techniques to prevent overfitting and improve generalisation.

5. Model Evaluation:
    - Applies the trained model to the test data.
    - Evaluates the model's performance using accuracy, precision, recall, F1 score, and confusion matrix metrics.
"""

import os
import pickle
import pandas as pd
from auxiliary import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Define file paths for the pickle files and model
model_file_path = 'e04_lstm_model.keras'
vectorizer_file_path = 'e04_bow_vectorizer.pkl'
labelencoder_file_path = 'e04_label_encoder.pkl'

max_features = 10000
max_len = 100

# Load the list of predators
with open('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt', 'r') as f:
    pan_predator_ids = set(f.read().splitlines())

# Check if the model and vectorizer pickle files exist
if os.path.exists(model_file_path) and os.path.exists(vectorizer_file_path) and os.path.exists(labelencoder_file_path):
    # Load the LSTM model, vectorizer, and label encoder from files
    model = load_model(model_file_path)
    with open(vectorizer_file_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open(labelencoder_file_path, 'rb') as labelencoder_file:
        le = pickle.load(labelencoder_file)
    logging.info(f"LSTM model {model_file_path}, vectorizer {vectorizer_file_path}, and label encoder {labelencoder_file_path} loaded from files.")
else:
    # Parse the training XML file
    pan_training_conversations = parse_conversations('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml')

    # Label the training data
    pan_training_data = label_messages(pan_training_conversations, pan_predator_ids)

    # Convert to DataFrame
    pan_train_df = pd.DataFrame(pan_training_data)

    # Fill NaN values in the 'text' column with empty strings
    pan_train_df['text'] = pan_train_df['text'].fillna('')

    # Preprocess training data
    pan_train_df['text'] = pan_train_df['text'].apply(preprocess_text)

    # Extract labels
    y_train = pan_train_df['label']

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Prepare input data for LSTM
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(pan_train_df['text'])

    # Pad sequences to ensure uniform input size
    X_train_padded = pad_sequences(X_train_bow.toarray(), maxlen=max_len)

    # LSTM Model
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_padded, y_train_encoded, epochs=5, batch_size=64, validation_split=0.2)

    # Save the LSTM model, vectorizer, and label encoder to files
    model.save(model_file_path)
    with open(vectorizer_file_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    with open(labelencoder_file_path, 'wb') as labelencoder_file:
        pickle.dump(le, labelencoder_file)

    logging.info(f"LSTM model {model_file_path}, vectorizer {vectorizer_file_path}, and label encoder {labelencoder_file_path} trained and saved to files.")

# Parse and preprocess the test data
pan_test_conversations = parse_conversations('data/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')
pan_test_data = label_messages(pan_test_conversations, pan_predator_ids)
pan_test_df = pd.DataFrame(pan_test_data)
pan_test_df['text'] = pan_test_df['text'].fillna('')
pan_test_df['text'] = pan_test_df['text'].apply(preprocess_text)

# Encode labels
y_test = pan_test_df['label']
y_test_encoded = le.transform(y_test)

# Convert test text to BoW features and pad sequences
X_test_bow = vectorizer.transform(pan_test_df['text'])
X_test_padded = pad_sequences(X_test_bow.toarray(), maxlen=max_len)

# Predict on test data
y_pred_lstm = (model.predict(X_test_padded) > 0.5).astype(int)

# Predict probabilities (for AUC-ROC calculation)
y_pred_prob = model.predict(X_test_padded).flatten()

# Evaluate LSTM model
accuracy = accuracy_score(y_test_encoded, y_pred_lstm)
precision = precision_score(y_test_encoded, y_pred_lstm)
recall = recall_score(y_test_encoded, y_pred_lstm)
f1 = f1_score(y_test_encoded, y_pred_lstm)
tn, fp, fn, tp = confusion_matrix(y_test_encoded, y_pred_lstm).ravel()
auc_roc = roc_auc_score(y_test_encoded, y_pred_prob)

logging.info(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC-ROC: {auc_roc:.2f}, TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}')

