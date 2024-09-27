"""
Project:    Advancements in Identifying Online Grooming: A Critical Analysis of Machine Learning Methodologies
            Edge Hill University - CIS4114 - MSc Research & Development Project
Author:     Mr Ricardo Lopes (24305731@edgehill.ac.uk)
Created:    2024-09-01
Version:    1.0
Ownership:  Mr Ricardo Lopes (24305731@edgehill.ac.uk)
License:    MIT License

This script processes and analyzes chat conversations from the PAN2012 Sexual Predator Identification dataset using an
LSTM-based deep learning approach. The main steps include data parsing, preprocessing, feature extraction, class
balancing, model training, and evaluation. Here’s a breakdown of the process:

1. Data Parsing and Labeling
    Parsing: The script parses XML files containing chat conversations from the PAN2012 dataset for both training and testing.
    Labeling: Messages are labeled as 'predator' or 'non-predator' based on a predefined list of predator IDs. This is done by matching the IDs from the chat messages with the known predator IDs provided in the dataset.

2. Data Preprocessing
    Handling Missing Data: Any missing text data in the chat messages is filled with empty strings to ensure consistency during processing.
    Text Preprocessing: The text data undergoes cleaning and standardization, which includes steps such as tokenization and stopword removal. This step is crucial for ensuring that the input text is in a suitable format for vectorization and subsequent model training.

3. Feature Extraction (Bag of Words Vectorization)
    Vectorization: The preprocessed text data is converted into numerical features using the Bag of Words (BoW) model. This involves creating a matrix where each row corresponds to a document (chat message), and each column corresponds to a word in the vocabulary.
    Padding: The resulting sequences are padded to ensure that all input data has a uniform size, which is required for feeding into the LSTM model.

4. Data Balancing
    SMOTE: Synthetic Minority Over-sampling Technique (SMOTE) is applied to balance the classes. Specifically, synthetic samples are generated for the minority class ('predator' messages), addressing the class imbalance issue in the training data.

5. Model Training
    LSTM Model: The script defines and trains a Long Short-Term Memory (LSTM) neural network. The LSTM is designed to handle sequential data, making it suitable for processing chat conversations.
    Embedding Layer: Converts input words into dense vectors of fixed size.
    Spatial Dropout: Applied to the embedding layer to prevent overfitting by randomly setting a fraction of input units to zero at each update during training.
    LSTM Layer: A recurrent layer that captures dependencies between words in a sequence.
    Dense Layer: The final fully connected layer with a sigmoid activation function outputs the probability of a message being from a predator.
    Batch Training: Due to the large size of the dataset, training is performed in batches. A custom function generates batches of data, ensuring that the model can be trained efficiently without memory overflow.
    Training Epochs: The model is trained over 5 epochs, iterating through the batches for each epoch.

6. Model Evaluation
    Prediction: After training, the model is used to predict labels on the test data.
    Evaluation Metrics: The model's performance is evaluated using several metrics:
    Accuracy: The overall correctness of the model.
    Precision, Recall, F1 Score: These metrics are particularly important in imbalanced classification problems like this one. However, in this case, they all returned 0.00 due to the model's failure to identify any positive cases (predators).
    Confusion Matrix: Provides a breakdown of true positives, false negatives, false positives, and true negatives.
    Results:

7. Memory Management
    Garbage Collection: After training and evaluation, unnecessary variables are deleted, and garbage collection is invoked to free up memory.

"""
import gc
import logging
import os
import pickle
import pandas as pd
import tensorflow as tf
from auxiliary import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

# Define file paths for the pickle files and the model
model_file_path = 'e06_lstm_model.keras'
vectorizer_file_path = 'e06_bow_vectorizer.pkl'
label_encoder_file_path = 'e06_label_encoder.pkl'

# Prepare input data for LSTM
max_features = 10000
max_len = 100

# Load the list of predators
with open('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt', 'r') as f:
    pan_predator_ids = set(f.read().splitlines())

# Check if the model, vectorizer, and label encoder pickle files exist
if os.path.exists(model_file_path) and os.path.exists(vectorizer_file_path) and os.path.exists(label_encoder_file_path):
    # Load the LSTM model, vectorizer, and label encoder from files
    model = load_model(model_file_path)
    with open(vectorizer_file_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open(label_encoder_file_path, 'rb') as le_file:
        le = pickle.load(le_file)
    logging.info(f"LSTM model {model_file_path}, vectorizer {vectorizer_file_path}, and label encoder {label_encoder_file_path} loaded from files.")
else:
    # Parse the training and testing XML files
    pan_training_conversations = parse_conversations('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml')
    pan_test_conversations = parse_conversations('data/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')

    # Label the training data
    pan_training_data = label_messages(pan_training_conversations, pan_predator_ids)
    pan_train_df = pd.DataFrame(pan_training_data)
    pan_train_df['text'] = pan_train_df['text'].fillna('').apply(preprocess_text)

    # Label the test data
    pan_test_data = label_messages(pan_test_conversations, pan_predator_ids)
    pan_test_df = pd.DataFrame(pan_test_data)
    pan_test_df['text'] = pan_test_df['text'].fillna('').apply(preprocess_text)

    # Extract labels
    y_train = pan_train_df['label']
    y_test = pan_test_df['label']

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(pan_train_df['text'])
    X_test_bow = vectorizer.transform(pan_test_df['text'])

    # Apply SMOTE to balance classes
    smote = SMOTE(sampling_strategy='minority')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_bow, y_train_encoded)

    # Function to generate batches of data
    def generate_batches(X, y, batch_size):
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            yield X[start:end].toarray(), y[start:end]

    # LSTM Model definition
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=64))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Batch size
    batch_size = 5000

    # Train the model in batches
    for X_batch, y_batch in generate_batches(X_train_resampled, y_train_resampled, batch_size):
        model.fit(X_batch, y_batch, epochs=5, batch_size=32)

    # Save the model, vectorizer, and label encoder to files
    model.save(model_file_path)
    with open(vectorizer_file_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    with open(label_encoder_file_path, 'wb') as le_file:
        pickle.dump(le, le_file)

    logging.info(f"LSTM model {model_file_path}, vectorizer {vectorizer_file_path}, and label encoder {label_encoder_file_path} trained and saved to files.")

# Parse and preprocess the test data
pan_test_conversations = parse_conversations('data/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')
pan_test_data = label_messages(pan_test_conversations, pan_predator_ids)
pan_test_df = pd.DataFrame(pan_test_data)
pan_test_df['text'] = pan_test_df['text'].fillna('')
pan_test_df['text'] = pan_test_df['text'].apply(preprocess_text)

# Extract labels
y_test = pan_test_df['label']
y_test_encoded = le.transform(y_test)

# Vectorize and pad test data
X_test_bow = vectorizer.transform(pan_test_df['text'])
X_test_dense = X_test_bow.toarray()

# Predict probabilities on the test set
y_pred_prob_lstm = model.predict(X_test_dense)

# Convert probabilities to binary predictions
y_pred_lstm = (y_pred_prob_lstm > 0.5).astype(int)

# Evaluate LSTM model
accuracy = accuracy_score(y_test_encoded, y_pred_lstm)
precision = precision_score(y_test_encoded, y_pred_lstm)
recall = recall_score(y_test_encoded, y_pred_lstm)
f1 = f1_score(y_test_encoded, y_pred_lstm)
tn, fp, fn, tp = confusion_matrix(y_test_encoded, y_pred_lstm).ravel()
auc_roc = roc_auc_score(y_test_encoded, y_pred_prob_lstm)

logging.info(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC-ROC: {auc_roc:.2f}, TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}')

# Clear memory after use
del X_train_resampled, X_test_bow, X_test_dense
gc.collect()
