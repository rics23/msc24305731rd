"""
Project:    Advancements in Identifying Online Sexual Predators: A Critical Analysis of Machine Learning Methodologies
            Edge Hill University - CIS4114 - MSc Research & Development Project
Author:     Mr Ricardo Lopes (24305731@edgehill.ac.uk)
Created:    2024-09-01
Version:    2.0
Ownership:  Mr Ricardo Lopes (24305731@edgehill.ac.uk)
License:    MIT License
Filename:   experiment_06a.py

This script processes and analyses chat conversations from multiple datasets to identify online predators. It utilises
an LSTM-based deep learning approach.
The key steps are as follows:

1. Data Parsing and Labeling
    Parsing: The script parses XML files containing chat conversations from the PAN2012 dataset for both training and testing.
    Labeling: Messages are labeled as 'predator' or 'non-predator' based on a predefined list of predator IDs. This is done by matching the IDs from the chat messages with the known predator IDs provided in the dataset.

2. Data Preprocessing
    Handling Missing Data: Any missing text data in the chat messages is filled with empty strings to ensure consistency during processing.
    Text Preprocessing: The text data undergoes cleaning and standardization, which includes steps such as tokenization and stopword removal. This step is crucial for ensuring that the input text is in a suitable format for tokenization and subsequent model training.

3. Feature Extraction (Tokenization)
    Tokenization: Instead of using a Bag of Words (BoW) model, the script now utilizes a Tokenizer to convert the text into sequences of integer word indices, which are compatible with the LSTM model's Embedding layer.
    Padding: The resulting sequences are padded to ensure that all input data has a uniform length. This step is required for feeding into the LSTM model, as it expects fixed-length input sequences.

4. Data Balancing
    Class Weights: Instead of using the Synthetic Minority Over-sampling Technique (SMOTE), class imbalance is now handled by computing class weights. This ensures that the model pays more attention to the minority class ('predator' messages) during training, addressing the imbalance issue directly within the model.

5. Model Training
    LSTM Model: The script defines and trains a Long Short-Term Memory (LSTM) neural network. The LSTM is designed to handle sequential data, making it suitable for processing chat conversations.
    Embedding Layer: Converts input words into dense vectors of fixed size.
    Spatial Dropout: Applied to the embedding layer to prevent overfitting by randomly setting a fraction of input units to zero at each update during training.
    LSTM Layer: A recurrent layer that captures dependencies between words in a sequence.
    Dense Layer: The final fully connected layer with a sigmoid activation function outputs the probability of a message being from a predator.
    Entire Dataset Training: Due to the memory capacity of the upgraded system, batch processing has been removed, and the entire dataset is now processed at once.
    Training Epochs: The model is trained over 5 epochs, iterating over the entire dataset in each epoch.

6. Model Evaluation
    Prediction: After training, the model is used to predict labels on the test data.
    Evaluation Metrics: The model's performance is evaluated using several metrics:
        Accuracy: The overall correctness of the model.
        Precision, Recall, F1 Score: These metrics are crucial in imbalanced classification problems like this one, as they measure the model's ability to correctly identify positive cases (predators).
        Confusion Matrix: Provides a breakdown of true positives, false negatives, false positives, and true negatives.
        AUC-ROC: The Area Under the Receiver Operating Characteristic curve measures the model's ability to distinguish between predator and non-predator messages.
"""

import os
import pickle
import pandas as pd
from auxiliary import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight
import numpy as np

# Define file paths for the pickle files and the model
model_file_path = 'e06_lstm_model.keras'
tokenizer_file_path = 'e06_tokenizer.pkl'
labelencoder_file_path = 'e06_label_encoder.pkl'

# Prepare input data for LSTM
max_features = 10000
max_len = 100

# Load the list of predators
with open('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt', 'r') as f:
    pan_predator_ids = set(f.read().splitlines())

# Check if the model, tokenizer, and label encoder pickle files exist
if os.path.exists(model_file_path) and os.path.exists(tokenizer_file_path) and os.path.exists(labelencoder_file_path):
    # Load the LSTM model, tokenizer, and label encoder from files
    model = load_model(model_file_path)
    with open(tokenizer_file_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    with open(labelencoder_file_path, 'rb') as le_file:
        le = pickle.load(le_file)
    logging.info(f"LSTM model {model_file_path}, tokenizer {tokenizer_file_path}, and label encoder {labelencoder_file_path} loaded from files.")
else:
    # Parse and label the training data for PAN2012 dataset
    pan_training_conversations = parse_conversations('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml')
    pan_training_data = label_messages(pan_training_conversations, pan_predator_ids)

    # Convert to DataFrame
    pan_train_df = pd.DataFrame(pan_training_data)
    pan_train_df['text'] = pan_train_df['text'].fillna('').apply(preprocess_text)

    # Extract labels
    y_train = pan_train_df['label']

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Tokenize the text data
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(pan_train_df['text'])
    X_train_seq = tokenizer.texts_to_sequences(pan_train_df['text'])

    # Pad sequences to ensure uniform input size
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)

    # Compute class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    class_weight_dict = dict(enumerate(class_weights))

    # LSTM Model
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=64, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_padded, y_train_encoded, epochs=5, batch_size=32, class_weight=class_weight_dict)

    # Save the model, tokenizer, and label encoder to files
    model.save(model_file_path)
    with open(tokenizer_file_path, 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)
    with open(labelencoder_file_path, 'wb') as le_file:
        pickle.dump(le, le_file)

    logging.info(f"Model {model_file_path}, tokenizer {tokenizer_file_path}, and label encoder {labelencoder_file_path} trained and saved to files.")

# Parse and preprocess the test data
pan_test_conversations = parse_conversations('data/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')
pan_test_data = label_messages(pan_test_conversations, pan_predator_ids)
pan_test_df = pd.DataFrame(pan_test_data)
pan_test_df['text'] = pan_test_df['text'].fillna('').apply(preprocess_text)

# Encode labels
y_test = pan_test_df['label']
y_test_encoded = le.transform(y_test)

# Tokenize and pad test data
X_test_seq = tokenizer.texts_to_sequences(pan_test_df['text'])
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

# Predict on test data
y_pred_prob_lstm = model.predict(X_test_padded)

# Predict probabilities (for AUC-ROC calculation)
y_pred_lstm = (y_pred_prob_lstm > 0.5).astype(int)

# Evaluate LSTM model
accuracy = accuracy_score(y_test_encoded, y_pred_lstm)
precision = precision_score(y_test_encoded, y_pred_lstm)
recall = recall_score(y_test_encoded, y_pred_lstm)
f1 = f1_score(y_test_encoded, y_pred_lstm)
tn, fp, fn, tp = confusion_matrix(y_test_encoded, y_pred_lstm).ravel()
auc_roc = roc_auc_score(y_test_encoded, y_pred_prob_lstm)

logging.info(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC-ROC: {auc_roc:.2f}, TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}')

# File path for PJZ/PJZC datasets
pjz_dataset_path = 'data/pj/PJZ.txt'
pjzc_dataset_path = 'data/pj/PJZC.txt'

# Parse the PJZ/PJZC datasets
pjz_conversations = parse_pj_dataset(pjz_dataset_path)
pjzc_conversations = parse_pj_dataset(pjzc_dataset_path)

# Label the PJZ/PJZC dataset
pjz_data = label_pj_messages(pjz_conversations)
pjzc_data = label_pj_messages(pjzc_conversations)

# Convert PJZ/PJZC data to DataFrames
pjz_df = pd.DataFrame(pjz_data)
pjzc_df = pd.DataFrame(pjzc_data)

# Fill NaN values in the 'text' column with empty strings and preprocess
pjz_df['text'] = pjz_df['text'].fillna('').apply(preprocess_text)
pjzc_df['text'] = pjzc_df['text'].fillna('').apply(preprocess_text)

# Tokenize the PJZ/PJZC text data using the pre-trained tokenizer
X_pjz_seq = tokenizer.texts_to_sequences(pjz_df['text'])
X_pjzc_seq = tokenizer.texts_to_sequences(pjzc_df['text'])

# Pad sequences to match input size expected by the LSTM model
X_pjz_padded = pad_sequences(X_pjz_seq, maxlen=max_len)
X_pjzc_padded = pad_sequences(X_pjzc_seq, maxlen=max_len)

# Encode labels for PJZ and PJZC
y_pjz = pjz_df['label']
y_pjzc = pjzc_df['label']
y_pjz_encoded = le.transform(y_pjz)
y_pjzc_encoded = le.transform(y_pjzc)

# Predict on the PJZ and PJZC datasets
y_pjz_pred = (model.predict(X_pjz_padded) > 0.5).astype(int)
y_pjzc_pred = (model.predict(X_pjzc_padded) > 0.5).astype(int)

# Predict probabilities (for AUC-ROC calculation)
y_pjz_pred_prob = model.predict(X_pjz_padded).flatten()
y_pjzc_pred_prob = model.predict(X_pjzc_padded).flatten()

# Evaluate the model on the PJZ/PJZC dataset
accuracy_pjz = accuracy_score(y_pjz_encoded, y_pjz_pred)
precision_pjz = precision_score(y_pjz_encoded, y_pjz_pred)
recall_pjz = recall_score(y_pjz_encoded, y_pjz_pred)
f1_pjz = f1_score(y_pjz_encoded, y_pjz_pred)
auc_roc_pjz = roc_auc_score(y_pjz_encoded, y_pjz_pred_prob)

accuracy_pjzc = accuracy_score(y_pjzc_encoded, y_pjzc_pred)
precision_pjzc = precision_score(y_pjzc_encoded, y_pjzc_pred)
recall_pjzc = recall_score(y_pjzc_encoded, y_pjzc_pred)
f1_pjzc = f1_score(y_pjzc_encoded, y_pjzc_pred)
auc_roc_pjzc = roc_auc_score(y_pjzc_encoded, y_pjzc_pred_prob)

# Calculate confusion matrices for PJZ and PJZC datasets
tn_pjz, fp_pjz, fn_pjz, tp_pjz = confusion_matrix(y_pjz_encoded, y_pjz_pred).ravel()
tn_pjzc, fp_pjzc, fn_pjzc, tp_pjzc = confusion_matrix(y_pjzc_encoded, y_pjzc_pred).ravel()

logging.info(f'PJZ Accuracy: {accuracy_pjz:.2f}, Precision: {precision_pjz:.2f}, Recall: {recall_pjz:.2f}, F1 Score: {f1_pjz:.2f}, AUC-ROC: {auc_roc_pjz:.2f}, TP: {tp_pjz}, FN: {fn_pjz}, FP: {fp_pjz}, TN: {tn_pjz}')
logging.info(f'PJZC Accuracy: {accuracy_pjzc:.2f}, Precision: {precision_pjzc:.2f}, Recall: {recall_pjzc:.2f}, F1 Score: {f1_pjzc:.2f}, AUC-ROC: {auc_roc_pjzc:.2f}, TP: {tp_pjzc}, FN: {fn_pjzc}, FP: {fp_pjzc}, TN: {tn_pjzc}')
