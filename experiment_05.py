"""
Project:    Advancements in Identifying Online Grooming: A Critical Analysis of Machine Learning Methodologies
            Edge Hill University - CIS4114 - MSc Research & Development Project
Author:     Mr Ricardo Lopes (24305731@edgehill.ac.uk)
Created:    2024-08-30
Version:    1.0
Ownership:  Mr Ricardo Lopes (24305731@edgehill.ac.uk)
License:    MIT License
Filename:   experiment_05.py

This script processes and analyses chat conversations from multiple datasets to identify online predators. It utilises
a CNN-based deep learning approach, with logistic regression used as a baseline in earlier experiments.
The key steps are as follows:

1. Data Loading:
    - Loads the list of predator IDs from a text file.

2. Model and Vectorizer Loading:
    - Checks if pre-trained model and vectorizer pickle files exist.
    - If they exist, loads the CNN model, BoW vectorizer and Label Encoder from pickle/keras files.
    - If not, parses and processes the PAN2012 training data, trains a CNN model, and saves the model, vectorizer, and label encoder to pickle/keras files.

3. Data Preprocessing and Feature Extraction:
    - Parses chat conversations from XML files for both training and test datasets.
    - Labels messages as 'predator' or 'non-predator' based on the predator IDs.
    - Fills missing text data with empty strings.
    - Applies text preprocessing to clean and standardise messages (e.g., tokenization, stopword removal).
    - Converts text data into BoW features, limiting the number of features to 10000.

4. Model Training (if needed):
    - Trains a CNN model on the training data, and applies dropout techniques to prevent overfitting and improve generalisation, if the model does not already exist.
    - Saves the trained model, vectorizer, and label encoder for future use.

5. Model Evaluation:
    - Applies the trained model to the PAN2012 test dataset.
    - Evaluates the model's performance using accuracy, precision, recall, F1 score, AUC-ROC, and confusion matrix metrics.
    - Logs the results.

6. Evaluation on Additional Datasets (PJZ and PJZC):
    - Parses and processes PJZ/PJZC datasets.
    - Labels the messages as 'groomer' or 'non-groomer'.
    - Encodes the labels for PJZ and PJZC datasets reusing the same LabelEncoder for consistency in label representation.
    - Converts text data into BoW features using the pre-trained vectorizer.
    - Predicts and evaluates the model on PJZ and PJZC datasets.
    - Logs the results for these additional datasets.
"""

import os
import pickle
import pandas as pd
from auxiliary import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Define file paths for the pickle files and the model
model_file_path = 'e05_cnn_model.keras'
vectorizer_file_path = 'e05_bow_vectorizer.pkl'
labelencoder_file_path = 'e05_label_encoder.pkl'

# Prepare input data for CNN
max_features = 10000
max_len = 100
svd_features = 300

# Define SVD outside the if-else block
svd = TruncatedSVD(n_components=svd_features)

# Load the list of predators
with open('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt', 'r') as f:
    pan_predator_ids = set(f.read().splitlines())

# Check if the model, vectorizer, and label encoder pickle files exist
if os.path.exists(model_file_path) and os.path.exists(vectorizer_file_path) and os.path.exists(labelencoder_file_path):
    # Load the CNN model, vectorizer, and label encoder from files
    model = load_model(model_file_path)
    with open(vectorizer_file_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open(labelencoder_file_path, 'rb') as labelencoder_file:
        le = pickle.load(labelencoder_file)
    logging.info(f"Model {model_file_path}, vectorizer {vectorizer_file_path}, and label encoder {labelencoder_file_path} loaded from files.")
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

    # Convert text to BoW features
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(pan_train_df['text'])

    # Apply TruncatedSVD to reduce dimensionality
    X_train_reduced = svd.fit_transform(X_train_bow)

    # Pad sequences to ensure uniform input size
    X_train_padded = pad_sequences(X_train_reduced, maxlen=max_len)

    # CNN Model
    model = Sequential()
    model.add(Embedding(input_dim=svd_features, output_dim=128, input_length=max_len))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_padded, y_train_encoded, epochs=5, batch_size=64, validation_split=0.2)

    # Save the model, vectorizer, and label encoder to files
    model.save(model_file_path)
    with open(vectorizer_file_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    with open(labelencoder_file_path, 'wb') as labelencoder_file:
        pickle.dump(le, labelencoder_file)

    logging.info(f"Model {model_file_path}, vectorizer {vectorizer_file_path}, and label encoder {labelencoder_file_path} trained and saved to files.")

# Parse and preprocess the test data
pan_test_conversations = parse_conversations('data/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')
pan_test_data = label_messages(pan_test_conversations, pan_predator_ids)
pan_test_df = pd.DataFrame(pan_test_data)
pan_test_df['text'] = pan_test_df['text'].fillna('').apply(preprocess_text)

# Encode labels
y_test = pan_test_df['label']
y_test_encoded = le.transform(y_test)

# Convert test text to BoW features
X_test_bow = vectorizer.transform(pan_test_df['text'])

# Apply the same SVD transformation to the test set
X_test_reduced = svd.transform(X_test_bow)

# Pad sequences to ensure uniform input size
X_test_padded = pad_sequences(X_test_reduced, maxlen=max_len)

# Predict on test data
y_pred_cnn = (model.predict(X_test_padded) > 0.5).astype(int)

# Predict probabilities (for AUC-ROC calculation)
y_pred_prob = model.predict(X_test_padded).flatten()

# Evaluate the CNN model
accuracy = accuracy_score(y_test_encoded, y_pred_cnn)
precision = precision_score(y_test_encoded, y_pred_cnn)
recall = recall_score(y_test_encoded, y_pred_cnn)
f1 = f1_score(y_test_encoded, y_pred_cnn)
tn, fp, fn, tp = confusion_matrix(y_test_encoded, y_pred_cnn).ravel()
auc_roc = roc_auc_score(y_test_encoded, y_pred_prob)

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

# Convert the PJZ/PJZC text to BoW features using the pre-trained vectorizer
y_pjz = pjz_df['label']
y_pjzc = pjzc_df['label']

# Encode labels for PJZ and PJZC
y_pjz_encoded = le.transform(y_pjz)
y_pjzc_encoded = le.transform(y_pjzc)

# Convert PJZ/PJZC text to BoW features using the pre-trained vectorizer
X_pjz_bow = vectorizer.transform(pjz_df['text'])
X_pjzc_bow = vectorizer.transform(pjzc_df['text'])

# Apply the same SVD transformation to the PJZ/PJZC datasets
X_pjz_reduced = svd.transform(X_pjz_bow)
X_pjzc_reduced = svd.transform(X_pjzc_bow)

# Pad sequences to match input size expected by the LSTM model
X_pjz_padded = pad_sequences(X_pjz_reduced, maxlen=max_len)
X_pjzc_padded = pad_sequences(X_pjzc_reduced, maxlen=max_len)

# Predict on the PJZ and PJZC datasets
y_pjz_pred = (model.predict(X_pjz_padded) > 0.5).astype(int)
y_pjzc_pred = (model.predict(X_pjzc_padded) > 0.5).astype(int)

# Predict probabilities (for AUC-ROC calculation)
y_pjz_pred_prob = model.predict(X_pjz_padded).flatten()
y_pjzc_pred_prob = model.predict(X_pjzc_padded).flatten()

# Evaluate the model on the PJZ/PJZC datasets
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
