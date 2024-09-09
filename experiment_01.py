"""
Project:    Advancements in Identifying Online Sexual Predators: A Critical Analysis of Machine Learning Methodologies
            Edge Hill University - CIS4114 - MSc Research & Development Project
Author:     Mr Ricardo Lopes (24305731@edgehill.ac.uk)
Created:    2024-08-30
Version:    1.0
Ownership:  Mr Ricardo Lopes (24305731@edgehill.ac.uk)
License:    MIT License

This script processes and analyses chat conversations from the PAN2012 Sexual Predator Identification dataset.
It performs the following steps:

1. Data Parsing and Labeling:
    - Parses the XML files containing chat conversations for both training and testing data.
    - Labels the messages as 'predator' or 'non-predator' based on a predefined list of predator IDs.

2. Data Preprocessing:
    - Fills missing text data with empty strings.
    - Applies text preprocessing to clean and standardise the messages (e.g., tokenization, stopword removal).

3. Feature Extraction (TF-IDF Vectorisation):
    - Converts the preprocessed text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorisation.
    - Limits the number of features to 5000.

4. Model Training:
    - Trains a Logistic Regression model on the training data to classify messages as 'predator' or 'non-predator'.

5. Model Evaluation:
    - Applies the trained model to the test data.
    - Evaluates the model's performance using accuracy, precision, recall, F1 score, and confusion matrix metrics.
"""

import os
import pickle
import pandas as pd
from auxiliary import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# File paths for the pickle files
model_file_path = 'e01_logistic_regression_model.pkl'
vectorizer_file_path = 'e01_tfidf_vectorizer.pkl'

# Load the list of predators
with open('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt', 'r') as f:
    pan_predator_ids = set(f.read().splitlines())

# Check if the model and vectorizer pickle files exist
if os.path.exists(model_file_path) and os.path.exists(vectorizer_file_path):
    # Load the model and vectorizer from pickle files
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_file_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    logging.info(f"Model {model_file_path} and vectorizer {vectorizer_file_path} loaded from pickle files.")
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

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(pan_train_df['text'])
    y_train = pan_train_df['label']

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Save the model and vectorizer to pickle files
    with open(model_file_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(vectorizer_file_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    logging.info(f"Model {model_file_path} and vectorizer {vectorizer_file_path} trained and saved to pickle files.")

# Preprocess test data
pan_test_conversations = parse_conversations('data/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')
pan_test_data = label_messages(pan_test_conversations, pan_predator_ids)
pan_test_df = pd.DataFrame(pan_test_data)
pan_test_df['text'] = pan_test_df['text'].fillna('').apply(preprocess_text)

# Convert test text to TF-IDF features
X_test_tfidf = vectorizer.transform(pan_test_df['text'])
y_test = pan_test_df['label']

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Store the predicted probabilities
y_pred_prob = model.predict_proba(X_test_tfidf)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_prob)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

logging.info(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC-ROC: {auc_roc:.2f}, TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}')
