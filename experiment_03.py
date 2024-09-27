"""
Project:    Advancements in Identifying Online Grooming: A Critical Analysis of Machine Learning Methodologies
            Edge Hill University - CIS4114 - MSc Research & Development Project
Author:     Mr Ricardo Lopes (24305731@edgehill.ac.uk)
Created:    2024-08-30
Version:    1.0
Ownership:  Mr Ricardo Lopes (24305731@edgehill.ac.uk)
License:    MIT License
Filename:   experiment_03.py

This script processes and analyses chat conversations from the PAN2012 Sexual Predator Identification dataset and additional datasets (PJZ and PJZC).
It performs the following steps:

1. Data Loading:
    - Loads the list of predator IDs from a text file.

2. Model and Vectorizer Loading:
    - Checks if pre-trained model and vectorizer pickle files exist.
    - If they exist, loads the Logistic Regression model and BoW vectorizer from pickle files.
    - If not, parses and processes the PAN2012 training data, trains a Logistic Regression model, and saves both the model and vectorizer to pickle files.

3. Data Preprocessing and Feature Extraction:
    - Parses chat conversations from XML files for both training and test datasets.
    - Labels messages as 'predator' or 'non-predator' based on the predator IDs.
    - Fills missing text data with empty strings.
    - Applies text preprocessing to clean and standardise messages (e.g., tokenization, stopword removal).
    - Converts text data into BoW features, limiting the number of features to 10000.

4. Model Training (if needed):
    - Trains a Logistic Regression model with class weights balanced (class_weight='balanced') to account for the imbalance in the PAN2012 training data, and a maximum of 1000 iterations (max_iter=1000) to ensure sufficient time for convergence, if the model does not already exist.
    - Saves the trained model and vectorizer for future use.

5. Model Evaluation:
    - Applies the trained model to the PAN2012 test dataset.
    - Evaluates the model's performance using accuracy, precision, recall, F1 score, AUC-ROC, and confusion matrix metrics.
    - Logs the results.

6. Evaluation on Additional Datasets (PJZ and PJZC):
    - Parses and processes PJZ/PJZC datasets.
    - Labels the messages as 'groomer' or 'non-groomer'.
    - Converts text data into BoW features using the pre-trained vectorizer.
    - Predicts and evaluates the model on PJZ and PJZC datasets.
    - Logs the results for these additional datasets.
"""

import os
import pickle
import pandas as pd
from auxiliary import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# File paths for the pickle files
model_file_path = 'e03_logistic_regression_model.pkl'
vectorizer_file_path = 'e03_bow_vectorizer.pkl'

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
    # Parse and label the training data for PAN2012 dataset
    pan_training_conversations = parse_conversations('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml')
    pan_training_data = label_messages(pan_training_conversations, pan_predator_ids)

    # Convert to DataFrame and preprocess the training data
    pan_train_df = pd.DataFrame(pan_training_data)
    pan_train_df['text'] = pan_train_df['text'].fillna('').apply(preprocess_text)

    # Convert text to BoW features
    vectorizer = CountVectorizer(max_features=10000)
    X_train_bow = vectorizer.fit_transform(pan_train_df['text'])
    y_train = pan_train_df['label']

    # Train a Logistic Regression model (maximum of 1000 iterations)
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_bow, y_train)

    # Save the model and vectorizer to pickle files
    with open(model_file_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(vectorizer_file_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    logging.info(f"Model {model_file_path} and vectorizer {vectorizer_file_path} trained and saved to pickle files.")

# Parse and preprocess the PAN2012 test data
pan_test_conversations = parse_conversations('data/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')
pan_test_data = label_messages(pan_test_conversations, pan_predator_ids)
pan_test_df = pd.DataFrame(pan_test_data)
pan_test_df['text'] = pan_test_df['text'].fillna('').apply(preprocess_text)

# Convert test text to BoW features
X_test_bow = vectorizer.transform(pan_test_df['text'])
y_test = pan_test_df['label']

# Predict on the PAN2012 test set and evaluate
y_pred = model.predict(X_test_bow)
y_pred_prob = model.predict_proba(X_test_bow)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
auc_roc = roc_auc_score(y_test, y_pred_prob)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

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
X_pjz_bow = vectorizer.transform(pjz_df['text'])
X_pjzc_bow = vectorizer.transform(pjzc_df['text'])
y_pjz = pjz_df['label']
y_pjzc = pjzc_df['label']

# Predict on the PJZ/PJZC dataset using the pre-trained PAN2012 model
y_pjz_pred = model.predict(X_pjz_bow)
y_pjzc_pred = model.predict(X_pjzc_bow)

# Store the predicted probabilities
y_pjz_pred_prob = model.predict_proba(X_pjz_bow)[:, 1]
y_pjzc_pred_prob = model.predict_proba(X_pjzc_bow)[:, 1]

# Evaluate the model on the PJZ/PJZC dataset
accuracy_pjz = accuracy_score(y_pjz, y_pjz_pred)
precision_pjz = precision_score(y_pjz, y_pjz_pred)
recall_pjz = recall_score(y_pjz, y_pjz_pred)
f1_pjz = f1_score(y_pjz, y_pjz_pred)
auc_roc_pjz = roc_auc_score(y_pjz, y_pjz_pred_prob)

accuracy_pjzc = accuracy_score(y_pjzc, y_pjzc_pred)
precision_pjzc = precision_score(y_pjzc, y_pjzc_pred)
recall_pjzc = recall_score(y_pjzc, y_pjzc_pred)
f1_pjzc = f1_score(y_pjzc, y_pjzc_pred)
auc_roc_pjzc = roc_auc_score(y_pjzc, y_pjzc_pred_prob)

# Calculate confusion matrices for PJZ and PJZC datasets
tn_pjz, fp_pjz, fn_pjz, tp_pjz = confusion_matrix(y_pjz, y_pjz_pred).ravel()
tn_pjzc, fp_pjzc, fn_pjzc, tp_pjzc = confusion_matrix(y_pjzc, y_pjzc_pred).ravel()

logging.info(f'PJZ Accuracy: {accuracy_pjz:.2f}, Precision: {precision_pjz:.2f}, Recall: {recall_pjz:.2f}, F1 Score: {f1_pjz:.2f}, AUC-ROC: {auc_roc_pjz:.2f}, TP: {tp_pjz}, FN: {fn_pjz}, FP: {fp_pjz}, TN: {tn_pjz}')
logging.info(f'PJZC Accuracy: {accuracy_pjzc:.2f}, Precision: {precision_pjzc:.2f}, Recall: {recall_pjzc:.2f}, F1 Score: {f1_pjzc:.2f}, AUC-ROC: {auc_roc_pjzc:.2f}, TP: {tp_pjzc}, FN: {fn_pjzc}, FP: {fp_pjzc}, TN: {tn_pjzc}')
