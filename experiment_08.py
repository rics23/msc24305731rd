"""
Project:    Advancements in Identifying Online Grooming: A Critical Analysis of Machine Learning Methodologies
            Edge Hill University - CIS4114 - MSc Research & Development Project
Author:     Mr Ricardo Lopes (24305731@edgehill.ac.uk)
Created:    2024-09-01
Version:    1.0
Ownership:  Mr Ricardo Lopes (24305731@edgehill.ac.uk)
License:    MIT License
Filename:   experiment_08.py

This script processes and analyzes chat conversations from the PAN2012 Sexual Predator Identification dataset using a
hybrid CNN-LSTM deep learning approach. The main steps include data parsing, preprocessing, feature extraction, class
balancing, model training, and evaluation. Here’s a breakdown of the process:

1. Data Parsing and Labeling
    Parsing: The script parses XML files containing chat conversations from the PAN2012 dataset for both training and testing.
    Labeling: Messages are labeled as 'predator' or 'non-predator' based on a predefined list of predator IDs. This is done by matching the IDs from the chat messages with the known predator IDs provided in the dataset.

2. Data Preprocessing
    Handling Missing Data: Any missing text data in the chat messages is filled with empty strings to ensure consistency during processing.
    Text Preprocessing: The text data undergoes cleaning and standardization, which includes steps such as tokenization and stopword removal. This step is crucial for ensuring that the input text is in a suitable format for vectorization and subsequent model training.

3. Feature Extraction
    - TF-IDF Vectorization: The preprocessed text data is converted into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) model, which captures the importance of words in each document.
    - LDA Topic Modeling: After vectorization, Latent Dirichlet Allocation (LDA) is applied for dimensionality reduction, generating topic-based features from the text.
    - Behavioral Features: In addition to text-based features, the script extracts behavioral features (user-based metadata) for analysis, which are scaled using `StandardScaler`.

4. Data Balancing
    SMOTE: Synthetic Minority Over-sampling Technique (SMOTE) is applied to balance the classes. Specifically, synthetic samples are generated for the minority class ('predator' messages), addressing the class imbalance issue in the training data.

5. Model Training
    - Hybrid CNN-LSTM Model: The script defines and trains a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) network. CNNs are useful for capturing local patterns in sequences, while LSTMs capture long-term dependencies in text data.
        - Embedding Layer: Converts input words into dense vectors using pre-trained GloVe embeddings.
        - Convolutional Layer: Extracts local features from sequences using 1D convolutions.
        - MaxPooling Layer: Reduces the spatial dimensions, selecting the most important features.
        - Bidirectional LSTM Layer: Captures sequential dependencies from both directions (past and future) in the text.
        - Global MaxPooling: Aggregates the most prominent features from the entire sequence.
        - Dense Layers: Fully connected layers output the probability of a message being from a predator.
    - Training Epochs: The model is trained over multiple epochs with early stopping to prevent overfitting.

6. Model Evaluation
    Prediction: After training, the model is used to predict labels on the test data.
    Evaluation Metrics: The model's performance is evaluated using several metrics:
    Accuracy: The overall correctness of the model.
    Precision, Recall, F1 Score: These metrics are particularly important in imbalanced classification problems like this one. However, in this case, they all returned 0.00 due to the model's failure to identify any positive cases (predators).
    Confusion Matrix: Provides a breakdown of true positives, false negatives, false positives, and true negatives.


7. Memory Management
    - Preprocessing Artifacts: After training and evaluation, preprocessed data, models, and embeddings are saved to disk using pickle and NumPy for reuse.
    - Garbage Collection: Unnecessary variables are deleted, and garbage collection is invoked to free up memory.

8. Additional Dataset Processing (PJZ and PJZC Datasets)
    - The script also processes two additional datasets (PJZ and PJZC), following the same preprocessing, feature extraction, and evaluation steps as for the PAN2012 dataset.
    - These datasets are used to further test the model's ability to generalize to unseen data.

"""

import os
import pickle
import pandas as pd
import numpy as np
from auxiliary import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, LSTM, Bidirectional, SpatialDropout1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec

# Load the list of predators
with open('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt', 'r') as f:
    pan_predator_ids = set(f.read().splitlines())

# Define file paths for loading/saving preprocessed data
vectorizer_path = 'e08_vectorizer.pkl'
label_encoder_path = 'e08_label_encoder.pkl'
resampled_data_path = 'e08_resampled_data.pkl'
model_path = 'e08_hybrid_model.keras'
embedding_matrix_path = 'e08_embedding_matrix.npy'
pan_train_path = 'e08_pan_train_df.pkl'
pan_test_path = 'e08_pan_test_df.pkl'

smote = SMOTE(sampling_strategy='minority')

# Load or prepare the data
if os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path) and os.path.exists(resampled_data_path) and os.path.exists(model_path) and os.path.exists(embedding_matrix_path):
    # Load the pickled data
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        le = pickle.load(f)
    with open(resampled_data_path, 'rb') as f:
        X_train_resampled, y_train_resampled = pickle.load(f)
    embedding_matrix = np.load(embedding_matrix_path)

    # Load DataFrames
    pan_train_df = pd.read_pickle(pan_train_path)
    pan_test_df = pd.read_pickle(pan_test_path)
    y_test_encoded = le.transform(pan_test_df['label'])
    X_test_tfidf = vectorizer.transform(pan_test_df['text'])

    logging.info(f"Loaded preprocessed data from pickle files {vectorizer_path}, {label_encoder_path}, {resampled_data_path}, {model_path}, {embedding_matrix_path}")
else:
    pan_training_conversations = parse_conversations('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml')
    pan_test_conversations = parse_conversations('data/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')

    pan_training_data = label_messages(pan_training_conversations, pan_predator_ids)
    pan_train_df = pd.DataFrame(pan_training_data)
    pan_train_df['text'] = pan_train_df['text'].fillna('').apply(preprocess_text)

    pan_test_data = label_messages(pan_test_conversations, pan_predator_ids)
    pan_test_df = pd.DataFrame(pan_test_data)
    pan_test_df['text'] = pan_test_df['text'].fillna('').apply(preprocess_text)

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(pan_train_df['label'])
    y_test_encoded = le.transform(pan_test_df['label'])

    # TF-IDF Vectorization
    max_features = 1000
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(pan_train_df['text'])
    X_test_tfidf = vectorizer.transform(pan_test_df['text'])

    # Dimensionality Reduction with LDA
    embedding_dim = 100
    glove_path = 'data/glove.6B/glove.6B.100d.txt'
    embedding_matrix = load_glove_embeddings(glove_path, vectorizer.vocabulary_, embedding_dim)
    np.save(embedding_matrix_path, embedding_matrix)

    # Apply SMOTE
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train_encoded)

    # Save the processed data using pickle
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(le, f)
    with open(resampled_data_path, 'wb') as f:
        pickle.dump((X_train_resampled, y_train_resampled), f)
    pan_train_df.to_pickle(pan_train_path)
    pan_test_df.to_pickle(pan_test_path)

    logging.info(f"Saved preprocessed data to pickle files {vectorizer_path}, {label_encoder_path}, {resampled_data_path}, {embedding_matrix_path}.")

# LDA Topic Modeling
print("LDA Topic Modeling")
n_topics = 10
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
X_train_lda = lda_model.fit_transform(X_train_resampled)
X_test_lda = lda_model.transform(X_test_tfidf)

print("Extracting Behavioural Features")
X_train_behavioral = extract_behavioral_features(pan_train_df)
X_test_behavioral = extract_behavioral_features(pan_test_df)

# Scale features
print("Scaling Features")
scaler = StandardScaler()
X_train_behavioral = scaler.fit_transform(X_train_behavioral)
X_test_behavioral = scaler.transform(X_test_behavioral)

# Resample the behavioral features to match the resampled text features
print("Resample the behavioral features to match the resampled text features")
X_train_behavioral_resampled, y_train_behavioral_resampled = smote.fit_resample(X_train_behavioral, y_train_encoded)

# Ensure all features have the same number of samples
print("Ensuring all features have the same number of samples")
assert X_train_resampled.shape[0] == X_train_behavioral_resampled.shape[0] == X_train_lda.shape[0], \
    "Mismatch in the number of samples across different feature sets."

# Combine all features after resampling
print("Combining all features after resampling")
X_train_combined = np.hstack((X_train_resampled.toarray(), X_train_lda, X_train_behavioral_resampled))
X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_lda, X_test_behavioral))

# Define and compile the hybrid CNN-LSTM model
input_shape = X_train_combined.shape[1]

model = Sequential()
model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_dim, input_length=input_shape, weights=[embedding_matrix], trainable=False))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the hybrid model
model.fit(X_train_combined, y_train_resampled, epochs=1, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Save the trained model
print("Saving the trained model")
model.save(model_path)
logging.info(f"Trained model saved as {model_path}")

# Evaluate the model
print("Evaluating the model")
y_pred = (model.predict(X_test_combined) > 0.5).astype("int32")

accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred)
recall = recall_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred)
auc_roc = roc_auc_score(y_test_encoded, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test_encoded, y_pred).ravel()

logging.info(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-ROC: {auc_roc}, TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}')

# File path for PJZ/PJZC datasets
pjz_dataset_path = 'data/pj/PJZ.txt'
pjzc_dataset_path = 'data/pj/PJZC.txt'

# Parse the PJZ/PJZC datasets
pjz_conversations = parse_pj_dataset(pjz_dataset_path)
pjzc_conversations = parse_pj_dataset(pjzc_dataset_path)

# Label the PJZ/PJZC datasets
pjz_data = label_pj_messages(pjz_conversations)
pjzc_data = label_pj_messages(pjzc_conversations)

pjz_df = pd.DataFrame(pjz_data)
pjzc_df = pd.DataFrame(pjzc_data)

pjz_df['text'] = pjz_df['text'].fillna('').apply(preprocess_text)
pjzc_df['text'] = pjzc_df['text'].fillna('').apply(preprocess_text)

# Transform the PJZ/PJZC datasets
X_pjz_tfidf = vectorizer.transform(pjz_df['text'])
X_pjzc_tfidf = vectorizer.transform(pjzc_df['text'])

X_pjz_lda = lda_model.transform(X_pjz_tfidf)
X_pjzc_lda = lda_model.transform(X_pjzc_tfidf)

X_pjz_behavioral = extract_behavioral_features(pjz_df)
X_pjzc_behavioral = extract_behavioral_features(pjzc_df)

X_pjz_behavioral_scaled = scaler.transform(X_pjz_behavioral)
X_pjzc_behavioral_scaled = scaler.transform(X_pjzc_behavioral)

X_pjz_combined = np.hstack((X_pjz_tfidf.toarray(), X_pjz_lda, X_pjz_behavioral_scaled))
X_pjzc_combined = np.hstack((X_pjzc_tfidf.toarray(), X_pjzc_lda, X_pjzc_behavioral_scaled))

y_pjz_pred = (model.predict(X_pjz_combined) > 0.5).astype(int)
y_pjzc_pred = (model.predict(X_pjzc_combined) > 0.5).astype(int)

# Predict probabilities (for AUC-ROC calculation)
y_pjz_pred_prob = model.predict(X_pjz_combined).flatten()
y_pjzc_pred_prob = model.predict(X_pjzc_combined).flatten()

# Evaluate PJZ dataset
accuracy_pjz = accuracy_score(pjz_df['label'], y_pjz_pred)
precision_pjz = precision_score(pjz_df['label'], y_pjz_pred)
recall_pjz = recall_score(pjz_df['label'], y_pjz_pred)
f1_pjz = f1_score(pjz_df['label'], y_pjz_pred)
auc_roc_pjz = roc_auc_score(pjz_df['label'], y_pjz_pred_prob)

# Evaluate PJZC dataset
accuracy_pjzc = accuracy_score(pjzc_df['label'], y_pjzc_pred)
precision_pjzc = precision_score(pjzc_df['label'], y_pjzc_pred)
recall_pjzc = recall_score(pjzc_df['label'], y_pjzc_pred)
f1_pjzc = f1_score(pjzc_df['label'], y_pjzc_pred)
auc_roc_pjzc = roc_auc_score(pjzc_df['label'], y_pjzc_pred_prob)

# Confusion matrix for PJZ and PJZC
tn_pjz, fp_pjz, fn_pjz, tp_pjz = confusion_matrix(pjz_df['label'], y_pjz_pred).ravel()
tn_pjzc, fp_pjzc, fn_pjzc, tp_pjzc = confusion_matrix(pjzc_df['label'], y_pjzc_pred).ravel()

# Logging the evaluation results
logging.info(f'PJZ Accuracy: {accuracy_pjz:.2f}, Precision: {precision_pjz:.2f}, Recall: {recall_pjz:.2f}, F1 Score: {f1_pjz:.2f}, AUC-ROC: {auc_roc_pjz:.2f}, TP: {tp_pjz}, FN: {fn_pjz}, FP: {fp_pjz}, TN: {tn_pjz}')
logging.info(f'PJZC Accuracy: {accuracy_pjzc:.2f}, Precision: {precision_pjzc:.2f}, Recall: {recall_pjzc:.2f}, F1 Score: {f1_pjzc:.2f}, AUC-ROC: {auc_roc_pjzc:.2f}, TP: {tp_pjzc}, FN: {fn_pjzc}, FP: {fp_pjzc}, TN: {tn_pjzc}')
