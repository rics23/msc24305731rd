"""
Project:    Advancements in Identifying Online Sexual Predators: A Critical Analysis of Machine Learning Methodologies
            Edge Hill University - CIS4114 - MSc Research & Development Project
Author:     Mr Ricardo Lopes (24305731@edgehill.ac.uk)
Created:    2024-09-01
Version:    1.0
Ownership:  Mr Ricardo Lopes (24305731@edgehill.ac.uk)
License:    MIT License
Filename:   experiment_07.py

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
    Padding: The resulting sequences are padded to ensure that all input data has a uniform size, which is required for feeding into the CNN model.

4. Data Balancing
    SMOTE: Synthetic Minority Over-sampling Technique (SMOTE) is applied to balance the classes. Specifically, synthetic samples are generated for the minority class ('predator' messages), addressing the class imbalance issue in the training data.

5. Model Training
    CNN Model: The script defines and trains a Convolutional Neural Network (CNN). CNNs are suitable for extracting local patterns and features from the input sequences, which is valuable for text classification tasks like identifying predator messages.
    Embedding Layer: Converts input words into dense vectors of fixed size.
    Convolutional Layers: Two Conv1D layers are used to capture local patterns in the text sequences.
    MaxPooling & Global MaxPooling: MaxPooling reduces the spatial dimensions, while Global MaxPooling selects the most prominent features across the entire sequence.
    Dense Layers: The final fully connected layers with ReLU and sigmoid activation functions output the probability of a message being from a predator.
    Batch Training: Training is performed using the built-in batching mechanism of Keras, ensuring efficient training without memory overflow.
    Training Epochs: The model is trained over 5 epochs with a batch size of 32.

6. Model Evaluation
    Prediction: After training, the model is used to predict labels on the test data.
    Evaluation Metrics: The model's performance is evaluated using several metrics:
    Accuracy: The overall correctness of the model.
    Precision, Recall, F1 Score: These metrics are particularly important in imbalanced classification problems like this one. However, in this case, they all returned 0.00 due to the model's failure to identify any positive cases (predators).
    Confusion Matrix: Provides a breakdown of true positives, false negatives, false positives, and true negatives.
    Results:
        Epoch 1/5 - 53915/53915 ━━━━━━━━━━━━━━━━━━━━ 501s 9ms/step - accuracy: 0.5011 - loss: 0.6919
        Epoch 2/5 - 53915/53915 ━━━━━━━━━━━━━━━━━━━━ 561s 10ms/step - accuracy: 0.5014 - loss: 0.6920
        Epoch 3/5 - 53915/53915 ━━━━━━━━━━━━━━━━━━━━ 649s 12ms/step - accuracy: 0.5011 - loss: 0.6918
        Epoch 4/5 - 53915/53915 ━━━━━━━━━━━━━━━━━━━━ 651s 12ms/step - accuracy: 0.5014 - loss: 0.6926
        Epoch 5/5 - 53915/53915 ━━━━━━━━━━━━━━━━━━━━ 654s 12ms/step - accuracy: 0.5000 - loss: 0.6930
                    64337/64337 ━━━━━━━━━━━━━━━━━━━━ 106s 2ms/step
        CNN Model Accuracy: 1.00
        CNN Model Precision: 0.00
        CNN Model Recall: 0.00
        CNN Model F1 Score: 0.00
        CNN TP: 0, FN: 185, FP: 2, TN: 2058594

7. Memory Management
    Garbage Collection: After training and evaluation, unnecessary variables are deleted, and garbage collection is invoked to free up memory.

"""

import os
import pickle
import pandas as pd
import tensorflow as tf
from auxiliary import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define file paths for the pickle files and the model
vectorizer_path = 'e07_vectorizer.pkl'
labelencoder_path = 'e07_label_encoder.pkl'
resampled_data_path = 'e07_resampled_data.pkl'
model_path = 'e07_cnn_model.keras'

max_features = 10000
max_len = 100

# Load the list of predators
with open('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt', 'r') as f:
    pan_predator_ids = set(f.read().splitlines())

# Check if we have pickled data to load
if os.path.exists(vectorizer_path) and os.path.exists(labelencoder_path) and os.path.exists(resampled_data_path) and os.path.exists(model_path):
    # Load the pickled data
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(labelencoder_path, 'rb') as f:
        le = pickle.load(f)
    with open(resampled_data_path, 'rb') as f:
        X_train_resampled, y_train_resampled = pickle.load(f)

    logging.info(f"Loaded preprocessed data from pickle files {vectorizer_path}, {labelencoder_path}, {resampled_data_path}, {model_path}")
else:
    # Parse and label the training and test data for PAN2012 dataset
    pan_training_conversations = parse_conversations('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml')
    pan_test_conversations = parse_conversations('data/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')

    pan_training_data = label_messages(pan_training_conversations, pan_predator_ids)
    pan_train_df = pd.DataFrame(pan_training_data)
    pan_train_df['text'] = pan_train_df['text'].fillna('').apply(preprocess_text)

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

    # Prepare input data for CNN
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(pan_train_df['text'])
    X_test_bow = vectorizer.transform(pan_test_df['text'])

    # Apply SMOTE to balance classes
    smote = SMOTE(sampling_strategy='minority')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_bow, y_train_encoded)

    # Save the processed data using pickle
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(labelencoder_path, 'wb') as f:
        pickle.dump(le, f)
    with open(resampled_data_path, 'wb') as f:
        pickle.dump((X_train_resampled, y_train_resampled), f)

    logging.info(f"Saved preprocessed data to pickle files {vectorizer_path}, {labelencoder_path}, {resampled_data_path}, {model_path}.")

# Pad the sequences for CNN input
X_train_padded = pad_sequences(X_train_resampled.toarray(), maxlen=max_len, padding='post')
X_test_padded = pad_sequences(X_test_bow.toarray(), maxlen=max_len, padding='post')

# CNN Model
if os.path.exists(model_path):
    # Load the model if it exists
    model = tf.keras.models.load_model(model_path)
    logging.info(f"Loaded model {model_path} from file.")
else:
    # Define and compile the model
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=64, input_length=max_len))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the CNN model
    model.fit(X_train_padded, y_train_resampled, epochs=5, batch_size=32)

    # Save the trained model
    model.save(model_path)
    logging.info(f"Saved model {model_path} to file.")

# Evaluate the CNN model on the test data
y_pred_prob = model.predict(X_test_padded)
y_pred_cnn = (y_pred_prob > 0.5).astype(int)

# Evaluate CNN model
accuracy = accuracy_score(y_test_encoded, y_pred_cnn)
precision = precision_score(y_test_encoded, y_pred_cnn)
recall = recall_score(y_test_encoded, y_pred_cnn)
f1 = f1_score(y_test_encoded, y_pred_cnn)
auc_roc = roc_auc_score(y_test_encoded, y_pred_prob)
tn, fp, fn, tp = confusion_matrix(y_test_encoded, y_pred_cnn).ravel()

logging.info(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-ROC: {auc_roc}, TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}')

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


# Transform the PJZ/PJZC datasets using the same vectorizer
X_pjz_bow = vectorizer.transform(pjz_df['text'])
X_pjzc_bow = vectorizer.transform(pjzc_df['text'])

# Pad the PJZ/PJZC sequences to match the input size expected by the CNN model
X_pjz_padded = pad_sequences(X_pjz_bow.toarray(), maxlen=max_len, padding='post')
X_pjzc_padded = pad_sequences(X_pjzc_bow.toarray(), maxlen=max_len, padding='post')

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
