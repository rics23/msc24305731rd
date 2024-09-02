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
    - Trains a Convolutional Neural Network on the training data.
    - Applies dropout techniques to prevent overfitting and improve generalisation.

5. Model Evaluation:
    - Applies the trained model to the test data.
    - Evaluates the model's performance using accuracy, precision, recall, F1 score, and confusion matrix metrics.
    - RESULTS:
        Epoch 1/5 - 14119/14119 ━━━━━━━━━━━━━━━━━━━━ 702s 49ms/step - accuracy: 0.9542 - loss: 0.1937 - val_accuracy: 0.9999 - val_loss: 0.0575
        Epoch 2/5 - 14119/14119 ━━━━━━━━━━━━━━━━━━━━ 635s 45ms/step - accuracy: 0.9545 - loss: 0.1860 - val_accuracy: 0.9999 - val_loss: 0.0494
        Epoch 3/5 - 14119/14119 ━━━━━━━━━━━━━━━━━━━━ 634s 45ms/step - accuracy: 0.9547 - loss: 0.1846 - val_accuracy: 0.9999 - val_loss: 0.0438
        Epoch 4/5 - 14119/14119 ━━━━━━━━━━━━━━━━━━━━ 636s 45ms/step - accuracy: 0.9544 - loss: 0.1853 - val_accuracy: 0.9999 - val_loss: 0.0466
        Epoch 5/5 - 14119/14119 ━━━━━━━━━━━━━━━━━━━━ 639s 45ms/step - accuracy: 0.9548 - loss: 0.1842 - val_accuracy: 0.9999 - val_loss: 0.0476
                    64337/64337 ━━━━━━━━━━━━━━━━━━━━ 128s 2ms/step
        CNN Model Accuracy: 1.00
        CNN Model Precision: 0.00
        CNN Model Recall: 0.00
        CNN Model F1 Score: 0.00
        CNN TP: 0, FN: 185, FP: 0, TN: 2058596
"""

import pandas as pd
from auxiliary import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Parse the training and testing XML files
pan_training_conversations = parse_conversations('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml')
pan_test_conversations = parse_conversations('data/pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')

# Load the list of predators
with open('data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt', 'r') as f:
    pan_predator_ids = set(f.read().splitlines())

# Label the training data
pan_training_data = label_messages(pan_training_conversations, pan_predator_ids)

# Convert to DataFrame
pan_train_df = pd.DataFrame(pan_training_data)

# Fill NaN values in the 'text' column with empty strings
pan_train_df['text'] = pan_train_df['text'].fillna('')

# Preprocess training data
pan_train_df['text'] = pan_train_df['text'].apply(preprocess_text)

# Preprocess test data
pan_test_data = label_messages(pan_test_conversations, pan_predator_ids)
pan_test_df = pd.DataFrame(pan_test_data)
pan_test_df['text'] = pan_test_df['text'].fillna('')
pan_test_df['text'] = pan_test_df['text'].apply(preprocess_text)

# Extract labels
y_train = pan_train_df['label']
y_test = pan_test_df['label']

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Prepare input data for LSTM
max_features = 10000
max_len = 100

vectorizer = CountVectorizer(max_features=max_features)
X_train_bow = vectorizer.fit_transform(pan_train_df['text'])
X_test_bow = vectorizer.transform(pan_test_df['text'])

# Pad sequences to ensure uniform input size
X_train_padded = pad_sequences(X_train_bow.toarray(), maxlen=max_len)
X_test_padded = pad_sequences(X_test_bow.toarray(), maxlen=max_len)

# CNN Model
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_len))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the CNN model
model.fit(X_train_padded, y_train_encoded, epochs=5, batch_size=64, validation_data=(X_test_padded, y_test_encoded))

# Predict on the test set
y_pred_cnn = (model.predict(X_test_padded) > 0.5).astype(int)

# Evaluate the CNN model
accuracy_cnn = accuracy_score(y_test_encoded, y_pred_cnn)
precision_cnn = precision_score(y_test_encoded, y_pred_cnn)
recall_cnn = recall_score(y_test_encoded, y_pred_cnn)
f1_cnn = f1_score(y_test_encoded, y_pred_cnn)
tn_cnn, fp_cnn, fn_cnn, tp_cnn = confusion_matrix(y_test_encoded, y_pred_cnn).ravel()

print(f'CNN Model Accuracy: {accuracy_cnn:.2f}')
print(f'CNN Model Precision: {precision_cnn:.2f}')
print(f'CNN Model Recall: {recall_cnn:.2f}')
print(f'CNN Model F1 Score: {f1_cnn:.2f}')
print(f'CNN TP: {tp_cnn}, FN: {fn_cnn}, FP: {fp_cnn}, TN: {tn_cnn}')

