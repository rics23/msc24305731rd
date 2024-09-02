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
    - RESULTS:
        Epoch 1/5 - 14119/14119 ━━━━━━━━━━━━━━━━━━━━ 1442s 102ms/step - accuracy: 0.9543 - loss: 0.1861 - val_accuracy: 0.9999 - val_loss: 0.0520
        Epoch 2/5 - 14119/14119 ━━━━━━━━━━━━━━━━━━━━ 1559s 110ms/step - accuracy: 0.9547 - loss: 0.1847 - val_accuracy: 0.9999 - val_loss: 0.0497
        Epoch 3/5 - 14119/14119 ━━━━━━━━━━━━━━━━━━━━ 1497s 106ms/step - accuracy: 0.9548 - loss: 0.1841 - val_accuracy: 0.9999 - val_loss: 0.0440
        Epoch 4/5 - 14119/14119 ━━━━━━━━━━━━━━━━━━━━ 1610s 114ms/step - accuracy: 0.9541 - loss: 0.1864 - val_accuracy: 0.9999 - val_loss: 0.0452
        Epoch 5/5 - 14119/14119 ━━━━━━━━━━━━━━━━━━━━ 1608s 114ms/step - accuracy: 0.9548 - loss: 0.1842 - val_accuracy: 0.9999 - val_loss: 0.0480
                    64337/64337 ━━━━━━━━━━━━━━━━━━━━ 628s 10ms/step
        LSTM Model Accuracy: 1.00
        LSTM Model Precision: 0.00
        LSTM Model Recall: 0.00
        LSTM Model F1 Score: 0.00
        LSTM TP: 0, FN: 185, FP: 0, TN: 2058596

"""

import pandas as pd
from auxiliary import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
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

# LSTM Model
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_padded, y_train_encoded, epochs=5, batch_size=64, validation_data=(X_test_padded, y_test_encoded))

# Predict on test data
y_pred_lstm = (model.predict(X_test_padded) > 0.5).astype(int)

# Evaluate LSTM model
accuracy_lstm = accuracy_score(y_test_encoded, y_pred_lstm)
precision_lstm = precision_score(y_test_encoded, y_pred_lstm)
recall_lstm = recall_score(y_test_encoded, y_pred_lstm)
f1_lstm = f1_score(y_test_encoded, y_pred_lstm)
tn_lstm, fp_lstm, fn_lstm, tp_lstm = confusion_matrix(y_test_encoded, y_pred_lstm).ravel()

print(f'LSTM Model Accuracy: {accuracy_lstm:.2f}')
print(f'LSTM Model Precision: {precision_lstm:.2f}')
print(f'LSTM Model Recall: {recall_lstm:.2f}')
print(f'LSTM Model F1 Score: {f1_lstm:.2f}')
print(f'LSTM TP: {tp_lstm}, FN: {fn_lstm}, FP: {fp_lstm}, TN: {tn_lstm}')
