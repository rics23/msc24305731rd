import os
import pickle
import pandas as pd
from auxiliary import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, LSTM, Bidirectional, SpatialDropout1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
from sklearn.model_selection import GridSearchCV

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

    # Extract labels
    y_train = pan_train_df['label']
    y_test = pan_test_df['label']

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Prepare input data for hybrid CNN-LSTM
    max_features = 1000
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(pan_train_df['text'])
    X_test_tfidf = vectorizer.transform(pan_test_df['text'])

    # Apply dimensionality reduction
    embedding_dim = 100
    glove_path = 'data/glove.6B/glove.6B.100d.txt'
    embedding_matrix = load_glove_embeddings(glove_path, vectorizer.vocabulary_, embedding_dim)
    np.save(embedding_matrix_path, embedding_matrix)

    # Apply SMOTE to balance classes
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
print("Defining and compiling the hybrid CNN-LSTM model")
input_shape = X_train_combined.shape[1]

model = Sequential()
print("Embedding")
model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_dim, input_length=input_shape, weights=[embedding_matrix], trainable=False))
print("SpatialDropout1D")
model.add(SpatialDropout1D(0.2))
print("Conv1D")
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
print("MaxPooling1D")
model.add(MaxPooling1D(pool_size=5))
print("Bidirectional")
model.add(Bidirectional(LSTM(64, return_sequences=True)))
print("GlobalMaxPooling1D")
model.add(GlobalMaxPooling1D())
print("Dense relu")
model.add(Dense(64, activation='relu'))
print("Dropout")
model.add(Dropout(0.5))
print("Dense sigmoid")
model.add(Dense(1, activation='sigmoid'))

print("Compiling the model")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to avoid overfitting
print("Setting up early stopping")
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the hybrid model
print("Training the hybrid model")
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
