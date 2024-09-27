"""
Project:    Advancements in Identifying Online Grooming: A Critical Analysis of Machine Learning Methodologies
            Edge Hill University - CIS4114 - MSc Research & Development Project
Author:     Mr Ricardo Lopes (24305731@edgehill.ac.uk)
Created:    2024-08-30
Version:    1.0
Ownership:  Mr Ricardo Lopes (24305731@edgehill.ac.uk)
License:    MIT License
Filename:   auxiliary.py

This script provides auxiliary functions for processing, parsing, and analysing chat conversations used in the identification
of online grooming. It includes the following key components:

1. XML Parsing and Data Labeling:
    - parse_conversations: Parses XML-formatted chat conversations (from the PAN2012 dataset), extracting messages, authors,
      and conversation IDs.
    - label_messages: Labels each message based on the author, using a list of known predator IDs to distinguish between
      predator and non-predator messages.

2. Text Preprocessing:
    - preprocess_text: Preprocesses the text data by converting it to lowercase, removing stopwords and punctuation, and
      lemmatizing the words using spaCy. This prepares the text for vectorization and model training.

3. Embedding Layer Setup:
    - load_glove_embeddings: Loads pre-trained GloVe embeddings and creates an embedding matrix for the words in the
      dataset, which will be used for model training.

4. Behavioral Feature Extraction:
    - extract_behavioral_features: Extracts basic behavioral features, such as message length, to augment the textual data
      with user behavior-based insights.

5. Parsing and Labeling for PJZ/PJZC Datasets:
    - parse_pj_dataset: Parses chat conversations in JSON format from additional datasets (PJZ and PJZC), extracting messages,
      conversation metadata, and labeling the data (1 for groomer, 0 for non-groomer).
    - label_pj_messages: Labels individual messages based on conversation-level grooming labels.

"""

import json
import numpy as np
import spacy
import ssl
import xml.etree.ElementTree as ET
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments.log'),
        logging.StreamHandler()
    ]
)

ssl._create_default_https_context = ssl._create_unverified_context

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")


def parse_conversations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    conversations = []

    for conversation in root.findall('conversation'):
        conversation_id = conversation.get('id')
        messages = []

        for message in conversation.findall('message'):
            line_number = message.get('line')
            author_id = message.find('author').text
            text = message.find('text').text

            messages.append({
                'line': line_number,
                'author': author_id,
                'text': text
            })

        conversations.append({
            'id': conversation_id,
            'messages': messages
        })

    return conversations


def label_messages(conversations, predator_ids):
    labeled_data = []

    for conversation in conversations:
        for message in conversation['messages']:
            author_id = message['author']
            text = message['text']
            label = 1 if author_id in predator_ids else 0  # 1 for predator, 0 for non-predator

            labeled_data.append({
                'author': author_id,
                'text': text,
                'label': label
            })

    return labeled_data


def preprocess_text(text):
    if text is None:
        return ""

    # Process the text with spaCy
    doc = nlp(text.lower())

    # Remove stopwords and punctuation, and use lemmatized tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return ' '.join(tokens)


# Load pre-trained GloVe embeddings
def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# Behavioural Feature Extraction
def extract_behavioral_features(df):
    df['msg_length'] = df['text'].apply(len)
    return df[['msg_length']]


# Parse PJZ/PJZC JSON formatted datasets
def parse_pj_dataset(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    conversations = []

    for conv in data['conversation']:
        conversation_id = conv['id']
        source = conv['source']
        label = int(conv['label'])  # 1 for groomer, 0 for non-groomer
        messages = []

        for message in conv['messages']:
            author = message['author']
            time = message['time']
            text = message['text']
            messages.append({
                'author': author,
                'time': time,
                'text': text
            })

        conversations.append({
            'id': conversation_id,
            'source': source,
            'label': label,
            'messages': messages
        })

    return conversations


def label_pj_messages(conversations):
    labeled_data = []

    for conversation in conversations:
        label = conversation['label']  # 1 for groomer, 0 for non-groomer
        for message in conversation['messages']:
            author = message['author']
            text = message['text']
            labeled_data.append({
                'author': author,
                'text': text,
                'label': label
            })

    return labeled_data


