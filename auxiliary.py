import numpy as np
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
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

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='data/nltk')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir='data/nltk')

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


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

    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word.lower() not in stop_words]

    return ' '.join(tokens)


def preprocess_text_stemmed(text):

    if text is None:
        return ""

    # tokens = nltk.word_tokenize(text.lower())
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]

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


# Behavioral Feature Extraction
def extract_behavioral_features(df):
    df['msg_length'] = df['text'].apply(len)
    return df[['msg_length']]