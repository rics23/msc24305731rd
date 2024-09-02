import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords

nltk.data.path.append('data/nltk/stopwords')

stop_words = set(stopwords.words('english'))


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