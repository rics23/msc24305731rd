import pandas as pd
import xml.etree.ElementTree as ET

training_file = 'data/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'

tree = ET.parse(training_file)
root = tree.getroot()

data = []

for conversation in root.findall('.//conversation'):
    conversation_id = conversation.get('id')
    for message in conversation.findall('message'):
        message_line = message.get('line')
        author = message.find('author').text
        time = message.find('time').text
        text = message.find('text').text
        data.append([conversation_id, message_line, author, time, text])

df = pd.DataFrame(data, columns=['conversation_id', 'message_line', 'author', 'time', 'text'])

print(df.head())
