# Advancements in Identifying Online Grooming: A Critical Analysis of Machine Learning Methodologies

## Overview
This project is part of an MSc in Big Data Analytics at Edge Hill University. It focuses on exploring and developing machine learning methodologies to identify online grooming behaviors in chat conversations. By leveraging multiple datasets and advanced natural language processing (NLP) techniques, the project aims to critically assess and improve upon existing models used for online grooming identification.

The main objective is to preprocess, analyse, and extract features from chat conversations using XML and JSON datasets. This project integrates various external resources, such as spaCy language models and GloVe embeddings, to aid in the development of machine learning models capable of detecting grooming behaviors.

## Datasets
The project uses three key datasets:

1. **PAN12 Dataset**: 
   - [Download PAN12 Dataset from Zenodo](https://zenodo.org/records/3713280).
   - This dataset is part of the PAN 2012 competition for sexual predator identification. It includes XML-formatted chat conversations, as well as ground truth files for training and testing.

2. **PJZ and PJZC Datasets**:
   - [Download PJZ and PJZC Datasets from GitHub](https://github.com/danielafe7-usp/BF-PSR-Framework/tree/master/JsonData).
   - These datasets are provided in JSON format and contain labeled chat conversations for identifying grooming behaviors.
     - You need to **manually extract** `PJZ.txt` and `PJZC.txt` from the `JsonData` folder in the GitHub repository.
     - Store these files in the `data/pj/` directory as `PJZ.txt` and `PJZC.txt`.

3. **Pre-trained GloVe Embeddings**:
   - [Download GloVe Embeddings from Stanford NLP](https://nlp.stanford.edu/data/glove.6B.zip).
   - This project uses pre-trained word vectors (GloVe) to enhance the text representation in machine learning models. Multiple dimensions of embeddings are available (50d, 100d, 200d, 300d). This project utilised 100d.

## Requirements

To run the experiments, you will need to install the following dependencies:

- **Python 3.12** (recommended)
- **PyCharm Professional** (optional but recommended for development)
- **spaCy NLP Library** (essential)

### Installation

1. Clone this repository.
2. Ensure you have **Python 3.12** installed.
3. Install the required packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   
4. Download the small English language model with the following command:
   ```bash
   python -m spacy download en_core_web_sm
   
## Data Directory Structure
```
data/
├── glove.6B/
│   ├── glove.6B.50d.txt
│   ├── glove.6B.100d.txt
│   ├── glove.6B.200d.txt
│   ├── glove.6B.300d.txt
├── pan12-sexual-predator-identification-test-corpus-2012-05-21/
│   ├── pan12-sexual-predator-identification-test-corpus-2012-05-17.xml
│   ├── pan12-sexual-predator-identification-groundtruth-problem1.txt
│   ├── pan12-sexual-predator-identification-groundtruth-problem2.txt
│   ├── readme.txt
├── pan12-sexual-predator-identification-training-corpus-2012-05-01/
│   ├── pan12-sexual-predator-identification-training-corpus-2012-05-01.xml
│   ├── pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt
│   ├── readme.txt
├── pj/
│   ├── PJZ.txt
│   ├── PJZC.txt
```
