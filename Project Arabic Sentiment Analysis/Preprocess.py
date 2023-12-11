import numpy as np
import pandas as pd
from langdetect import detect
from nltk.tokenize import ToktokTokenizer
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from farasa.stemmer import FarasaStemmer
from farasa.segmenter import FarasaSegmenter
from farasa.pos import FarasaPOSTagger
from farasa.ner import FarasaNamedEntityRecognizer
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')


# NLP pipeline classes for Arabic and English

class ArabicPreprocessor:
    def __init__(self):
        self.segmenter = FarasaSegmenter()
        self.stemmer = FarasaStemmer()
        self.postagger = FarasaPOSTagger()
        self.ner = FarasaNamedEntityRecognizer()
        self.stop_words = set(stopwords.words('arabic'))
        
    def preprocess(self, text):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize and segment the text
        tokens = self.segmenter.segment(text).split()
        
        # Apply stemming/lemmatization to the tokens
        stemmed_words = [self.stemmer.stem(token) for token in tokens]
        
        # Remove stop words
        filtered_tokens = [token for token in stemmed_words if token not in self.stop_words]
        
        # Remove named entities
        named_entities = self.ner.recognize(text)
        filtered_tokens = [token for token in filtered_tokens if token not in named_entities]

        # Rejoin tokens into a string
        return ' '.join(filtered_tokens)

class EnglishPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer=ToktokTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def preprocess(self, text):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize text and transform to lower cass
        tokens = word_tokenize(text.lower())

        # Remove stop words
        filtered_tokens = [token for token in tokens if token not in self.stop_words]

        # Lemmatize/stem
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Rejoin tokens into a string
        return ' '.join(lemmatized_tokens)


# Main function
def preprocess_df(df, out_name, num_samples=-1):
    # If num_rows is not -1 and less than the total rows, randomly sample the specified number of rows
    if num_samples != -1 and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)  # random_state ensures reproducibility

    # Preprocess English and Arabic text data
    ar_pre = ArabicPreprocessor()
    en_pre = EnglishPreprocessor()
    
    # List for processed text
    processed_texts = []
    
    # Iterate over text and preprocess according to language
    for text in df['review_description']:
        print(text)
        try:
            lang = detect(text)
        except:
            # If language detection fails, default to Arabic
            lang = 'ar'
 
        if lang == 'en':
            processed_text = en_pre.preprocess(text)
        else:
            processed_text = ar_pre.preprocess(text)
         
        processed_texts.append(processed_text)
    
    # Replace text with preprocessed version
    df['review_description'] = processed_texts
    
    # Save to Excel
    df.to_excel(f'preprocessed_{out_name}.xlsx', index=False)
    
    return df


