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
import pyarabic.araby as ar
# Use pip install emoji==1.4.1 when installing emoji, newer versions are missing the get_emoji_regexp function
import re , emoji, Stemmer, functools, operator, string
from nltk.stem.isri import ISRIStemmer
from sklearn.model_selection import train_test_split
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
        self.st =  Stemmer.Stemmer('arabic')
        
    def preprocess(self, text):
        # Replace one or more whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text)

        # Remove all standalone numbers - these regex patterns match numbers that are on their own
        text = re.sub("(\s\d+)","",text) 
        text = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", text)
        text = re.sub("\d+", " ", text)

        # Remove Arabic diacritics (tashkeel) and letter elongation (tatweel)
        text = ar.strip_tashkeel(text)
        text = ar.strip_tatweel(text)

        # Remove punctuation marks
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        # Split emojis from other text and rejoin without spaces
        em = text
        em_split_emoji = emoji.get_emoji_regexp().split(em)
        em_split_whitespace = [substr.split() for substr in em_split_emoji]
        em_split = functools.reduce(operator.concat, em_split_whitespace)
        text = " ".join(em_split)

        # Reduce characters that appear more than twice in a row to a single character
        text = re.sub(r'(.)\1+', r'\1', text)

        # Remove stop words
        words = text.split()  # Tokenize the text into words
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        text =  ' '.join(filtered_words)  # Join words back into a string        
        
        # Stem the words in the text
        text = " ".join([self.st.stemWord(i) for i in text.split()])

        # Normalize certain Arabic characters to their most common forms
        text = text.replace("آ", "ا")
        text = text.replace("إ", "ا")
        text = text.replace("أ", "ا")
        text = text.replace("ؤ", "و")
        text = text.replace("ئ", "ي")
        

        return text
        
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



label_mapping = {-1: 0, 0: 1, 1: 2}

# Main function (Optionally define number of samples to preprocess, to process smaller batches for testing if required)
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
         
    # Map sentiments from -1,0, 0 to 1, 1 ,2
    df['rating'] = df['rating'].map(label_mapping)
    
    # Save to Excel
    df.to_excel(f'preprocessed_{out_name}.xlsx', index=False)
    
    return df


