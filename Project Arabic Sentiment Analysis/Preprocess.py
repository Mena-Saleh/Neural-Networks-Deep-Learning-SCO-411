import numpy as np
import pandas as pd
from langdetect import detect
from nltk.tokenize import ToktokTokenizer
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
import re
import nltk
import pyarabic.araby as ar
import re , emoji, Stemmer, functools, operator, string
from nltk.stem.isri import ISRIStemmer
#nltk.download('wordnet')
#nltk.download('stopwords')


# NLP pipeline classes for Arabic and English

class ArabicPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('arabic'))
        self.st =  Stemmer.Stemmer('arabic')
        self.snowball_stmmer = SnowballStemmer("arabic")
        self.isris_stemmer = ISRIStemmer()
        
    def preprocess(self, text):
        # Replace one or more whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text)

        # Normalize certain Arabic characters to their most common forms and remove unnecessary artificats like tashkeel and tatweel
        text = text.strip()
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)

        #remove repetetions
        text = re.sub("[إأٱآا]", "ا", text)
        text = text.replace('وو', 'و')
        text = text.replace('يي', 'ي')
        text = text.replace('ييي', 'ي')
        text = text.replace('اا', 'ا')

        # Remove longation
        text = re.sub(r'(.)\1+', r"\1\1", text) 
        
        # Remove tashkeel and tatweel
        text = ar.strip_tashkeel(text)
        text = ar.strip_tatweel(text)

        # Remove punctuation marks
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Reduce characters that appear more than twice in a row to a single character
        text = re.sub(r'(.)\1+', r'\1', text)

        # Remove stop words
        text = ' '.join([word for word in text.split() if word.lower() not in self.stop_words])
        
        # Stem the words in the text (Try different stemmers)
        
        # Stemmer library
        text = " ".join([self.st.stemWord(word) for word in text.split()])
        
        # Snowball stemmer
        #text = " ".join([self.snowball_stmmer.stem(word) for word in text.split()])
        
        # Isris stemmer
        # text = self.isris_stemmer.stem(text)
        # text = self.isris_stemmer.pre32(text)
        # text = self.isris_stemmer.suf32(text)
        
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
def preprocess_df(df, out_name, num_samples=-1, isPredict = False):
    
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
    
    # Preprocess target values and save preprocessed data (only if data is labeled, i.e in training and testing, not prediction)
    if not isPredict:
        # Map sentiments from -1,0, 0 to 1, 1 ,2
        df['rating'] = df['rating'].map(label_mapping)
    
        # Save to Excel
        df.to_excel(f'preprocessed_{out_name}.xlsx', index=False)
    
    return df


