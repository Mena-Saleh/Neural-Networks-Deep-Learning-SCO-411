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
        text = text.strip()
        
        # Remove elongation
        text = re.sub(r'(.)\1+', r"\1", text) 
        
        # Normalize letters
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("[إأٱآا]", "ا", text)

        # Remove repetetions
        text = text.replace('وو', 'و')
        text = text.replace('يي', 'ي')
        text = text.replace('ييي', 'ي')
        text = text.replace('اا', 'ا')
        
    
        # Remove tashkeel and tatweel
        text = ar.strip_tashkeel(text)
        text = ar.strip_tatweel(text)

        # Remove punctuation marks
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        
        # Replacing with synonyms and replacing expressions with meaningful words
        
        # Words or expressions associated with a good review
        text = text.replace('ما شاء الله', 'ممتاز')
        text = text.replace('ما شاء اله', 'ممتاز')
        text = text.replace('احلا', 'ممتاز')
        text = text.replace('احلي', 'ممتاز')
        text = text.replace('اخلي', 'ممتاز')
        text = text.replace('احله', 'ممتاز')
        text = text.replace('طيب', 'ممتاز')
        text = text.replace('حلو', 'ممتاز')
        text = text.replace('يجن', 'ممتاز')
        text = text.replace('يجنن', 'ممتاز')
        text = text.replace('افضل', 'ممتاز')
        text = text.replace('wow', 'ممتاز')
        text = text.replace('سريع', 'ممتاز')        
        text = text.replace('كويسة', 'ممتاز')
        text = text.replace('كويس', 'ممتاز')
        text = text.replace('توفيق', 'ممتاز')
        text = text.replace('جميل', 'ممتاز')
        text = text.replace('جميلة', 'ممتاز')
        text = text.replace('سهل', 'ممتاز')
        text = text.replace('شكر', 'ممتاز')
        text = text.replace('شكرا', 'ممتاز')
        text = text.replace('جيد', 'ممتاز')
        text = text.replace('روعه', 'ممتاز')
        
        text = text.replace('نظيف', 'ممتاز')
        text = text.replace('اقوى', 'ممتاز')
        text = text.replace('قوى', 'ممتاز')
        text = text.replace('منتاز', 'ممتاز')
        text = text.replace('perfect', 'ممتاز')



        # Words associated usually with a bad review (also some of them are removed as stop words but they are useful)
        text = text.replace('bad', 'زفت')
        text = text.replace('فاشل', 'زفت')
        text = text.replace('خيس', 'زفت')
        text = text.replace('عسير', 'زفت')
        text = text.replace('مشكله', 'زفت')
        text = text.replace('مشاكل', 'زفت')
        text = text.replace('عسير', 'زفت')
        text = text.replace('نصب', 'زفت')
        text = text.replace('احتيال', 'زفت')
        text = text.replace('صعب', 'زفت')
        text = text.replace('يلع', 'زفت')
        text = text.replace('يلعن', 'زفت')
        text = text.replace('يتأخر', 'زفت')
        text = text.replace('بايخ', 'زفت')
        text = text.replace('اسوء', 'زفت')
        text = text.replace('مفيش', 'زفت')
        text = text.replace('تفو', 'زفت')
        
        text = text.replace('ضغيف', 'زفت')
        text = text.replace('لا يعمل', 'زفت')
        text = text.replace('ذق', 'زفت')
        text = text.replace('يسرق', 'زفت')
        text = text.replace('يلغي', 'زفت')
        text = text.replace('لصوص', 'زفت')
        text = text.replace('خايس', 'زفت')
        text = text.replace('افشل', 'زفت')
        text = text.replace('غبي', 'زفت')
        text = text.replace('للاسف', 'زفت')
        text = text.replace('خرا', 'زفت')
        text = text.replace('قذر', 'زفت')
        text = text.replace('قزر', 'زفت')
        text = text.replace('اي كلام', 'زفت')
        text = text.replace('بخزى', 'زفت')
        text = text.replace('منهم لله', 'زفت')
        text = text.replace('khara', 'زفت')  
        text = text.replace('طظ', 'زفت')  



        # replacing most common emojis with equivalent words
        text = text.replace('👍', 'ممتاز')
        text = text.replace('😘', 'ممتاز')        
        text = text.replace('👌', 'ممتاز')
        text = text.replace('😍', 'ممتاز')
        text = text.replace('😊', 'ممتاز')
        text = text.replace('💙', 'ممتاز')
        text = text.replace('💕', 'ممتاز')
        text = text.replace('💜', 'ممتاز')
        text = text.replace('👏', 'ممتاز')
        text = text.replace('😋', 'ممتاز')
        text = text.replace('😁', 'ممتاز')
        text = text.replace('🥰', 'ممتاز')
        text = text.replace('🤩', 'ممتاز')
        text = text.replace('🔥', 'ممتاز')
        text = text.replace('💞', 'ممتاز')
        text = text.replace('🤗', 'ممتاز')
        text = text.replace('😉', 'ممتاز')
        text = text.replace('💓', 'ممتاز')
        text = text.replace('💋', 'ممتاز')
        text = text.replace('💛', 'ممتاز')
        text = text.replace('💗', 'ممتاز')
        text = text.replace('🖒', 'ممتاز')
        text = text.replace('💖', 'ممتاز')

        text = text.replace('🤬', 'زفت')
        text = text.replace('😤', 'زفت')
        text = text.replace('😒', 'زفت')
        text = text.replace('😢', 'زفت')
        text = text.replace('😭', 'زفت')
        text = text.replace('😠', 'زفت')
        text = text.replace('😡', 'زفت')
        text = text.replace('👎', 'زفت')
    
        
        # # Remove stop words (لا و غير بيغيرو معني الجملة)
        # words_to_remove = {'لا', 'غير', 'ما', 'لم'}

        # # Remove the specified words from the set of Arabic stopwords
        # arabic_stopwords = self.stop_words - words_to_remove
        # text = ' '.join([word for word in text.split() if word.lower() not in arabic_stopwords])
        
        # Stem the words in the text (Try different stemmers)
        
        # Stemmer library
        text = " ".join([self.st.stemWord(word) for word in text.split()])
        
        # Snowball stemmer
        #text = " ".join([self.snowball_stmmer.stem(word) for word in text.split()])
        
        # Isris stemmer
        # text = self.isris_stemmer.stem(text)
        # text = self.isris_stemmer.pre32(text)
        # text = self.isris_stemmer.suf32(text)
        
        # More synonym replacements after stemming
        
        # Positive reviews
        text = text.replace('5', 'ممتاز')
        text = text.replace('روعه', 'ممتاز')
        text = text.replace('روع', 'ممتاز')

        
        # Negative synonyms
        text = text.replace('سيء', 'زفت')  
        text = text.replace('حرام', 'زفت') 
        text = text.replace('0', 'زفت')  
        text = text.replace('مقرف', 'زفت')  

        # Remove words that make conflicts
        text = text.replace('جدا', '')  
        

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


