import pandas as pd
import re
import string
from textblob import TextBlob
import nltk

nltk.download('punkt')
nltk.download('wordnet')

class ReviewCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def basic_clean(self, text):
        if pd.isna(text):
            return ""
        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_polarity(self, text):
        return TextBlob(text).sentiment.polarity

    def clean(self):
        self.df['clean_review'] = self.df['review'].apply(self.basic_clean)
        self.df = self.df[self.df['clean_review'].str.strip() != ""]
        self.df['sentiment'] = self.df['clean_review'].apply(self.get_polarity)
        print(f"Dropped {(1 - len(self.df)/len(self.df.index))*100:.2f}% rows with empty reviews")
        return self.df.reset_index(drop=True)
