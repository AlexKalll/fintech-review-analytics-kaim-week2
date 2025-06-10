import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

class KeywordExtractor:
    def __init__(self, input_path: str, output_dir: str = "./data/processed"):
        self.input_path = input_path
        self.output_dir = output_dir
        self.expected_columns = ['review_id', 'review', 'rating', 'date', 'bank', 'source', 'sentiment', 'confidence', 'theme']
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.output_dir, exist_ok=True)
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english')) - {'best', 'good', 'nice', 'bad', 'worst'}
        self.lemmatizer = WordNetLemmatizer()
        self.custom_phrases = {
            'masha alla': 'mashallah',
            'dashen super app': 'dashensuperapp',
            'slow loading': 'slowloading',
            'user friendly': 'userfriendly',
            'best app': 'bestapp',
            'childish app': 'childishapp',
            'awesome bank': 'awesomebank',
            'not work': 'notwork',
            'can t': 'cannot'
        }

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        for phrase, replacement in self.custom_phrases.items():
            text = text.lower().replace(phrase.lower(), replacement)
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.input_path)
            missing_cols = [col for col in self.expected_columns if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing columns: {missing_cols}")
                raise ValueError(f"Missing columns: {missing_cols}")
            self.logger.info(f"Loaded {len(df)} reviews from {self.input_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def extract_keywords(self, df: pd.DataFrame) -> dict:
        try:
            keywords_by_bank = {}
            vectorizer = TfidfVectorizer(max_features=20, ngram_range=(1, 2), stop_words=list(self.stop_words))
            for bank in df['bank'].unique():
                bank_reviews = df[df['bank'] == bank]['review'].fillna('')
                processed_texts = [self.preprocess_text(review) for review in bank_reviews]
                valid_texts = [text for text in processed_texts if text.strip()]
                if not valid_texts:
                    self.logger.warning(f"No valid texts for bank: {bank}")
                    keywords_by_bank[bank] = []
                    continue
                tfidf_matrix = vectorizer.fit_transform(valid_texts)
                feature_names = vectorizer.get_feature_names_out()
                keywords = []
                for doc_idx in range(tfidf_matrix.shape[0]):
                    doc_scores = tfidf_matrix[doc_idx].toarray()[0]
                    top_indices = doc_scores.argsort()[-5:][::-1]
                    doc_keywords = [feature_names[idx] for idx in top_indices if doc_scores[idx] > 0]
                    keywords.extend(doc_keywords)
                keywords = list(dict.fromkeys(keywords))[:10]  # Top 10 unique
                keywords_by_bank[bank] = keywords
                self.logger.info(f"Keywords for {bank}: {keywords}")
            return keywords_by_bank
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            raise

    def save_keywords(self, keywords_by_bank: dict):
        try:
            output_path = os.path.join(self.output_dir, 'keywords_by_bank.csv')
            data = [(bank, keyword) for bank, keywords in keywords_by_bank.items() for keyword in keywords]
            df = pd.DataFrame(data, columns=['bank', 'keyword'])
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved keywords to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving keywords: {e}")
            raise

    def run(self):
        df = self.load_data()
        keywords_by_bank = self.extract_keywords(df)
        self.save_keywords(keywords_by_bank)
        return keywords_by_bank

if __name__ == "__main__":
    extractor = KeywordExtractor(input_path="./data/processed/thematic_results.csv")
    try:
        results = extractor.run()
    except Exception as e:
        print(f"Error running keyword extraction: {e}")