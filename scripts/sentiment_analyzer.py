import pandas as pd
import os
import logging
from transformers import pipeline
from tqdm import tqdm
import re

class SentimentAnalyzer:
    def __init__(self, input_path: str, output_dir: str = "./data/processed"):
        self.input_path = input_path
        self.output_dir = output_dir
        self.expected_columns = ['review_id', 'review', 'rating', 'date', 'bank', 'source']
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info("Loading DistilBERT sentiment model...")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.misspellings = {'niec': 'nice', 'masha alla': 'mashallah'}
        self.positive_keywords = {'good', 'nice', 'best', 'awesome', 'mashallah', 'great', 'amazing', 'super', 'wow', 'userfriendly'}
        self.negative_keywords = {'issues', 'problem', 'bad', 'worst', 'childish', 'slow', 'not'}

    def preprocess_text(self, text: str) -> str:
        """Correct misspellings and normalize text."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        for wrong, correct in self.misspellings.items():
            text = text.replace(wrong, correct)
        self.logger.debug(f"Preprocessed text: {text[:50]}...")
        return text

    def load_data(self) -> pd.DataFrame:
        """Load input CSV and validate columns."""
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

    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform sentiment analysis with overrides."""
        try:
            sentiments = []
            confidences = []
            for review, rating in tqdm(zip(df['review'].fillna(""), df['rating']), desc="Analyzing sentiments", total=len(df)):
                processed_review = self.preprocess_text(review)
                if not processed_review.strip() or not re.search(r'[a-z]', processed_review):
                    sentiments.append("neutral")
                    confidences.append(0.0)
                    self.logger.debug(f"Neutral (non-text): {review[:50]}...")
                    continue
                result = self.sentiment_pipeline(processed_review, truncation=True, max_length=512)[0]
                label = "positive" if result['label'] == "POSITIVE" else "negative"
                score = result['score']
                # Neutral threshold
                if score < 0.7:
                    label = "neutral"
                    score = 1.0 - score
                # Keyword overrides
                review_lower = processed_review.lower()
                if any(keyword in review_lower for keyword in self.positive_keywords):
                    label = "positive"
                    score = max(score, 0.95)
                    self.logger.debug(f"Positive override: {review[:50]}...")
                elif any(keyword in review_lower for keyword in self.negative_keywords):
                    label = "negative"
                    score = max(score, 0.95)
                    self.logger.debug(f"Negative override: {review[:50]}...")
                # Rating-based correction
                if rating >= 4 and label != "positive":
                    self.logger.debug(f"Rating override (positive, rating={rating}): {review[:50]}...")
                    label = "positive"
                    score = max(score, 0.9)
                elif rating <= 2 and label != "negative":
                    self.logger.debug(f"Rating override (negative, rating={rating}): {review[:50]}...")
                    label = "negative"
                    score = max(score, 0.9)
                elif rating == 3 and score < 0.7:
                    label = "neutral"
                    score = 1.0 - score
                    self.logger.debug(f"Rating override (neutral, rating={rating}): {review[:50]}...")
                sentiments.append(label)
                confidences.append(score)
            df['sentiment'] = sentiments
            df['confidence'] = confidences
            self.logger.info(f"Completed sentiment analysis for {len(df)} reviews")
            return df
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            raise

    def save_results(self, df: pd.DataFrame):
        """Save results to CSV."""
        try:
            output_path = os.path.join(self.output_dir, 'sentiment_results.csv')
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved results to {output_path}")
            sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
            self.logger.info(f"Sentiment distribution:\n{sentiment_counts}")
            # Log mismatches
            mismatches = df[df.apply(lambda x: (x['rating'] >= 4 and x['sentiment'] != 'positive') or (x['rating'] <= 2 and x['sentiment'] != 'negative'), axis=1)]
            if not mismatches.empty:
                self.logger.warning(f"Found {len(mismatches)} rating-sentiment mismatches:\n{mismatches[['review_id', 'review', 'rating', 'sentiment', 'confidence']].head()}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise

    def run(self):
        """Execute full sentiment analysis pipeline."""
        df = self.load_data()
        df = self.analyze_sentiment(df)
        self.save_results(df)
        return df

if __name__ == "__main__":
    analyzer = SentimentAnalyzer(input_path="./data/processed/all_reviews.csv")
    try:
        results = analyzer.run()
    except Exception as e:
        print(f"Error running sentiment analysis: {e}")
