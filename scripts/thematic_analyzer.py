import pandas as pd
import os
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm
import re
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

class ThematicAnalyzer:
    def __init__(self, input_path: str, output_dir: str = "./data/processed", num_topics: int = 3):
        self.input_path = input_path
        self.output_dir = output_dir
        self.num_topics = num_topics
        self.expected_columns = ['review_id', 'review', 'rating', 'date', 'bank', 'source', 'sentiment', 'confidence']
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.output_dir, exist_ok=True)
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english')) - {'best', 'good', 'nice', 'bad', 'worst'}
        self.lemmatizer = WordNetLemmatizer()
        self.theme_mappings = {}
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
        self.technical_keywords = {'slowloading', 'notwork', 'crash', 'uninstall', 'problem', 'issue', 'loading', 'bugs', 'lag', 'error'}
        self.security_keywords = {'safety', 'security', 'safe', 'not secure'}
        self.negative_keywords = {'bad', 'worst', 'childishapp', 'boring', 'unreliable'}
        self.feature_keywords = {'add', 'suggest', 'improve', 'include', 'upgrade'}
        self.service_keywords = {'service', 'support', 'help', 'customer'}

    def preprocess_text(self, text: str) -> list:
        if not isinstance(text, str) or not text.strip():
            self.logger.debug(f"Skipped empty text: {text[:50]}...")
            return []
        for phrase, replacement in self.custom_phrases.items():
            text = text.lower().replace(phrase.lower(), replacement)
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        self.logger.debug(f"Processed text: {text[:50]}... -> {tokens}")
        return tokens

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

    def train_lda(self, texts: list):
        try:
            if not texts or all(not t for t in texts):
                self.logger.error("No valid texts for LDA training")
                raise ValueError("Empty corpus")
            dictionary = corpora.Dictionary(texts)
            dictionary.filter_extremes(no_below=5, no_above=0.5)
            corpus = [dictionary.doc2bow(text) for text in texts]
            self.logger.info(f"Training LDA model with {self.num_topics} topics...")
            lda_model = LdaModel(corpus, num_topics=self.num_topics, id2word=dictionary, passes=30, alpha='auto', eta='auto', random_state=42)
            coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_score = coherence_model.get_coherence()
            self.logger.info(f"LDA Coherence Score: {coherence_score}")
            return lda_model, dictionary, corpus
        except Exception as e:
            self.logger.error(f"Error training LDA: {e}")
            raise

    def inspect_topics(self, lda_model):
        self.logger.info("=== Inspecting Topics ===")
        for idx, topic in lda_model.print_topics():
            self.logger.info(f"Topic {idx}: {topic}")

    def map_topics_to_themes(self, lda_model):
        topics = lda_model.print_topics()
        self.theme_mappings = {}
        for idx, topic in topics:
            words = [word.split('*')[1].strip('"') for word in topic.split(' + ')]
            if any(w in ['bestapp', 'awesomebank', 'userfriendly', 'good', 'nice'] for w in words):
                self.theme_mappings[idx] = 'Positive User Experience'
            elif any(w in ['slowloading', 'notwork', 'bugs', 'crash', 'uninstall'] for w in words):
                self.theme_mappings[idx] = 'Technical Issues'
            elif any(w in ['security', 'safety', 'secure'] for w in words):
                self.theme_mappings[idx] = 'Security Issues'
            else:
                self.theme_mappings[idx] = 'Positive User Experience'  # Default
        unmapped = set(range(self.num_topics)) - set(self.theme_mappings.keys())
        if unmapped:
            self.logger.warning(f"Unmapped topics: {unmapped}")
        self.logger.info(f"Theme mappings: {self.theme_mappings}")

    def assign_themes(self, df: pd.DataFrame, lda_model, corpus):
        try:
            themes = []
            probabilities = []
            for idx, (doc_bow, sentiment, confidence, review, tokens) in enumerate(tqdm(
                zip(corpus, df['sentiment'], df['confidence'], df['review'].fillna(''), [self.preprocess_text(r) for r in df['review'].fillna('')]),
                total=len(df), desc="Assigning themes"
            )):
                review_lower = review.lower()
                # Non-text or neutral sentiment
                if not tokens or not re.search(r'[a-z]', review_lower) or len(tokens) < 2 or sentiment == 'neutral':
                    theme = 'none'
                    prob = 0.0
                    self.logger.debug(f"No theme (neutral/non-text): {review[:50]}...")
                else:
                    # Keyword-based assignment
                    if sentiment == 'positive':
                        if any(keyword in review_lower for keyword in self.feature_keywords):
                            theme = 'Feature Request'
                            prob = max(confidence, 0.95)
                        elif any(keyword in review_lower for keyword in self.service_keywords):
                            theme = 'Customer Service'
                            prob = max(confidence, 0.95)
                        else:
                            theme = 'Positive User Experience'
                            prob = confidence
                    elif sentiment == 'negative':
                        if any(keyword in review_lower for keyword in self.technical_keywords):
                            theme = 'Technical Issues'
                            prob = max(confidence, 0.95)
                        elif any(keyword in review_lower for keyword in self.security_keywords):
                            theme = 'Security Issues'
                            prob = max(confidence, 0.95)
                        else:
                            theme = 'Negative User Experience'
                            prob = max(confidence, 0.95)
                    # LDA-based assignment for longer reviews
                    if len(tokens) >= 2 and doc_bow:
                        topic_dist = lda_model[doc_bow]
                        dominant_topic, dominant_prob = max(topic_dist, key=lambda x: x[1])
                        if dominant_prob >= 0.5:
                            lda_theme = self.theme_mappings.get(dominant_topic, 'none')
                            if sentiment == 'positive' and lda_theme != 'Positive User Experience':
                                theme = 'Positive User Experience'  # Override for positive sentiment
                                prob = max(prob, dominant_prob)
                            elif sentiment == 'negative' and lda_theme in ['Technical Issues', 'Security Issues']:
                                theme = lda_theme
                                prob = max(prob, dominant_prob)
                    self.logger.debug(f"Assigned theme: {review[:50]}... -> {theme} (prob={prob}, sentiment={sentiment})")
                themes.append(theme)
                probabilities.append(prob)
            df['theme'] = themes
            df['topic_probability'] = probabilities
            self.logger.info(f"Completed theme assignment for {len(df)} reviews")
            return df
        except Exception as e:
            self.logger.error(f"Error in theme assignment: {e}")
            raise

    def summarize_themes(self, df: pd.DataFrame):
        try:
            theme_summary = df.groupby(['bank', 'theme']).size().unstack(fill_value=0)
            theme_summary['total'] = theme_summary.sum(axis=1)
            theme_summary_pct = theme_summary.div(theme_summary['total'], axis=0) * 100
            self.logger.info(f"Theme counts by bank:\n{theme_summary}")
            self.logger.info(f"Theme percentages by bank:\n{theme_summary_pct}")
            theme_summary.to_csv(os.path.join(self.output_dir, 'theme_summary.csv'))
            theme_summary_pct.to_csv(os.path.join(self.output_dir, 'theme_summary_percentage.csv'))
            self.logger.info("Saved theme summaries")
        except Exception as e:
            self.logger.error(f"Error summarizing themes: {e}")
            raise

    def visualize_topics(self, lda_model, corpus, dictionary):
        try:
            vis_data = gensimvis.prepare(lda_model, corpus, dictionary, n_jobs=1)
            output_path = os.path.join(self.output_dir, "lda_topics.html")
            pyLDAvis.save_html(vis_data, output_path)
            self.logger.info(f"Saved interactive topic visualization to {output_path}")
        except Exception as e:
            self.logger.error(f"Error generating LDA visualization: {e}")
            raise

    def save_model(self, lda_model, dictionary):
        try:
            lda_model.save(os.path.join(self.output_dir, "lda_model.gensim"))
            dictionary.save(os.path.join(self.output_dir, "lda_dictionary.pkl"))
            self.logger.info("Saved LDA model and dictionary")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def save_results(self, df: pd.DataFrame):
        try:
            output_path = os.path.join(self.output_dir, 'thematic_results.csv')
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved results to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise

    def run(self):
        df = self.load_data()
        texts = [self.preprocess_text(review) for review in df['review'].fillna("")]
        lda_model, dictionary, corpus = self.train_lda(texts)
        self.inspect_topics(lda_model)
        self.map_topics_to_themes(lda_model)
        df = self.assign_themes(df, lda_model, corpus)
        self.summarize_themes(df)
        self.visualize_topics(lda_model, corpus, dictionary)
        self.save_model(lda_model, dictionary)
        self.save_results(df)
        return df, lda_model, dictionary

if __name__ == "__main__":
    analyzer = ThematicAnalyzer(input_path="./data/processed/sentiment_results.csv")
    try:
        results, lda_model, dictionary = analyzer.run()
    except Exception as e:
        print(f"Error running thematic analysis: {e}")
