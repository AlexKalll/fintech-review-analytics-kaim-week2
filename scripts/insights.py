import pandas as pd
import oracledb
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

class ReviewAnalyzer:
    """
    Analyzes bank review data to generate insights and visualizations.

    Handles fetching data from an Oracle database, creating plots for sentiment,
    ratings, and themes, and saving the results.
    """

    def __init__(self, dsn, username, password, output_dir="plots/figures"):
        """
        Initializes the analyzer with database credentials and output path.

        Args:
            dsn (str): The Data Source Name for the Oracle connection.
            username (str): The database username.
            password (str): The database password.
            output_dir (str): The directory to save generated plots.
        """
        self.dsn = dsn
        self.username = username
        self.password = password
        self.output_dir = output_dir
        self.df = None  # To store the fetched DataFrame
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_data(self):
        """
        Fetches review data from the Oracle database and loads it into a DataFrame.

        Returns:
            bool: True if data was fetched successfully, False otherwise.
        """
        try:
            with oracledb.connect(user=self.username, password=self.password, dsn=self.dsn) as conn:
                query = """
                SELECT b.bank_name, r.review_id, r.review, r.rating, r.review_date,
                       r.source, r.sentiment, r.confidence, r.theme, r.topic_probability
                FROM Banks b
                JOIN Reviews r ON b.bank_id = r.bank_id
                """
                self.df = pd.read_sql(query, conn)
                print("Successfully fetched data from the database.")
                return True
        except oracledb.Error as e:
            print(f"Oracle Error: {e}")
            return False

    def run_analysis(self):
        """
        Runs all analysis and visualization tasks if data is loaded.
        """
        if self.df is None:
            print("Data not found. Please fetch data first using fetch_data().")
            return

        print("Generating visualizations...")
        self._plot_sentiment_trends()
        self._plot_rating_distributions()
        self._plot_theme_percentages()
        self._plot_keyword_clouds()

        # Save processed data for reporting
        processed_data_path = 'data/processed/task4_data.csv'
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        self.df.to_csv(processed_data_path, index=False)

        print(f"Plots saved to {self.output_dir}/")
        print(f"Data saved to {processed_data_path}")

    def _plot_sentiment_trends(self):
        """Generates and saves a plot of sentiment trends over time."""
        self.df['month'] = pd.to_datetime(self.df['REVIEW_DATE']).dt.to_period('M')
        sentiment_counts = self.df.groupby(['month', 'BANK_NAME', 'SENTIMENT']).size().unstack(fill_value=0)

        sentiment_counts.plot(kind='bar', stacked=True, figsize=(12, 7))
        plt.title('Monthly Sentiment Trends by Bank')
        plt.xlabel('Month')
        plt.ylabel('Number of Reviews')
        plt.legend(title='Sentiment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment_trends.png'))
        plt.close()

    def _plot_rating_distributions(self):
        """Generates and saves a boxplot of rating distributions."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='BANK_NAME', y='RATING', data=self.df)
        plt.title('Rating Distributions by Bank')
        plt.xlabel('Bank')
        plt.ylabel('Rating')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rating_distributions.png'))
        plt.close()

    def _plot_theme_percentages(self):
        """Generates and saves a plot of review theme percentages."""
        theme_counts = self.df.groupby(['BANK_NAME', 'THEME']).size().unstack(fill_value=0)
        theme_percent = theme_counts.div(theme_counts.sum(axis=1), axis=0) * 100

        theme_percent.plot(kind='bar', stacked=True, figsize=(12, 7))
        plt.title('Theme Percentages by Bank')
        plt.xlabel('Bank')
        plt.ylabel('Percentage of Reviews (%)')
        plt.legend(title='Theme', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'theme_percentages.png'))
        plt.close()

    def _plot_keyword_clouds(self):
        """Generates and saves a word cloud for each bank."""
        for bank in self.df['BANK_NAME'].unique():
            bank_reviews = self.df[self.df['BANK_NAME'] == bank]['REVIEW'].str.cat(sep=' ')
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis'
            ).generate(bank_reviews)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Keyword Cloud for {bank}')
            filename = f'keyword_cloud_{bank.lower().replace(" ", "_")}.png'
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()

def main():
    """
    Main function to execute the bank review analysis.
    """
    # Database connection details
    dsn = "localhost:1521/XEPDB1"
    username = "bank_reviews"
    password = "Welcome123"

    # Initialize and run the analyzer
    analyzer = ReviewAnalyzer(dsn, username, password)
    if analyzer.fetch_data():
        analyzer.run_analysis()

if __name__ == "__main__":
    main()