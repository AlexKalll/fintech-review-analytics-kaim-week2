import pandas as pd
import os
import logging

class ReviewCombiner:
    def __init__(self, data_dir: str, bank_files: list):
        self.data_dir = data_dir
        self.bank_files = bank_files
        self.expected_columns = ['review_id', 'review', 'rating', 'date', 'bank', 'source']
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def combine(self) -> pd.DataFrame:
        dfs = []
        required_cols = ['review', 'rating', 'date', 'bank', 'source']
        for bank_file in self.bank_files:
            file_path = os.path.join(self.data_dir, bank_file)
            if not os.path.exists(file_path):
                self.logger.warning(f"File not found: {file_path}")
                continue
            try:
                df = pd.read_csv(file_path)
                # Validate required columns
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    self.logger.error(f"Missing columns in {file_path}: {missing_cols}")
                    continue
                dfs.append(df)
                self.logger.info(f"Loaded {len(df)} reviews from {file_path}")
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                continue

        if not dfs:
            self.logger.error("No valid CSV files found")
            raise ValueError("No valid CSV files")

        # Combine DataFrames
        all_reviews = pd.concat(dfs, ignore_index=True)
        # Add review_id
        all_reviews['review_id'] = range(1, len(all_reviews) + 1)

        # Select expected columns for output
        output_cols = [col for col in self.expected_columns if col in all_reviews.columns or col == 'review_id']
        all_reviews = all_reviews[output_cols]

        # Validate data quality
        total_reviews = len(all_reviews)
        missing_counts = all_reviews[required_cols].isna().sum()
        missing_percentages = missing_counts / total_reviews * 100
        self.logger.info(f"Total reviews: {total_reviews}")
        self.logger.info(f"Missing data counts:\n{missing_counts}")
        self.logger.info(f"Missing data percentages:\n{missing_percentages}")

        # Save combined CSV
        output_path = os.path.join(self.data_dir, 'all_reviews.csv')
        all_reviews.to_csv(output_path, index=False)
        self.logger.info(f"Combined {total_reviews} reviews into {output_path}")
        return all_reviews

if __name__ == "__main__":
    combiner = ReviewCombiner(
        data_dir="../data/processed/",
        bank_files=['cbe_raw_reviews.csv', 'boa_raw_reviews.csv', 'dashen_raw_reviews.csv']
    )
    try:
        all_reviews = combiner.combine()
    except Exception as e:
        print(f"Error combining reviews: {e}")