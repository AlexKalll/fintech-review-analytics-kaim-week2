import pandas as pd
import oracledb


class ReviewUploader:
    def __init__(self, csv_path, dsn, username, password):
        self.csv_path = csv_path
        self.dsn = dsn
        self.username = username
        self.password = password
        self.connection = None
        self.cursor = None

        # Hardcoded mapping based on pre-inserted banks
        self.bank_mapping = {
            "Commercial Bank of Ethiopia": 1,
            "Bank of Abyssinia": 2,
            "Dashen Bank": 3
        }

    def connect(self):
        self.connection = oracledb.connect(
            user=self.username,
            password=self.password,
            dsn=self.dsn
        )
        self.cursor = self.connection.cursor()

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def load_csv(self):
        return pd.read_csv(self.csv_path)

    def insert_reviews(self, df):
        insert_sql = """
        INSERT INTO Reviews (
            review_id, bank_id, review, rating, review_date, source,
            sentiment, confidence, theme, topic_probability
        ) VALUES (
            review_id_seq.NEXTVAL, :1, :2, :3, TO_DATE(:4, 'YYYY-MM-DD'), :5, :6, :7, :8, :9
        )
        """
        inserted = 0

        for _, row in df.iterrows():
            bank_id = self.bank_mapping.get(row["bank"])
            if not bank_id:
                continue

            rating = float(row["rating"]) if pd.notnull(row["rating"]) and 1 <= float(row["rating"]) <= 5 else None
            confidence = min(float(row["confidence"]), 0.99) if pd.notnull(row["confidence"]) else None
            topic_prob = min(float(row["topic_probability"]), 0.99) if pd.notnull(row["topic_probability"]) else None

            data = (
                bank_id,
                str(row["review"])[:4000] if pd.notnull(row["review"]) else None,
                rating,
                str(row["date"]),
                str(row["source"])[:100] if pd.notnull(row["source"]) else None,
                str(row["sentiment"])[:50] if pd.notnull(row["sentiment"]) else None,
                confidence,
                str(row["theme"])[:100] if pd.notnull(row["theme"]) else None,
                topic_prob
            )

            self.cursor.execute(insert_sql, data)
            inserted += 1

        self.connection.commit()
        print(f"Successfully inserted {inserted} reviews.")

    def run(self):
        try:
            self.connect()
            df = self.load_csv()
            self.insert_reviews(df)
        except oracledb.Error as e:
            print(f"Oracle Error: {e}")
        except Exception as e:
            print(f"General Error: {e}")
        finally:
            self.close()


if __name__ == "__main__":
    uploader = ReviewUploader(
        csv_path="data/processed/thematic_results.csv",
        dsn="localhost:1521/XEPDB1",
        username="bank_reviews",
        password="Welcome123"
    )
    uploader.run()