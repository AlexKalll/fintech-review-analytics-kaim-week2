from google_play_scraper import Sort, reviews
import pandas as pd
from datetime import datetime
import os

def fetch_reviews(app_id, app_name, max_reviews=500):
    print(f"Scraping reviews for {app_name}...")
    all_reviews = []
    count = 0

    while len(all_reviews) < max_reviews:
        result, _ = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=200,
            filter_score_with=None
        )
        all_reviews.extend(result)
        count += 1
        if len(result) < 200 or count > 10:
            break

    df = pd.DataFrame(all_reviews)
    df = df[['reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 'reviewCreatedVersion', 'at']]
    df['bank'] = app_name
    df.rename(columns={
        'content': 'review',
        'score': 'rating',
        'at': 'date',
        'reviewCreatedVersion': 'app_version'
    }, inplace=True)
    return df

def save_reviews(df, bank_name):
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/{bank_name}_reviews_{today}.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} reviews to {file_path}")

def scrape_and_save():
    apps = {
        "com.combanketh.mobilebanking": "CBE",
        "com.boa.boaMobileBanking": "BOA",
        "com.dashen.dashensuperapp": "Dashen"
    }

    for app_id, name in apps.items():
        df = fetch_reviews(app_id, name, max_reviews=500)
        save_reviews(df, name)

if __name__ == "__main__":
    scrape_and_save()
    print("Done")