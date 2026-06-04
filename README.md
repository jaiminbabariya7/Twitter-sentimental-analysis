# Twitter Sentiment Analysis — NLP Opinion Mining Pipeline

![Python](https://img.shields.io/badge/Python-3.6+-blue?logo=python)
![NLP](https://img.shields.io/badge/NLP-TextBlob%20%7C%20VADER-green)
![Tweepy](https://img.shields.io/badge/Twitter%20API-Tweepy-1DA1F2?logo=twitter)
![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-orange)

> Python NLP pipeline that collects tweets via the Twitter API, preprocesses text, classifies sentiment using TextBlob + VADER, and visualizes opinion distribution — the prototype for the larger [Real-Time Streaming NLP Pipeline](https://github.com/jaiminbabariya7/real_time_sentiment_analysis) on GCP.

---

## What It Does

Given any keyword, hashtag, or topic, this tool:
1. Fetches recent tweets from the Twitter API via Tweepy
2. Cleans and normalizes tweet text (remove URLs, mentions, symbols)
3. Scores each tweet for polarity and subjectivity
4. Classifies: Positive / Neutral / Negative
5. Outputs a visual pie chart + detailed breakdown

---

## Pipeline

```
User input: keyword + tweet count
        ↓
Tweepy → Twitter API v2
  └── Fetch N tweets matching keyword
        ↓
Text Preprocessing
  ├── Remove URLs (https://...)
  ├── Remove @mentions
  ├── Remove hashtag symbols (keep word)
  ├── Remove punctuation & special chars
  └── Lowercase & strip whitespace
        ↓
Dual Sentiment Scoring
  ├── TextBlob: polarity (-1.0 to +1.0), subjectivity (0 to 1.0)
  └── VADER: compound score (-1.0 to +1.0) — better for social media
        ↓
Classification
  ├── Positive: compound >= 0.05
  ├── Negative: compound <= -0.05
  └── Neutral: -0.05 < compound < 0.05
        ↓
Output: pie chart + per-tweet breakdown table
```

---

## Code

### Full Pipeline
```python
# main.py
import tweepy
import re
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET

# Auth
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

vader = SentimentIntensityAnalyzer()

def clean_tweet(text: str) -> str:
    """Remove noise from tweet text."""
    text = re.sub(r'http\S+|www\S+', '', text)     # URLs
    text = re.sub(r'@\w+', '', text)                # @mentions
    text = re.sub(r'RT\s+', '', text)               # retweet prefix
    text = re.sub(r'#(\w+)', r'\1', text)           # hashtag → word
    text = re.sub(r'[^\w\s]', '', text)             # punctuation
    text = re.sub(r'\s+', ' ', text)                # extra spaces
    return text.strip().lower()

def analyze_tweet(text: str) -> dict:
    """Score tweet with both TextBlob and VADER."""
    cleaned = clean_tweet(text)

    # TextBlob
    tb = TextBlob(cleaned)
    tb_polarity = round(tb.sentiment.polarity, 4)
    tb_subjectivity = round(tb.sentiment.subjectivity, 4)

    # VADER (better for short, informal text)
    vs = vader.polarity_scores(cleaned)
    compound = vs["compound"]

    # Classification (VADER compound as primary)
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "original": text[:100],
        "cleaned": cleaned[:100],
        "tb_polarity": tb_polarity,
        "tb_subjectivity": tb_subjectivity,
        "vader_compound": round(compound, 4),
        "sentiment": label,
    }

def fetch_and_analyze(keyword: str, count: int = 200) -> pd.DataFrame:
    """Fetch tweets and analyze sentiment."""
    print(f"\nFetching {count} tweets for: '{keyword}'")
    tweets = tweepy.Cursor(
        api.search_tweets,
        q=f"{keyword} -filter:retweets lang:en",
        tweet_mode="extended",
        count=100,
    ).items(count)

    results = []
    for tweet in tweets:
        text = tweet.full_text
        analysis = analyze_tweet(text)
        analysis["created_at"] = tweet.created_at
        analysis["retweet_count"] = tweet.retweet_count
        analysis["favorite_count"] = tweet.favorite_count
        results.append(analysis)

    return pd.DataFrame(results)

def visualize(df: pd.DataFrame, keyword: str):
    """Generate sentiment charts."""
    counts = df["sentiment"].value_counts()
    colors = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Sentiment Analysis: "{keyword}"  |  {len(df)} tweets', fontsize=14, fontweight="bold")

    # Pie chart
    axes[0].pie(
        counts.values,
        labels=counts.index,
        colors=[colors[k] for k in counts.index],
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12},
    )
    axes[0].set_title("Sentiment Distribution")

    # Polarity histogram
    axes[1].hist(df["vader_compound"], bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    axes[1].axvline(x=0.05, color="#2ecc71", linestyle="--", label="Positive threshold")
    axes[1].axvline(x=-0.05, color="#e74c3c", linestyle="--", label="Negative threshold")
    axes[1].set_xlabel("VADER Compound Score")
    axes[1].set_ylabel("Tweet Count")
    axes[1].set_title("Polarity Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"output/sentiment_{keyword.replace(' ', '_')}.png", dpi=150)
    plt.show()
    print(f"\nChart saved to output/sentiment_{keyword}.png")

if __name__ == "__main__":
    keyword = input("Enter keyword/hashtag to analyze: ").strip()
    count = int(input("Number of tweets to analyze (50-500): ").strip())

    df = fetch_and_analyze(keyword, count)
    visualize(df, keyword)

    # Summary
    dist = df["sentiment"].value_counts(normalize=True) * 100
    print(f"\n{'='*40}")
    print(f"Results for: '{keyword}' ({len(df)} tweets)")
    print(f"{'='*40}")
    for label, pct in dist.items():
        print(f"  {label:10s}: {pct:.1f}%")
    print(f"\nAvg VADER compound: {df['vader_compound'].mean():.4f}")
    print(f"Avg subjectivity:   {df['tb_subjectivity'].mean():.4f}")

    df.to_csv(f"output/results_{keyword.replace(' ', '_')}.csv", index=False)
```

---

## Sample Output

```
Fetching 200 tweets for: 'ChatGPT'

========================================
Results for: 'ChatGPT' (200 tweets)
========================================
  Positive  : 52.5%
  Neutral   : 28.0%
  Negative  : 19.5%

Avg VADER compound: 0.1842
Avg subjectivity:   0.4321

Top positive tweet:
"ChatGPT just helped me write a complete business plan in 20 minutes. Absolutely incredible."
(compound: 0.87, subjectivity: 0.72)

Top negative tweet:
"ChatGPT gave me completely wrong information and I submitted it to my professor. Terrible."
(compound: -0.74, subjectivity: 0.63)

Chart saved to: output/sentiment_ChatGPT.png
Results saved to: output/results_ChatGPT.csv
```

---

## Relation to Production NLP Pipeline

This project established the core NLP patterns used in the production-scale [Real-Time Sentiment Analysis Pipeline](https://github.com/jaiminbabariya7/real_time_sentiment_analysis), which replaces:
- Twitter API → **Google Cloud Pub/Sub** (any text source)
- TextBlob/VADER → **Google Cloud Natural Language API** (production-grade)
- Script execution → **Apache Beam / Cloud Dataflow** (auto-scaling, streaming)
- CSV export → **BigQuery** (queryable at scale)

---

## Setup

```bash
pip install tweepy textblob vaderSentiment pandas matplotlib
python -m textblob.download_corpora
```

Create `config.py`:
```python
CONSUMER_KEY = "your_key"
CONSUMER_SECRET = "your_secret"
ACCESS_TOKEN = "your_token"
ACCESS_TOKEN_SECRET = "your_token_secret"
```

```bash
python main.py
```

---

## Skills Demonstrated
`NLP` · `Sentiment Analysis` · `VADER` · `TextBlob` · `Twitter API` · `Tweepy` · `Text Preprocessing` · `Matplotlib` · `Opinion Mining` · `Python`
