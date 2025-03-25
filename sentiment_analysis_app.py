from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API keys from .env
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
twitter_api_key = os.getenv("TWITTER_API_KEY")
twitter_api_secret = os.getenv("TWITTER_API_SECRET")
twitter_access_token = os.getenv("ACCESS_TOKEN")
twitter_access_secret = os.getenv("ACCESS_TOKEN_SECRET")

import streamlit as st
from googleapiclient.discovery import build
import tweepy
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import emoji

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def get_youtube_comments(video_id, api_key, max_results=50):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )
    response = request.execute()
    
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    
    return comments

def get_twitter_tweets(query, api_key, api_secret, access_token, access_token_secret, max_tweets=50):
    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en").items(max_tweets):
        tweets.append(tweet.text)
    
    return tweets

def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    
    if sentiment['compound'] >= 0.05:
        return "Positive 😀"
    elif sentiment['compound'] <= -0.05:
        return "Negative 😡"
    else:
        return "Neutral 😐"

emoji_sentiment = {
    "😀": "Positive", "😂": "Positive", "😍": "Positive", 
    "😢": "Negative", "😡": "Negative", "😐": "Neutral"
}

def extract_emoji_sentiment(text):
    emojis = [char for char in text if char in emoji.EMOJI_DATA]
    sentiments = [emoji_sentiment.get(e, "Unknown") for e in emojis]
    return emojis, sentiments

st.title("📊 Social Media Sentiment Analysis")

# YouTube Section
st.header("🎥 Analyze YouTube Comments")
video_id = st.text_input("Enter YouTube Video ID")

if st.button("Get YouTube Comments"):
    if not video_id:
        st.error("Please enter a YouTube video ID.")
    else:
        comments = get_youtube_comments(video_id, youtube_api_key)
        for comment in comments:
            sentiment = analyze_sentiment(comment)
            emojis, emoji_sentiments = extract_emoji_sentiment(comment)
            st.write(f"**Comment:** {comment}")
            st.write(f"**Sentiment:** {sentiment}")
            if emojis:
                st.write(f"**Emojis:** {''.join(emojis)} - Sentiment: {emoji_sentiments}")

# Twitter Section
st.header("🐦 Analyze Twitter Tweets")
tweet_query = st.text_input("Enter a hashtag or keyword")

if st.button("Get Twitter Tweets"):
    if not tweet_query:
        st.error("Please enter a hashtag or keyword.")
    else:
        tweets = get_twitter_tweets(tweet_query, twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_secret)
        for tweet in tweets:
            sentiment = analyze_sentiment(tweet)
            emojis, emoji_sentiments = extract_emoji_sentiment(tweet)
            st.write(f"**Tweet:** {tweet}")
            st.write(f"**Sentiment:** {sentiment}")
            if emojis:
                st.write(f"**Emojis:** {''.join(emojis)} - Sentiment: {emoji_sentiments}")
import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import emoji

# Load dataset
st.title("📊 Kaggle Twitter Dataset Analysis")

# File path (update this with your CSV name)
dataset_path = "Twitter_Data.csv"  # Replace with the correct name
df = pd.read_csv(dataset_path)

# Display dataset preview
st.write("📄 **Dataset Preview:**")
st.write(df.head())

# Sentiment Analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment = sia.polarity_scores(str(text))
    if sentiment['compound'] >= 0.05:
        return "Positive 😀"
    elif sentiment['compound'] <= -0.05:
        return "Negative 😡"
    else:
        return "Neutral 😐"

# Apply sentiment analysis
st.write("🔥 **Sentiment Analysis on Tweets:**")

# Check for the correct column name in your CSV (e.g., 'text' or 'content')
if 'clean_text' in df.columns:
    df['Sentiment'] = df['text'].apply(analyze_sentiment)
elif 'category' in df.columns:
    df['Sentiment'] = df['content'].apply(analyze_sentiment)
else:
    st.error("No valid text column found!")

# Display results
st.write(df[['text', 'Sentiment']].head(10))

