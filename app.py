import os
import googleapiclient.discovery
import pandas as pd
import numpy as np
import streamlit as st
import re
import emoji
import matplotlib.pyplot as plt
# NEW imports (use TensorFlow's Keras)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Function to retrieve YouTube comments
def get_youtube_comments(video_id, max_results=100):
    api_key = "AIzaSyCZir4PlFLondxiJ_K5_FILe7M_K6X4g54"
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None
    total_comments_retrieved = 0

    while total_comments_retrieved < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - total_comments_retrieved),
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append([
                comment["authorDisplayName"],
                comment["publishedAt"],
                comment["likeCount"],
                comment["textDisplay"]
            ])
            total_comments_retrieved += 1
            if total_comments_retrieved >= max_results:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(comments, columns=['Author', 'Published At', 'Likes', 'Text'])

# Function to clean text
def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = text.lower().strip()
    return text

# Function to extract emojis
def extract_emojis(text):
    return "".join(c for c in text if c in emoji.UNICODE_EMOJI["en"])

# Function to analyze sentiment
def analyze_sentiment(df, model, tokenizer):
    df["Cleaned_Text"] = df["Text"].apply(clean_text)
    df["Emojis"] = df["Text"].apply(extract_emojis)

    X = tokenizer.texts_to_sequences(df["Cleaned_Text"])
    X = pad_sequences(X, maxlen=100)

    predicted_probabilities = model.predict(X)
    df["Text_Pos"] = predicted_probabilities[:, 2]
    df["Text_Neg"] = predicted_probabilities[:, 0]
    df["Text_Neut"] = predicted_probabilities[:, 1]

    df["Predicted_Sentiment"] = df[["Text_Pos", "Text_Neg", "Text_Neut"]].idxmax(axis=1)
    return df

# Streamlit App
st.title("YouTube Sentiment Analysis")

option = st.selectbox("Choose Option", ["Download YouTube Comments", "Analyze Sentiment"])

if option == "Download YouTube Comments":
    video_id = st.text_input("Enter YouTube Video ID:")
    max_results = st.slider("Max Comments:", min_value=1, max_value=1000, value=100)

    if st.button("Download Comments"):
        df = get_youtube_comments(video_id, max_results)
        df.to_csv("youtube_comments.csv", index=False)
        st.dataframe(df)
        st.success("Comments saved as `youtube_comments.csv`")

elif option == "Analyze Sentiment":
    st.subheader("Upload YouTube Comments File")
    uploaded_file = st.file_uploader("Upload `youtube_comments.csv`", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df["Text"])
        model = load_model("Model1.h5")  # Upload this file in Streamlit Cloud

        df = analyze_sentiment(df, model, tokenizer)
        st.dataframe(df)

        # Sentiment Distribution Pie Chart
        sentiment_counts = df["Predicted_Sentiment"].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["green", "red", "gray"])
        plt.title("Sentiment Distribution")
        st.pyplot(plt)
