import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from textblob import TextBlob
import googleapiclient.discovery
from urllib.parse import urlparse, parse_qs
import re
import requests
# Set the path to your custom favicon
favicon_path = "favicon.png"  # Ensure this matches your favicon file name

# Change title and favicon
st.set_page_config(page_title="My Sentiment Analysis App", page_icon=favicon_path)
# Load environment variables from .env file
load_dotenv()
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)
# Function to perform enhanced text cleaning
def clean_text_enhanced(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stopwords_list = set(['the', 'and', 'is', 'are', 'it', 'in', 'on', 'at', 'for', 'this', 'that', 'with', 'to', 'from', 'we', 'you', 'me', 'he', 'she', 'they', 'them'])
    text = ' '.join(word for word in text.split() if word not in stopwords_list)
    text = ' '.join(text.split())
    return text

# Function to classify sentiment based on score
def classify_sentiment(score):
    if score >= 0.3:
        return 'Positive'
    elif score <= -0.3:
        return 'Negative'
    else:
        return 'Neutral'

# Function to score sentiment
def score_sentiment(text):
    return round(TextBlob(text).sentiment.polarity, 2)

# Updated function to analyze YouTube video using YouTube Data API
def analyze_youtube_video(url):
    try:
        parsed_url = urlparse(url)
        video_id = parse_qs(parsed_url.query).get('v')
        if video_id:
            video_id = video_id[0]
            youtube_api_key = os.getenv('YOUTUBE_API_KEY')
            if not youtube_api_key:
                st.error("YouTube API key not found. Please make sure to set it in the .env file.")
                return

            comments = fetch_youtube_comments(video_id, youtube_api_key)

            if comments:
                cleaned_comments = [clean_text_enhanced(comment) for comment in comments]
                comment_sentiments = [score_sentiment(comment) for comment in cleaned_comments]
                sentiment_labels = [classify_sentiment(score) for score in comment_sentiments]

                df_comments = pd.DataFrame({
                    'Comment': comments,
                    'Cleaned Comment': cleaned_comments,
                    'Sentiment Score': comment_sentiments,
                    'Label': sentiment_labels
                })

                st.header('Comments Analysis')
                st.write(df_comments)

                sentiment_counts = df_comments['Label'].value_counts(normalize=True) * 100
                total_comments = len(df_comments)
                positive_percentage = sentiment_counts.get('Positive', 0)
                negative_percentage = sentiment_counts.get('Negative', 0)
                neutral_percentage = sentiment_counts.get('Neutral', 0)

                st.subheader('Sentiment Analysis Summary')
                st.write('Positive:', round(positive_percentage, 2), '%')
                st.progress(positive_percentage / 100)

                st.write('Negative:', round(negative_percentage, 2), '%')
                st.progress(negative_percentage / 100)

                st.write('Neutral:', round(neutral_percentage, 2), '%')
                st.progress(neutral_percentage / 100)

                csv_comments = df_comments.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="Download analyzed comments as CSV",
                    data=csv_comments,
                    file_name='analyzed_comments.csv',
                    mime='text/csv',
                )
            else:
                st.warning("No comments found for the given YouTube video.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Function to fetch comments using YouTube Data API
def fetch_youtube_comments(video_id, youtube_api_key):
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=youtube_api_key)
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        request = youtube.commentThreads().list_next(request, response)

    return comments

# Streamlit app
st.title('Sentiment Analysis')
nav_option = st.sidebar.selectbox(
    'Navigation',
    ['Analyze Text', 'Clean Text', 'Analyze CSV', 'Analyze YouTube Video']
)

if nav_option == 'Analyze Text':
    st.header('Analyze Text')
    text = st.text_input('Enter text:')
    if text:
        cleaned_text = clean_text_enhanced(text)
        blob = TextBlob(cleaned_text)
        st.write('Cleaned Text:', cleaned_text)
        st.write('Polarity:', round(blob.sentiment.polarity, 2))
        st.write('Subjectivity:', round(blob.sentiment.subjectivity, 2))

elif nav_option == 'Clean Text':
    st.header('Clean Text')
    pre = st.text_input('Enter text to clean:')
    if pre:
        cleaned_text = clean_text_enhanced(pre)
        st.write('Cleaned Text:', cleaned_text)

elif nav_option == 'Analyze CSV':
    st.header('Analyze CSV')
    upl = st.file_uploader('Upload CSV file')
    if upl:
        df = pd.read_csv(upl)

        # Ensure the selected column is not empty
        if not df.empty:
            # Apply enhanced text cleaning to the selected column
            text_column = st.selectbox('Select the column containing text to clean and analyze:', df.columns)
            df['cleaned_text'] = df[text_column].apply(clean_text_enhanced)

            if st.button('Analyze Sentiment'):
                # Apply sentiment analysis to the cleaned text
                df['score'] = df['cleaned_text'].apply(lambda x: score_sentiment(str(x)))
                df['analysis'] = df['score'].apply(lambda x: classify_sentiment(x))

                # Save analysis results to session state
                st.session_state.data = df[['cleaned_text', 'analysis', 'score']]

                # Display the sentiment analysis results using a bar chart
                sentiment_counts = df['analysis'].value_counts()
                st.bar_chart(sentiment_counts)

                # Display ANALYSIS RESULTS
                st.header('ANALYSIS RESULTS')
                if st.session_state.data is not None:
                    st.write(st.session_state.data)
                else:
                    st.warning("No data to display. Please upload a valid CSV file or analyze text.")

                # Download the resulting DataFrame as a CSV file
                csv = df[['cleaned_text', 'analysis', 'score']].to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="Download analyzed data as CSV",
                    data=csv,
                    file_name='analyzed_data.csv',
                    mime='text/csv',
                )
        else:
            st.warning("The uploaded CSV file is empty. Please upload a valid CSV file.")

elif nav_option == 'Analyze YouTube Video':
    st.header('Analyze YouTube Video')
    youtube_url = st.text_input('Enter YouTube video URL:')
    if youtube_url:
        analyze_youtube_video(youtube_url)
