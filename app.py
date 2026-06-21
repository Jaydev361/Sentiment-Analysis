import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import googleapiclient.discovery
import pandas as pd
import streamlit as st
import torch
from dotenv import load_dotenv
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_TITLE = "Sentiment Analysis App"
DEFAULT_MODEL_KEY = "twitter_xlm_roberta"
FALLBACK_MODEL_KEY = "nlptown_bert"
MAX_YOUTUBE_COMMENTS = 500


@dataclass(frozen=True)
class ModelConfig:
    display_name: str
    model_name: str
    cache_dir: str
    labels: Dict[int, str]
    description: str
    score_map: Dict[str, float]
    use_fast_tokenizer: bool = True


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "twitter_xlm_roberta": ModelConfig(
        display_name="XLM-RoBERTa Twitter Sentiment (recommended)",
        model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        cache_dir="models/twitter-xlm-roberta",
        labels={0: "Negative", 1: "Neutral", 2: "Positive"},
        description=(
            "A multilingual sentiment model trained for short, social-media-like text. "
            "It is a strong fit for YouTube comments, Romanized text, and mixed-language inputs."
        ),
        score_map={"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0},
        use_fast_tokenizer=False,
    ),
    "nlptown_bert": ModelConfig(
        display_name="Multilingual BERT 1-5 Stars (current fallback)",
        model_name="nlptown/bert-base-multilingual-uncased-sentiment",
        cache_dir="models/nlptown",
        labels={0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"},
        description=(
            "Your current multilingual model. It predicts a 1-5 star rating and is preserved "
            "as a stable fallback for the existing app behavior."
        ),
        score_map={
            "1 star": -1.0,
            "2 stars": -1.0,
            "3 stars": 0.0,
            "4 stars": 1.0,
            "5 stars": 1.0,
        },
    ),
}


st.set_page_config(page_title=APP_TITLE, page_icon="favicon.png")
load_dotenv()


def hide_streamlit_chrome() -> None:
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_text(text: str, aggressive_english_cleanup: bool = False) -> str:
    """Clean text without destroying multilingual content.

    The previous cleaner removed everything outside A-Z, which made Hindi, Bengali,
    emoji, and most Romanized punctuation-heavy comments vanish before inference.
    This version removes noisy links/mentions while preserving Unicode letters.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if aggressive_english_cleanup:
        stopwords = {
            "the",
            "and",
            "is",
            "are",
            "it",
            "in",
            "on",
            "at",
            "for",
            "this",
            "that",
            "with",
            "to",
            "from",
            "we",
            "you",
            "me",
            "he",
            "she",
            "they",
            "them",
        }
        text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        text = " ".join(word for word in text.split() if word not in stopwords)

    return " ".join(text.split())


def clean_text_enhanced(text: str) -> str:
    """Compatibility wrapper for the original app function name."""
    return normalize_text(text)


def classify_sentiment(score: float) -> str:
    if score >= 0.3:
        return "Positive"
    if score <= -0.3:
        return "Negative"
    return "Neutral"


def rating_to_score(rating: int) -> float:
    if rating in (1, 2):
        return -1.0
    if rating == 3:
        return 0.0
    if rating in (4, 5):
        return 1.0
    return 0.0


@st.cache_resource(show_spinner=False)
def load_sentiment_model(model_key: str):
    config = MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        use_fast=config.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    model.eval()
    return tokenizer, model


def get_model_or_fallback(model_key: str):
    try:
        return model_key, *load_sentiment_model(model_key)
    except Exception as primary_error:
        if model_key == FALLBACK_MODEL_KEY:
            raise primary_error

        st.warning(
            "The recommended model could not be loaded, so the app is using your existing "
            "multilingual BERT model instead. If this is the first run, check your internet "
            "connection so Transformers can download the recommended model once."
        )
        return FALLBACK_MODEL_KEY, *load_sentiment_model(FALLBACK_MODEL_KEY)


def analyze_sentiment(text: str, model_key: str) -> Dict[str, object]:
    active_model_key, tokenizer, model = get_model_or_fallback(model_key)
    config = MODEL_REGISTRY[active_model_key]

    cleaned_text = normalize_text(text)
    if not cleaned_text:
        return {
            "cleaned_text": "",
            "model": config.display_name,
            "raw_label": "Empty",
            "score": 0.0,
            "label": "Neutral",
            "confidence": 0.0,
        }

    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = softmax(outputs.logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    raw_label = config.labels.get(predicted_index, str(predicted_index))
    score = config.score_map.get(raw_label, 0.0)

    return {
        "cleaned_text": cleaned_text,
        "model": config.display_name,
        "raw_label": raw_label,
        "score": score,
        "label": classify_sentiment(score),
        "confidence": round(float(probabilities[predicted_index].item()), 4),
    }


def multilingual_sentiment_rating(text: str) -> int:
    """Compatibility helper that preserves the previous 1-5 rating behavior."""
    _, tokenizer, model = get_model_or_fallback(FALLBACK_MODEL_KEY)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = softmax(outputs.logits, dim=1)

    return int(torch.argmax(probabilities).item()) + 1


def extract_video_id(url: str) -> Optional[str]:
    parsed_url = urlparse(url.strip())
    hostname = (parsed_url.hostname or "").lower()

    if hostname in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
        if parsed_url.path.startswith("/shorts/") or parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/")[2]

    if hostname == "youtu.be":
        return parsed_url.path.lstrip("/") or None

    return None


def fetch_youtube_comments(video_id: str, youtube_api_key: str, max_comments: int = MAX_YOUTUBE_COMMENTS) -> List[str]:
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=youtube_api_key)
    comments: List[str] = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(100, max_comments),
        textFormat="plainText",
        order="relevance",
    )

    while request and len(comments) < max_comments:
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break
        request = youtube.commentThreads().list_next(request, response)

    return comments


def analyze_texts_to_dataframe(texts: List[str], model_key: str) -> pd.DataFrame:
    rows = []
    progress = st.progress(0, text="Analyzing sentiment...")

    for index, text in enumerate(texts, start=1):
        result = analyze_sentiment(str(text), model_key)
        rows.append(
            {
                "Original Text": text,
                "Cleaned Text": result["cleaned_text"],
                "Model": result["model"],
                "Raw Model Label": result["raw_label"],
                "Sentiment Score": result["score"],
                "Label": result["label"],
                "Confidence": result["confidence"],
            }
        )
        progress.progress(index / len(texts), text=f"Analyzing sentiment... {index}/{len(texts)}")

    progress.empty()
    return pd.DataFrame(rows)


def render_summary(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No rows were available for analysis.")
        return

    sentiment_counts = df["Label"].value_counts()
    avg_score = df["Sentiment Score"].mean()
    avg_confidence = df["Confidence"].mean()

    metric_cols = st.columns(3)
    metric_cols[0].metric("Rows analyzed", len(df))
    metric_cols[1].metric("Average score", f"{avg_score:.2f}")
    metric_cols[2].metric("Average confidence", f"{avg_confidence:.2%}")

    st.bar_chart(sentiment_counts)


def render_download(df: pd.DataFrame, file_name: str) -> None:
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download analysis as CSV",
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
    )


def analyze_youtube_video(url: str, model_key: str, max_comments: int) -> None:
    try:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL.")
            return

        youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if not youtube_api_key:
            st.error("YouTube API key not found. Add YOUTUBE_API_KEY to your .env file or Streamlit secrets.")
            return

        comments = fetch_youtube_comments(video_id, youtube_api_key, max_comments=max_comments)
        if not comments:
            st.warning("No comments found for the given YouTube video.")
            return

        df_comments = analyze_texts_to_dataframe(comments, model_key)
        st.header("Comments Analysis")
        render_summary(df_comments)
        st.dataframe(df_comments, use_container_width=True)
        render_download(df_comments, "analyzed_comments.csv")

    except Exception as error:
        st.error(f"Error while analyzing YouTube comments: {error}")


def render_sidebar() -> str:
    st.sidebar.title("Controls")
    model_display_names = {config.display_name: key for key, config in MODEL_REGISTRY.items()}
    selected_display_name = st.sidebar.selectbox(
        "Sentiment model",
        options=list(model_display_names.keys()),
        index=list(model_display_names.values()).index(DEFAULT_MODEL_KEY),
    )
    selected_model_key = model_display_names[selected_display_name]
    st.sidebar.caption(MODEL_REGISTRY[selected_model_key].description)
    return selected_model_key


hide_streamlit_chrome()

st.title("Sentiment Analysis")
selected_model_key = render_sidebar()

nav_option = st.sidebar.selectbox(
    "Navigation",
    ["Analyze Text", "Clean Text", "Analyze CSV", "Analyze YouTube Video"],
)

if nav_option == "Analyze Text":
    st.header("Analyze Text")
    text = st.text_area("Enter text (supports Hindi, Bengali, Romanized text, mixed-language comments, etc.):")

    if text:
        result = analyze_sentiment(text, selected_model_key)
        st.write("Cleaned Text:", result["cleaned_text"])
        st.write("Model:", result["model"])
        st.write("Raw Model Label:", result["raw_label"])
        st.write("Sentiment Score:", result["score"])
        st.write("Label:", result["label"])
        st.write("Confidence:", f"{result['confidence']:.2%}")

elif nav_option == "Clean Text":
    st.header("Clean Text")
    text_to_clean = st.text_area("Enter text to clean:")
    aggressive_cleanup = st.checkbox("Use old English-only cleanup mode")

    if text_to_clean:
        cleaned_text = normalize_text(text_to_clean, aggressive_english_cleanup=aggressive_cleanup)
        st.write("Cleaned Text:", cleaned_text)

elif nav_option == "Analyze CSV":
    st.header("Analyze CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.dropna(how="all")

        if df.empty:
            st.warning("The uploaded CSV file is empty. Please upload a valid CSV file.")
        else:
            text_column = st.selectbox("Select the column containing text to clean and analyze:", df.columns)
            selected_texts = df[text_column].dropna().astype(str).tolist()

            if st.button("Analyze Sentiment"):
                analyzed_df = analyze_texts_to_dataframe(selected_texts, selected_model_key)
                st.session_state.data = analyzed_df

                st.header("Analysis Results")
                render_summary(analyzed_df)
                st.dataframe(analyzed_df, use_container_width=True)
                render_download(analyzed_df, "analyzed_data.csv")

elif nav_option == "Analyze YouTube Video":
    st.header("Analyze YouTube Video")
    youtube_url = st.text_input("Enter YouTube video URL:")
    max_comments = st.slider("Maximum comments", min_value=25, max_value=500, value=100, step=25)

    if youtube_url:
        analyze_youtube_video(youtube_url, selected_model_key, max_comments)

