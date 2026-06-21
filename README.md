# Usage Notes

## Setup

Use Python 3.10 or newer.

```powershell
cd PATH_TO_YOUR_PROJECT_FOLDER
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Environment Variables

Create a `.env` file in the project root for private settings.

```env
YOUTUBE_API_KEY=your_youtube_data_api_key_here
```

The YouTube API key is only needed for the YouTube comment analysis screen. Text and CSV analysis work without it.

Do not commit `.env` to git.

## Run Command

```powershell
streamlit run app.py
```

Default local URL:

```text
http://localhost:8501
```

To run on a specific port:

```powershell
streamlit run app.py --server.port 8501
```

## Model Settings

Choose the model from the sidebar.

### XLM-RoBERTa Twitter Sentiment (recommended)

Model ID:

```text
cardiffnlp/twitter-xlm-roberta-base-sentiment
```

Cache folder:

```text
models/twitter-xlm-roberta
```

Labels:

```text
Negative, Neutral, Positive
```

This model is better for short multilingual comments, mixed-language text, and social-media-style writing. First use can take time because Hugging Face downloads the model weights.

Required dependencies for this model:

```text
sentencepiece
hf_xet
```

The app loads this tokenizer with `use_fast=False` to avoid the XLM-R tokenizer conversion issue on this environment.

### Multilingual BERT 1-5 Stars (fallback)

Model ID:

```text
nlptown/bert-base-multilingual-uncased-sentiment
```

Cache folder:

```text
models/nlptown
```

Raw labels:

```text
1 star, 2 stars, 3 stars, 4 stars, 5 stars
```

Score mapping:

```text
1 star, 2 stars -> Negative
3 stars -> Neutral
4 stars, 5 stars -> Positive
```

If the recommended model cannot load, the app falls back to this model automatically.

## Text Analysis

1. Open the sidebar.
2. Select `Analyze Text`.
3. Select a sentiment model.
4. Enter text in the text box.
5. Read the cleaned text, model label, sentiment score, final label, and confidence.

The default cleaning keeps Unicode text, so Hindi, Bengali, Romanized text, emoji, and mixed-language comments are not stripped away before model inference.

## Clean Text Mode

Use `Clean Text` to preview cleaning output.

The checkbox `Use old English-only cleanup mode` applies the older aggressive cleanup style. Use it only when you intentionally want English-only text normalization.

## CSV Analysis

1. Open `Analyze CSV`.
2. Upload a `.csv` file.
3. Select the text column.
4. Click `Analyze Sentiment`.
5. Review the summary, table, and chart.
6. Download the analyzed CSV from the download button.

Output columns include:

```text
Original Text
Cleaned Text
Model
Raw Model Label
Sentiment Score
Label
Confidence
```

## YouTube Comment Analysis

1. Add `YOUTUBE_API_KEY` to `.env`.
2. Open `Analyze YouTube Video`.
3. Paste a YouTube video URL.
4. Choose the maximum number of comments.
5. Wait for comments to load and analyze.
6. Download the analyzed CSV if needed.

Supported URL formats include:

```text
https://www.youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
https://www.youtube.com/shorts/VIDEO_ID
https://www.youtube.com/embed/VIDEO_ID
```


## Model Download After Cloning

Model files are not committed to this repository. The `models/` folder is ignored by git because transformer model weights can be very large.

When someone clones the project, they only need to install the Python dependencies and run the app:

```powershell
git clone <repo-url>
cd multilingual_sentiment_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The models download automatically the first time they are selected or used in the app.

### Recommended Model Download

If the user selects `XLM-RoBERTa Twitter Sentiment (recommended)`, Hugging Face downloads:

```text
cardiffnlp/twitter-xlm-roberta-base-sentiment
```

into:

```text
models/twitter-xlm-roberta
```

### Fallback Model Download

If the user selects `Multilingual BERT 1-5 Stars (fallback)`, or if the recommended model fails and the app falls back, Hugging Face downloads:

```text
nlptown/bert-base-multilingual-uncased-sentiment
```

into:

```text
models/nlptown
```

The download happens because `app.py` uses Hugging Face Transformers with `from_pretrained` and a local `cache_dir`:

```python
AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
AutoModelForSequenceClassification.from_pretrained(config.model_name, cache_dir=config.cache_dir)
```

Hugging Face checks the cache folder first. If the model is already present, it loads from disk. If the model is missing, it downloads it.

Internet access is required the first time each model is used. After the model is cached locally in `models/`, it can be loaded again without downloading.

## Model Cache

Downloaded model files are stored inside `models/`.

These files can be large and should not be committed. If the folder is deleted, Transformers will download the models again on the next run.

## Common Commands

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run app.py
```

Check Python syntax:

```powershell
python -m py_compile app.py
```

Clear Streamlit cache from the app UI:

```text
Menu -> Settings -> Clear cache
```

## Troubleshooting

If the recommended model does not load on first run, wait for the Hugging Face download to finish and try again. Large model downloads can be slow or interrupted.

If XLM-R tokenizer errors appear, reinstall the dependencies:

```powershell
python -m pip install --upgrade transformers sentencepiece hf_xet
```

If YouTube analysis fails, check that `.env` contains `YOUTUBE_API_KEY` and that the key has YouTube Data API v3 access enabled.

If model downloads are very slow on Windows, keep `hf_xet` installed. Hugging Face may fall back to regular HTTP downloads without it.

