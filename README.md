# Instructor Engagement Sentiment Analysis

NLP pipeline and Streamlit apps that turn student course reviews into sentiment labels, topics, and keyword tags — built to help course teams spot instructor engagement issues at scale instead of reading thousands of reviews by hand.

## Project Overview

Given ~10,000 student reviews (rating, review text, student/instructor/course IDs), this project:

- Classifies each review's **sentiment** (positive / negative / neutral)
- Assigns a **dominant topic** via LDA topic modeling
- Extracts **keyword tags** per review with KeyBERT
- Surfaces the results through two Streamlit apps: a single-review analyzer and a batch results dashboard

## Repository Structure

```
.
├── app.py              # Streamlit app: analyze one review, or batch-score an uploaded CSV
├── dashboard.ipynb      # Streamlit dashboard notebook: explore the full results dataset
│                        #   (sentiment mix, rating distribution, topics, keyword word cloud)
├── preprocess.py        # Text cleaning shared by the analysis pipeline
├── requirements.txt
├── review_analysis_results.csv       # Scored output: review_id, sentiment, topic, keywords, confidence
├── sentiment_analysis_results.csv    # Source reviews joined with the same scoring output
└── .devcontainer/        # Codespaces config to launch app.py automatically
```

## Tech Stack

- Python, Streamlit
- scikit-learn (sentiment classifier + TF-IDF)
- gensim (LDA topic modeling)
- KeyBERT (keyword extraction)
- pandas, Plotly, Matplotlib, Seaborn, WordCloud

## Getting Started

```bash
pip install -r requirements.txt

# Single-review analyzer / batch CSV scorer
streamlit run app.py
```

`app.py` loads pretrained artifacts (`sentiment_model.pkl`, `tfidf_vectorizer.pkl`, `lda_model.gensim`, `dictionary.dict`) — train and save these first if they aren't already present in your environment.

To explore the full results dataset instead, open `dashboard.ipynb` in Jupyter, or convert it to a script and run it with Streamlit.

## Author

**Ezekiel Ebuetse** — [GitHub](https://github.com/datawithezekiel) · [LinkedIn](https://linkedin.com/in/ezekiel-ebuetse) · [Portfolio](https://ezekielebuetse.com)
