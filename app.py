import pandas as pd
import plotly.express as px
import streamlit as st
from sentiment_model import SentimentAnalyzer

st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="💬", layout="wide")
st.title("💬 AI-Powered Sentiment Analyzer")
st.caption("Analyze customer feedback using a pretrained NLP transformer.")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

@st.cache_resource(show_spinner="Loading AI model (first run may take a few minutes)...")
def load_analyzer():
    return SentimentAnalyzer(MODEL_NAME)

with st.sidebar:
    st.header("About")
    st.write("Classifies text as positive, neutral, or negative.")
    st.caption("Model: CardiffNLP Twitter-RoBERTa")

single_tab, csv_tab, about_tab = st.tabs(["Single Text", "Bulk CSV Analysis", "About"])

with single_tab:
    text = st.text_area("Enter a review or comment", height=150,
                        placeholder="The product is excellent and delivery was fast!")
    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                result = load_analyzer().predict(text)
                col1, col2 = st.columns(2)
                col1.metric("Sentiment", result["sentiment"])
                col2.metric("Confidence", f'{result["confidence"]:.1%}')
                st.progress(result["confidence"])
                st.caption("Confidence is the model score, not a guarantee of correctness.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

with csv_tab:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if df.empty:
                st.warning("The CSV has no rows.")
            else:
                st.dataframe(df.head(), use_container_width=True)
                column = st.selectbox("Select the text/review column", df.columns)
                if st.button("Analyze CSV", type="primary"):
                    analyzer = load_analyzer()
                    sentiments, confidence = [], []
                    progress = st.progress(0)
                    for i, value in enumerate(df[column].fillna("").astype(str)):
                        result = analyzer.predict(value)
                        sentiments.append(result["sentiment"])
                        confidence.append(result["confidence"])
                        progress.progress((i + 1) / len(df))
                    output = df.copy()
                    output["predicted_sentiment"] = sentiments
                    output["confidence"] = confidence
                    st.success("Analysis complete.")
                    st.dataframe(output, use_container_width=True)

                    counts = output["predicted_sentiment"].value_counts().rename_axis(
                        "Sentiment").reset_index(name="Reviews")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(px.pie(counts, names="Sentiment",
                            values="Reviews", hole=0.4, title="Sentiment distribution"),
                            use_container_width=True)
                    with c2:
                        st.plotly_chart(px.bar(counts, x="Sentiment", y="Reviews",
                            color="Sentiment", title="Reviews by sentiment"),
                            use_container_width=True)
                    st.download_button("Download results as CSV",
                        output.to_csv(index=False).encode("utf-8"),
                        "sentiment_results.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")

with about_tab:
    st.subheader("How it works")
    st.markdown("""
    1. Enter a review or upload a CSV file.
    2. A pretrained Transformer model evaluates the text.
    3. The app displays the predicted sentiment and confidence score.
    4. CSV results are visualized and can be downloaded.

    **Limitation:** Predictions can be wrong for sarcasm, mixed opinions,
    slang, or domain-specific text. Review results before making decisions.
    """)
