import io

import pandas as pd
import plotly.express as px
import streamlit as st

from sentiment_model import SentimentAnalyzer


st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
)

st.title("💬 AI-Powered Sentiment Analyzer")
st.caption("Analyze customer feedback using a pretrained NLP transformer.")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@st.cache_resource(show_spinner="Loading AI model (first run may take a few minutes)...")
def load_analyzer():
    return SentimentAnalyzer(MODEL_NAME)


def read_uploaded_csv(uploaded_file):
    """Read CSV files exported with common encodings such as UTF-8 and Windows-1252."""
    file_bytes = uploaded_file.getvalue()
    encodings = ("utf-8-sig", "cp1252", "latin1")
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"Could not decode this CSV. Last encoding error: {last_error}")


with st.sidebar:
    st.header("About")
    st.write("Classifies text as positive, neutral, or negative.")
    st.caption("Model: CardiffNLP Twitter-RoBERTa")
    st.info("Use public or sample reviews. Avoid uploading confidential information.")


single_tab, csv_tab, about_tab = st.tabs(
    ["Single Text", "Bulk CSV Analysis", "About"]
)


with single_tab:
    st.subheader("Analyze one customer review")
    user_text = st.text_area(
        "Enter a review or comment",
        height=150,
        placeholder="The product quality is excellent and delivery was fast!",
    )

    if st.button("Analyze sentiment", type="primary", key="single_analysis"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                with st.spinner("Analyzing text..."):
                    result = load_analyzer().predict(user_text)

                col1, col2 = st.columns(2)
                col1.metric("Predicted sentiment", result["sentiment"])
                col2.metric("Confidence", f'{result["confidence"]:.1%}')
                st.progress(result["confidence"])
                st.caption(
                    "Confidence is the model score, not a guarantee that the prediction is correct."
                )
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


with csv_tab:
    st.subheader("Analyze multiple customer reviews")
    st.write("Upload a CSV file and select the column containing the review text.")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Supports UTF-8, UTF-8 with BOM, and common Windows CSV encodings.",
    )

    if uploaded_file is not None:
        try:
            df = read_uploaded_csv(uploaded_file)

            if df.empty:
                st.warning("The uploaded CSV contains no rows.")
            elif len(df.columns) == 0:
                st.warning("No columns were found in the uploaded CSV.")
            else:
                st.success(f"CSV loaded successfully: {len(df):,} rows and {len(df.columns)} columns.")
                st.write("File preview")
                st.dataframe(df.head(10), use_container_width=True)

                text_column = st.selectbox(
                    "Select the column containing customer reviews",
                    options=list(df.columns),
                )

                if st.button("Analyze CSV", type="primary", key="bulk_analysis"):
                    analyzer = load_analyzer()
                    texts = df[text_column].fillna("").astype(str).tolist()

                    sentiments = []
                    confidences = []
                    progress = st.progress(0)
                    status = st.empty()

                    for index, review in enumerate(texts):
                        prediction = analyzer.predict(review)
                        sentiments.append(prediction["sentiment"])
                        confidences.append(prediction["confidence"])
                        progress.progress((index + 1) / len(texts))
                        status.write(f"Analyzed {index + 1:,} of {len(texts):,} reviews")

                    output = df.copy()
                    output["predicted_sentiment"] = sentiments
                    output["confidence"] = confidences

                    status.empty()
                    st.success("Sentiment analysis completed.")

                    positive = sum(value == "Positive" for value in sentiments)
                    neutral = sum(value == "Neutral" for value in sentiments)
                    negative = sum(value == "Negative" for value in sentiments)

                    metric1, metric2, metric3 = st.columns(3)
                    metric1.metric("Positive", f"{positive:,}")
                    metric2.metric("Neutral", f"{neutral:,}")
                    metric3.metric("Negative", f"{negative:,}")

                    st.subheader("Sentiment results")
                    st.dataframe(output, use_container_width=True)

                    counts = (
                        output["predicted_sentiment"]
                        .value_counts()
                        .rename_axis("Sentiment")
                        .reset_index(name="Reviews")
                    )

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        pie = px.pie(
                            counts,
                            names="Sentiment",
                            values="Reviews",
                            hole=0.45,
                            title="Sentiment distribution",
                        )
                        st.plotly_chart(pie, use_container_width=True)

                    with chart_col2:
                        bar = px.bar(
                            counts,
                            x="Sentiment",
                            y="Reviews",
                            color="Sentiment",
                            title="Review count by sentiment",
                        )
                        st.plotly_chart(bar, use_container_width=True)

                    csv_output = output.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="Download analyzed results",
                        data=csv_output,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV is empty or has no readable header.")
        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")


with about_tab:
    st.subheader("How this application works")
    st.markdown(
        """
        1. Enter a review or upload a CSV file.
        2. A pretrained Transformer model analyzes the text.
        3. The model predicts positive, neutral, or negative sentiment.
        4. The app displays confidence scores and summary charts.
        5. Bulk results can be downloaded as a CSV file.

        **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`

        **Limitations:** Predictions may be inaccurate for sarcasm, mixed opinions,
        slang, or text outside the model's training domain. Review results before
        making business decisions.
        """
    )
