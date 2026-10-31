from transformers import pipeline


class SentimentAnalyzer:

    def __init__(self, model_name):
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name
        )

    def predict(self, text):

        if not str(text).strip():
            return {
                "sentiment": "Unknown",
                "confidence": 0.0
            }

        result = self.classifier(
            str(text),
            truncation=True,
            max_length=512
        )[0]

        label = result["label"].lower()

        labels = {
            "label_0": "Negative",
            "label_1": "Neutral",
            "label_2": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
            "positive": "Positive"
        }

        return {
            "sentiment": labels.get(label, label.title()),
            "confidence": float(result["score"])
        }