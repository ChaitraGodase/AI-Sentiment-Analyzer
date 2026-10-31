from transformers import pipeline


class SentimentAnalyzer:
    """Wrapper around a Hugging Face text-classification pipeline."""

    def __init__(self, model_name: str):
        self.classifier = pipeline(
            task="text-classification",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
        )

    def predict(self, text: str) -> dict:
        clean_text = str(text).strip()
        if not clean_text:
            return {"sentiment": "Unknown", "confidence": 0.0}

        raw = self.classifier(clean_text, truncation=True, max_length=512)
        scores = raw[0] if raw and isinstance(raw[0], list) else raw
        best = max(scores, key=lambda item: float(item["score"]))
        label = str(best["label"]).lower()

        labels = {
            "label_0": "Negative",
            "label_1": "Neutral",
            "label_2": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
            "positive": "Positive",
        }
        return {
            "sentiment": labels.get(label, label.replace("_", " ").title()),
            "confidence": float(best["score"]),
        }
