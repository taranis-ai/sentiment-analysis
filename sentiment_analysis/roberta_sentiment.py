from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentiment_analysis.predictor import Predictor


class RobertaSentiment(Predictor):
    def __init__(self, model="cardiffnlp/twitter-xlm-roberta-base-sentiment") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.roberta_sentiment_pipeline = AutoModelForSequenceClassification.from_pretrained(model)

        self.label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

    def predict(self, text: str) -> dict:
        tokens = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        outputs = self.roberta_sentiment_pipeline(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]  # Assuming batch size of 1
        predicted_class_id = logits.argmax().item()

        sentiment = self.label_mapping.get(f"LABEL_{predicted_class_id}", "Unknown")
        score = logits.softmax(dim=0)[predicted_class_id].item()

        return {"label": sentiment, "score": score}
