from transformers import AutoTokenizer, LongformerForSequenceClassification, LongformerTokenizer, AutoModelForSequenceClassification
from sentiment_analysis.log import logger


class SentimentAnalysis:
    def __init__(self, model="cardiffnlp/twitter-xlm-roberta-base-sentiment") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.roberta_sentiment_pipeline = AutoModelForSequenceClassification.from_pretrained(model)
        self.longformer_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.longformer_model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=3)

        self.label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze the sentiment of the given text. Select model based on text length.
        """
        if not text.strip():
            logger.info("Empty or unparseable text provided")
            raise ValueError("Empty or unparseable text provided")

        try:
            logger.debug(f"Analyzing sentiment for text: {text[:20]} - Length: {len(text)}")
            if len(text) > 15:
                return self._get_longform_sentiment_prediction(text)
            return self._get_sentiment_prediction(text)
        except Exception as e:
            logger.exception()
            raise ValueError("Error while analyzing sentiment") from e

    def _get_sentiment_prediction(self, text: str):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        outputs = self.roberta_sentiment_pipeline(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]  # Assuming batch size of 1
        predicted_class_id = logits.argmax().item()

        sentiment = self.label_mapping.get(f"LABEL_{predicted_class_id}", "Unknown")
        score = logits.softmax(dim=0)[predicted_class_id].item()

        return {"label": sentiment, "score": score}

    def _get_longform_sentiment_prediction(self, text: str) -> dict:
        inputs = self.longformer_tokenizer(text, return_tensors="pt", padding=True)
        outputs = self.longformer_model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()

        sentiment = self.label_mapping.get(f"LABEL_{predicted_class_id}", "Unknown")
        return {"label": sentiment, "score": logits.softmax(dim=1).max().item()}
