import asyncio
from collections import Counter

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Roberta:
    CHUNK_SIZE = 500

    def __init__(self, model="cardiffnlp/twitter-xlm-roberta-base-sentiment") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.roberta_sentiment_pipeline = AutoModelForSequenceClassification.from_pretrained(model)

        self.label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

    async def predict(self, text: str) -> dict:
        model_inputs = self._build_model_inputs(text)
        outputs = await asyncio.to_thread(self.roberta_sentiment_pipeline, **model_inputs)
        logits = outputs.logits

        predicted_class_ids = logits.argmax(dim=1).tolist()
        winning_class_id = Counter(predicted_class_ids).most_common(1)[0][0]

        probabilities = logits.softmax(dim=1)
        winning_scores = [
            probabilities[idx][winning_class_id].item()
            for idx, predicted_class_id in enumerate(predicted_class_ids)
            if predicted_class_id == winning_class_id
        ]

        sentiment = self.label_mapping.get(f"LABEL_{winning_class_id}", "Unknown")
        score = sum(winning_scores) / len(winning_scores)

        return {"label": sentiment, "score": score}

    def _build_model_inputs(self, text: str) -> dict:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) <= self.CHUNK_SIZE:
            return self.tokenizer(text, return_tensors="pt", padding=True)

        chunked_token_ids = [
            token_ids[idx:idx + self.CHUNK_SIZE]
            for idx in range(0, len(token_ids), self.CHUNK_SIZE)
        ]

        encoded_chunks = [
            self.tokenizer.prepare_for_model(
                chunk,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors=None,
                truncation=False,
            )
            for chunk in chunked_token_ids
        ]

        return self.tokenizer.pad(encoded_chunks, padding=True, return_tensors="pt")
