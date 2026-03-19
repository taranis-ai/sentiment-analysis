import asyncio

from transformers import LongformerForSequenceClassification, LongformerTokenizer


class Longformer:
    def __init__(self) -> None:
        self.longformer_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.longformer_model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=3)

        self.label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

    async def predict(self, text: str) -> dict:
        inputs = self.longformer_tokenizer(text, return_tensors="pt", padding=True)
        outputs = await asyncio.to_thread(self.longformer_model, **inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()

        sentiment = self.label_mapping.get(f"LABEL_{predicted_class_id}", "Unknown")
        return {"label": sentiment, "score": logits.softmax(dim=1).max().item()}
