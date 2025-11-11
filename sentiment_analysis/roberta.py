import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Roberta:
    def __init__(self, model="cardiffnlp/twitter-xlm-roberta-base-sentiment") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.roberta_sentiment_pipeline = AutoModelForSequenceClassification.from_pretrained(model)

        self.label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

    def predict(self, text: str) -> dict:
        model_inputs = self.tokenize_and_reshape(text)
        logits = self.roberta_sentiment_pipeline(**model_inputs).logits

        predicted_class_id = Counter(logits.argmax(axis=1).tolist()).most_common()[0][0]
        sentiment = self.label_mapping.get(f"LABEL_{predicted_class_id}", "Unknown")
        
        # average the score of the predicted label for each chunk
        score = logits.softmax(axis=1).gather(dim=1, index=logits.argmax(axis=1, keepdims=True)).mean().item()

        return {"label": sentiment, "score": score}
    
    def tokenize_and_reshape(self, text: str, chunk_size: int = 500) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if remainder := len(tokens) % chunk_size:
            tokens += [1] * (chunk_size - remainder)

        input_ids = torch.tensor(tokens).view(-1, chunk_size)
        attention_mask = (input_ids != 1).long()

        return {"input_ids": input_ids, "attention_mask": attention_mask}
