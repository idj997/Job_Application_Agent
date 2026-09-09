import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RequirementNLIClassifier:

    MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        ).to(self.device)

        self.model.eval()

        self.id2label = self.model.config.id2label

    def classify(
        self,
        context: str,
        hypothesis: str
    ) -> dict:

        inputs = self.tokenizer(
            context,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        results = {}

        for idx, probability in enumerate(probabilities):
            label = self.id2label[idx].lower()

            results[label] = float(
                probability.item()
            )

        predicted_label = max(
            results,
            key=results.get
        )

        return {
            "label": predicted_label,
            "confidence": results[predicted_label],
            "probabilities": results,
        }