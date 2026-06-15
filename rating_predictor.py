import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./bart_rating_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


def predict_rating(summary: str) -> float:
    inputs = tokenizer(
        summary,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        rating = torch.argmax(outputs.logits, dim=1).item() + 1

    return rating
