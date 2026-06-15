import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_MODEL = "facebook/bart-large-cnn"
ADAPTER_PATH = "./bart_qlora_finetuned_colab"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()


def summarize(review_text: str) -> str:
    inputs = tokenizer(
        review_text,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
