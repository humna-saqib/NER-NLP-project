from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

app = FastAPI(title="NER API")

MODEL_PATH = "./ner_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "NER FastAPI backend is running"}

@app.post("/predict")
def predict(data: TextInput):
    text = data.text

    if not text.strip():
        return {"entities": []}

    words = text.split()

    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    word_ids = inputs.word_ids()

    results = []
    previous_word_id = None

    for word_id, pred_id in zip(word_ids, predictions):
        if word_id is not None and word_id != previous_word_id:
            label = model.config.id2label[pred_id]
            word = words[word_id]

            if label != "O":
                results.append({
                    "word": word,
                    "label": label
                })

        previous_word_id = word_id

    return {"entities": results}