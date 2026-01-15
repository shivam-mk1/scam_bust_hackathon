from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

app = FastAPI()

# Absolute-safe path resolution (works locally + Railway)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "scam_model")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

class PredictRequest(BaseModel):
    text: str
    modality: str  # sms | whatsapp | call | audio

@app.post("/predict")
def predict(req: PredictRequest):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        scam_prob = probs[0][1].item()

    # ✅ Risk level mapping (FINAL, FIXED)
    if scam_prob >= 0.85:
        risk_level = "HIGH"
    elif scam_prob >= 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "is_scam": int(scam_prob >= 0.5),
        "confidence": round(scam_prob, 3),
        "risk_level": risk_level,
        "recommended_action": "DO_NOT_INTERACT" if scam_prob >= 0.5 else "SAFE"
    }
