from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Scam Detection API",
    description="ML-powered API for detecting scam messages across SMS, WhatsApp, calls, and audio",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Absolute-safe path resolution (works locally + Railway)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "scam_model")

# Global variables for model and tokenizer
tokenizer = None
model = None
model_loaded = False

def load_model():
    """Load the model and tokenizer with error handling"""
    global tokenizer, model, model_loaded
    
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model directory not found at: {MODEL_PATH}")
            logger.info("Please train the model first by running: cd training && python train.py")
            return False
        
        logger.info(f"Loading model from: {MODEL_PATH}")
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        model_loaded = True
        logger.info("Model loaded successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Attempt to load model on startup
load_model()

class PredictRequest(BaseModel):
    text: str
    modality: str  # sms | whatsapp | call | audio

@app.get("/")
def health_check():
    """Health check endpoint for Railway and monitoring"""
    return {
        "status": "healthy",
        "service": "scam-detection-api",
        "version": "1.0.0",
        "model_loaded": model_loaded
    }

@app.get("/health")
def health():
    """Detailed health check"""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    return {
        "status": "healthy",
        "model_status": "loaded",
        "model_path": MODEL_PATH
    }

@app.post("/predict")
def predict(req: PredictRequest):
    """Predict if a message is a scam"""
    
    # Check if model is loaded
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running: cd training && python train.py"
        )
    
    # Validate input
    if not req.text or not req.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text field cannot be empty"
        )
    
    valid_modalities = ["sms", "whatsapp", "call", "audio"]
    if req.modality.lower() not in valid_modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid modality. Must be one of: {', '.join(valid_modalities)}"
        )
    
    try:
        # Tokenize input
        inputs = tokenizer(
            req.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        # Make prediction
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            scam_prob = probs[0][1].item()

        # Risk level mapping
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
            "recommended_action": "DO_NOT_INTERACT" if scam_prob >= 0.5 else "SAFE",
            "modality": req.modality.lower()
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
