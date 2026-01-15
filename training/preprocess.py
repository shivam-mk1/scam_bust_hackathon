import pandas as pd
import re
import os

def clean_text(text):
    """Clean and normalize text data"""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'http\S+', ' URL ', text)
    text = re.sub(r'\b\d{4,6}\b', ' OTP ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_all():
    """Load and preprocess all data sources"""
    
    # Use absolute paths for reliability
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data.csv")
    
    print(f"Loading data from: {DATA_DIR}")
    
    # Load SMS data
    sms = pd.read_csv(os.path.join(DATA_DIR, "public_sms.csv"))
    sms["text"] = sms["message_text"]
    sms["modality"] = "sms"
    print(f"Loaded {len(sms)} SMS messages")

    # Load WhatsApp data
    wa = pd.read_csv(os.path.join(DATA_DIR, "public_whatsapp.csv"))
    wa["text"] = wa["conversation_text"]
    wa["modality"] = "whatsapp"
    print(f"Loaded {len(wa)} WhatsApp messages")

    # Load call data
    calls = pd.read_csv(os.path.join(DATA_DIR, "public_calls.csv"))
    calls["text"] = calls["call_transcript"]
    calls["modality"] = "call"
    print(f"Loaded {len(calls)} call transcripts")

    # Load audio data
    audio = pd.read_csv(os.path.join(DATA_DIR, "public_audio_transcripts.csv"))
    audio["text"] = audio["audio_transcript_text"]
    audio["modality"] = "audio"
    print(f"Loaded {len(audio)} audio transcripts")

    # Combine all data
    df = pd.concat([sms, wa, calls, audio], ignore_index=True)
    df = df[["text", "modality", "is_scam", "severity"]]
    
    # Clean text
    print("Cleaning text data...")
    df["clean_text"] = df["text"].apply(clean_text)
    
    # Save processed data
    print(f"Saving processed data to: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Preprocessing complete! Total samples: {len(df)}")
    print(f"Scam distribution: {df['is_scam'].value_counts().to_dict()}")

if __name__ == "__main__":
    load_all()
