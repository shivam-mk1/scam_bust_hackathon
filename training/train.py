import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split

# Use absolute paths for reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed_data.csv")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "model")
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "scam_model")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print(f"Dataset size: {len(df)} samples")
print(f"Scam distribution: {df['is_scam'].value_counts().to_dict()}")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class ScamDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=256
        )
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Split data into train and validation sets (80/20 split)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["clean_text"],
    df["is_scam"],
    test_size=0.2,
    random_state=42,
    stratify=df["is_scam"]  # Maintain class distribution
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

train_dataset = ScamDataset(train_texts, train_labels)
val_dataset = ScamDataset(val_texts, val_labels)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
    eval_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",
    save_total_limit=2,  # Keep only best 2 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset  # Add validation dataset
)

print("Starting training...")
trainer.train()

print(f"Saving model to: {MODEL_SAVE_PATH}")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("✅ Training complete! Model saved successfully.")
