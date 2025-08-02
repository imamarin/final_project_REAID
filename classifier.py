from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import torch
import joblib
import numpy as np

# Konfigurasi
REPO_ID = "imamamirulloh/kenandocsclassification"  # Ganti dengan repo Hugging Face kamu
MODEL_FILENAME = "pytorch_model.bin"  # Default file model
CONFIG_FILENAME = "config.json"
TOKENIZER_DIR = "."  # bisa diatur ke repo_id jika tokenizernya juga di-host di sana
LABEL_ENCODER_FILENAME = "label_encoder.pkl"

# Muat tokenizer dan model dari Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
model = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
model.eval()  # Set model ke evaluasi (non-training)

# Muat label encoder dari Hugging Face Hub
label_encoder_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=LABEL_ENCODER_FILENAME,
    repo_type="model",
    # token="hf_..." jika model kamu private
)
label_encoder = joblib.load(label_encoder_path)

def classify_text(text, selected_labels=None):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
    
    label = label_encoder.inverse_transform([predicted_class_id])[0]

    # ✅ Filter hanya yang dipilih user
    if selected_labels and label not in selected_labels:
        return None  # atau "Lainnya" jika kamu mau
    return label
