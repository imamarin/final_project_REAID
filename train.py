from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

# === Konfigurasi ===
MODEL_NAME = "indobenchmark/indobert-base-p1"  # atau distilbert-base-uncased
DATA_PATH = "train.jsonl"
NUM_EPOCHS = 5
BATCH_SIZE = 4
OUTPUT_DIR = "model_output"

# === Load dan encode dataset ===
raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
label_encoder = LabelEncoder()
raw_dataset = raw_dataset.train_test_split(test_size=0.2)

# Encode label jadi angka
label_encoder.fit(raw_dataset['train']['label'])
num_labels = len(label_encoder.classes_)

def encode_labels(example):
    example['label'] = label_encoder.transform([example['label']])[0]
    return example

dataset = raw_dataset.map(encode_labels)

# === Tokenisasi ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# === Load model ===
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

# === Trainer ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == labels)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# === Mulai training ===
trainer.train()

# === Simpan model + label encoder ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Simpan label encoder ke file
import joblib
joblib.dump(label_encoder, f"{OUTPUT_DIR}/label_encoder.pkl")

print("✅ Training selesai dan model disimpan di:", OUTPUT_DIR)
