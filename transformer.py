from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi, create_repo, upload_folder

REPO_ID = "imamamirulloh/kenandocsclassification"  # ganti sesuai
LOCAL_DIR = "model_output"

# Buat repo (opsional, skip kalau sudah ada)
create_repo(REPO_ID, exist_ok=True)

# Upload seluruh folder model_output/
upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    ignore_patterns=[]
)
