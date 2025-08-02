import os
import zipfile
import shutil
from PyPDF2 import PdfReader
from docx import Document

def extract_zip(zip_file, extract_to="temp"):
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

def read_file(filepath):
    if filepath.endswith(".txt"):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif filepath.endswith(".docx"):
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])
    elif filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    else:
        return ""
