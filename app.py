import streamlit as st
import os
import shutil
import zipfile
from classifier import classify_text
from utils import extract_zip, read_file

# Judul aplikasi
st.title("📁 Klasifikasi Dokumen AI")
st.write("Upload file `.zip` berisi dokumen, dan sistem akan mengklasifikasikan ke kategori yang dipilih.")

# Sidebar kategori + kontrol
with st.sidebar:
    st.header("🔘 Pilih Kategori Klasifikasi")

    # Daftar kategori + key session_state
    label_list = [
        ("Sertifikat", "sertifikat"),
        ("Ijazah", "ijazah"),
        ("Surat Tugas", "surat_tugas"),
        ("Surat Keterangan", "surat_keterangan"),
        ("Paper Jurnal", "paper_jurnal"),
        ("Surat Kerja Sama", "surat_mou"),
    ]

    # Inisialisasi session_state
    for _, key in label_list:
        if key not in st.session_state:
            st.session_state[key] = True

    # Tombol Select/Unselect All
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select All"):
            for _, key in label_list:
                st.session_state[key] = True
    with col2:
        if st.button("Unselect All"):
            for _, key in label_list:
                st.session_state[key] = False

    # Tampilkan checkbox
    selected_labels = []
    for label_name, key in label_list:
        if st.checkbox(label_name, key=key):
            selected_labels.append(label_name)

# Upload ZIP file
uploaded_zip = st.file_uploader("📤 Upload file ZIP", type=["zip"])

if uploaded_zip:
    if not selected_labels:
        st.warning("⚠️ Silakan pilih minimal satu kategori di sidebar.")
        st.stop()

    # Simpan file ZIP
    with open("uploaded.zip", "wb") as f:
        f.write(uploaded_zip.getbuffer())

    # Ekstrak file
    extract_path = extract_zip("uploaded.zip")
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Inisialisasi struktur hasil
    categorized = {label: [] for label in selected_labels}

    # Proses file satu per satu
    st.info("⏳ Sedang mengklasifikasikan dokumen...")
    files = [f for f in os.listdir(extract_path) if os.path.isfile(os.path.join(extract_path, f))]
    progress = st.progress(0)

    for idx, file in enumerate(files):
        filepath = os.path.join(extract_path, file)
        text = read_file(filepath)

        if text.strip() == "":
            continue

        # Klasifikasi
        label = classify_text(text, selected_labels)
        if label is None:
            continue  # skip dokumen jika tidak cocok kategori
        categorized[label].append(filepath)
        progress.progress((idx + 1) / len(files))

    st.success("✅ Klasifikasi selesai!")

    # Tampilkan tombol download per kategori
    for category, files in categorized.items():
        if not files:
            continue

        # Buat folder kategori
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        for f in files:
            shutil.copy(f, cat_dir)

        # ZIP-kan folder
        zip_filename = f"{category}.zip"
        zip_path = os.path.join(output_dir, zip_filename)
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in os.listdir(cat_dir):
                zf.write(os.path.join(cat_dir, f), arcname=f)

        # Tombol download
        with open(zip_path, "rb") as f:
            st.download_button(
                label=f"⬇️ Download Kategori: {category}",
                data=f,
                file_name=zip_filename,
                mime="application/zip"
            )
