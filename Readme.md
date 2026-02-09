
# Pipeline Fine-Tuning LLM untuk Regulasi Kesehatan (JDIH)

Proyek ini bertujuan untuk melatih Large Language Model (LLM) agar dapat menjawab pertanyaan seputar **Peraturan Menteri Kesehatan** (khususnya JDIH) secara akurat. Pipeline ini mengotomatisasi proses dari ekstraksi data mentah PDF hingga proses Fine-Tuning menggunakan teknik QLoRA.

## 🧠 Metodologi & Pendekatan

Kekuatan utama dari pipeline ini terletak pada strategi **Data Engineering** yang diterapkan sebelum model dilatih. saya menyadari bahwa kualitas data lebih penting daripada sekadar ukuran data (*Data-Centric AI*).

### 1. Structure-Aware Dataset Generation (`processing.py`)
Berbeda dengan metode *naive chunking* (memotong teks berdasarkan jumlah token tetap) yang sering memutus kalimat atau konteks, saya menggunakan metode **Semantic & Structural Parsing**:

* **Logic-Based Cleaning:** Script secara cerdas menyambung kembali baris yang terputus (broken lines) akibat format PDF dengan mendeteksi *glue words* (kata sambung seperti "dan", "yang", "dalam") di akhir baris.
* **Legal Structure Parsing:** Dokumen dipecah berdasarkan struktur hierarki hukum (Menimbang, Mengingat, Memutuskan, Pasal demi Pasal). Ini memastikan setiap *chunk* data memiliki konteks hukum yang utuh.
* **Definition Extraction:** Algoritma khusus dibuat untuk mendeteksi "Pasal 1" (biasanya berisi definisi) dan mengekstrak setiap istilah menjadi pasangan Q&A terpisah. Ini sangat krusial untuk akurasi terminologi.

### 2. Instruction Augmentation (`augmented.py`)
Untuk mencegah model menghafal jawaban (*overfitting*) dan meningkatkan kemampuan generalisasi, saya menerapkan **Rule-Based Augmentation**:

* **Template Variabel:** saya tidak mengubah *output* (isi pasal), tetapi memvariasikan *instruction* (pertanyaan pengguna).
* **Context Detection:** Script mendeteksi konteks instruksi asli (apakah tentang `definisi`, `tujuan`, atau `tugas`) dan menerapkan template pertanyaan yang relevan secara acak.
    * *Contoh:* "Apa definisi X?" bisa diubah menjadi "Jelaskan pengertian X menurut regulasi ini" atau "Definisikan istilah X."
* **Dampak:** Teknik ini melipatgandakan variasi linguistik data latih, membuat model lebih siap menghadapi berbagai gaya bertanya dari pengguna akhir.

### 3. Efficient Fine-Tuning (`Finetuning QLora.py`)
Menggunakan library **Unsloth** dan teknik **QLoRA** (Quantized Low-Rank Adaptation):
* **4-bit Quantization:** Memungkinkan pelatihan model besar (Qwen) pada GPU dengan VRAM terbatas (seperti T4 di Google Colab).
* **LoRA Adapters:** Hanya melatih sebagian kecil parameter model, mempercepat proses training hingga 2-5x lipat dibanding metode konvensional.

---

## 🛠️ Instalasi & Persiapan

1. **Persyaratan Sistem:**
   * Python 3.10+
   * GPU dengan dukungan CUDA (NVIDIA T4/RTX 30xx ke atas direkomendasikan).

2. **Instalasi Dependensi:**
   ```bash
   pip install -r requirements.txt


3. **File Input:**
Pastikan file PDF peraturan (contoh: `permenkes-no-10-tahun-2024.pdf`) berada di direktori yang sama dengan script.

---

## 🚀 Cara Menjalankan

Cukup jalankan satu perintah berikut untuk mengeksekusi seluruh pipeline (Cleaning -> Augmentasi -> Training):

```bash
python run_pipeline.py

```

### Output

Setelah proses selesai, Anda akan mendapatkan:

1. `"dataset_final.jsonl`: Dataset final yang siap pakai.
2. `outputs/`: Folder berisi adapter model LoRA yang telah dilatih.
3. `hasil_perbandingan_final.txt`: Laporan evaluasi *Before vs After* fine-tuning.
