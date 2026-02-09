
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

1. `dataset_augmented_v1.jsonl`: Dataset final yang siap pakai.
2. `outputs/`: Folder berisi adapter model LoRA yang telah dilatih.
3. `hasil_perbandingan_final.txt`: Laporan evaluasi *Before vs After* fine-tuning.

```

---

### 3. Cara Menjalankan Script (Langkah demi Langkah)

Berikut adalah panduan praktis untuk Anda:

#### Langkah 1: Siapkan Lingkungan (Folder)
Buat satu folder baru, lalu masukkan semua file berikut ke dalamnya:
1.  `script_final.py` (Script asli Anda)
2.  `augmented.py` (Script asli Anda)
3.  `Finetuning QLora.py` (Script asli Anda)
4.  `requirements.txt` (File dependensi Anda)
5.  **`run_pipeline.py`** (Script baru yang saya buatkan di atas)
6.  `permenkes-no-10-tahun-2024.pdf` (File PDF target. **Penting:** Pastikan nama file ini sesuai dengan yang tertulis di dalam `script_final.py` atau ubah kodenya jika nama file Anda berbeda).

#### Langkah 2: Install Library
Buka terminal/command prompt di folder tersebut, lalu jalankan:

```bash
pip install -r requirements.txt

```

*(Catatan: Jika Anda menjalankan ini di Windows lokal tanpa GPU NVIDIA, langkah Fine-tuning mungkin akan sangat lambat atau error karena Unsloth/bitsandbytes sangat bergantung pada CUDA. Script ini paling optimal berjalan di Google Colab atau Linux server dengan GPU).*

#### Langkah 3: Eksekusi Pipeline

Jalankan perintah berikut di terminal:

```bash
python run_pipeline.py

```

#### Apa yang akan terjadi?

1. Script akan membaca PDF dan membuat `dataset_finetuning.jsonl`.
2. Script akan membaca JSONL tersebut, melakukan augmentasi, dan menyimpan `dataset_final.jsonl`.
3. `run_pipeline.py` akan otomatis me-rename file menjadi `dataset_augmented_v1.jsonl` dan memperbaiki path di script fine-tuning agar tidak error.
4. Proses training dimulai. Anda bisa melihat *progress bar* berjalan.
5. Hasil akhir berupa perbandingan jawaban model sebelum dan sesudah training akan muncul di layar dan tersimpan di file txt.
