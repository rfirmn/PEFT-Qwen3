def install_dependencies():
    """Instalasi dependensi jika berjalan di Google Colab."""
    import os, re, torch
    if "COLAB_" in "".join(os.environ.keys()):
        print("📦 Mendeteksi Google Colab, menginstal dependensi...")
        
        # Deteksi versi torch untuk xformers
        v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
        xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
        
        # 1. Instalasi library inti tanpa dependensi untuk mencegah konflik
        os.system(f"pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo")
        
        # 2. Perbaikan: Tambahkan spasi di antara datasets dan huggingface_hub
        os.system("pip install sentencepiece protobuf hf_transfer")
        os.system("pip install 'huggingface_hub>=0.34.0'")
        
        # 3. Instalasi Unsloth dan Transformers
        os.system("pip install --no-deps unsloth")
        os.system("pip install transformers==4.56.2")
        os.system("pip install --no-deps trl==0.22.2")
        
        print("✅ Instalasi selesai. Silakan restart runtime jika perlu.")

# jika di collab langsung install dependensi, jika tidak, asumsikan sudah terinstal
install_dependencies()

import os
import re
import torch
import json
import sys
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 1. KONFIGURASI & PARAMETER (TIDAK DIUBAH)
# ==========================================
MODEL_NAME = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 1024
DTYPE = None
LOAD_IN_4BIT = True
DATASET_PATH = "dataset_final.jsonl" # Pastikan file ini ada

# Parameter LoRA
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]

# Parameter Training
TRAIN_BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
WARMUP_STEPS = 5
MAX_STEPS = 60
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
SEED = 3407
OUTPUT_DIR = "outputs"

# System Prompt & Pertanyaan Test
SYSTEM_INSTRUCTION = """
Anda adalah asisten hukum cerdas Kementerian Kesehatan.
Tugas Anda adalah menjawab pertanyaan pengguna secara akurat berdasarkan
Peraturan Menteri Kesehatan tentang Jaringan Dokumentasi dan Informasi Hukum (JDIH).
"""

DAFTAR_PERTANYAAN = [
    "Siapa yang melaksanakan tugas sebagai Pusat JDIH Kemenkes?",
    "Sebutkan siapa saja yang termasuk dalam anggota JDIH Kemenkes!",
    "Apa saja tugas dari Pusat JDIH Kemenkes menurut peraturan ini?",
    "Melalui website apa pengelolaan dokumentasi JDIH dilakukan?",
    "Apa definisi dari Dokumen Hukum dalam peraturan ini?"
]

# Template Prompt (Alpaca Style)
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""



def install_dependencies():
    """Instalasi dependensi jika berjalan di Google Colab."""
    if "COLAB_" in "".join(os.environ.keys()):
        print("📦 Mendeteksi Google Colab, menginstal dependensi...")
        v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
        xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
        os.system(f"pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo")
        os.system("pip install sentencepiece protobuf 'huggingface_hub>=0.34.0' hf_transfer")
        os.system("pip install --no-deps unsloth")
        os.system("pip install transformers==4.56.2")
        os.system("pip install --no-deps trl==0.22.2")

def load_base_model():
    """Memuat model dasar dan tokenizer."""
    print(f"🔄 Memuat Model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )
    return model, tokenizer

def apply_lora_adapters(model):
    """Menerapkan konfigurasi LoRA pada model."""
    print("🔧 Menerapkan Adapter LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_R,
        target_modules = TARGET_MODULES,
        lora_alpha = LORA_ALPHA,
        lora_dropout = LORA_DROPOUT,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = SEED,
        use_rslora = False,
        loftq_config = None,
    )
    return model

def generate_response(model, tokenizer, question):
    """Fungsi tunggal untuk menghasilkan jawaban dari model."""
    FastLanguageModel.for_inference(model)
    
    inputs = tokenizer(
        [ALPACA_PROMPT.format(SYSTEM_INSTRUCTION, question, "")],
        return_tensors = "pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    if "### Response:" in decoded:
        return decoded.split("### Response:")[1].strip()
    return decoded

def jalankan_pengujian(model, tokenizer, label_fase):
    """
    Menjalankan loop pertanyaan dan mengembalikan dictionary hasil.
    label_fase: string, misal 'BEFORE' atau 'AFTER'
    """
    print(f"\n🔍 MEMULAI PENGUJIAN FASE: {label_fase}")
    hasil = {}
    
    for i, tanya in enumerate(DAFTAR_PERTANYAAN):
        print(f"   Processing Q{i+1}...", end="\r")
        jawaban = generate_response(model, tokenizer, tanya)
        hasil[tanya] = jawaban
    
    print(f"\n✅ Pengujian {label_fase} Selesai.")
    return hasil

def formatting_prompts_func(examples):
    """Format dataset untuk training."""
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    EOS_TOKEN = tokenizer.eos_token # Akan diambil dari scope global saat dipanggil
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def train_model(model, tokenizer):
    """Proses Fine-tuning."""
    print("\n🚀 Memulai Proses Training...")
    
    # Load Dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # Inject tokenizer ke scope formatting function (teknik closure sederhana)
    # atau pastikan tokenizer global tersedia
    formatted_dataset = dataset.map(lambda x: formatting_prompts_func(x), batched = True)
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = TRAIN_BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUMULATION,
            warmup_steps = WARMUP_STEPS,
            max_steps = MAX_STEPS,
            learning_rate = LEARNING_RATE,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = WEIGHT_DECAY,
            lr_scheduler_type = "linear",
            seed = SEED,
            output_dir = OUTPUT_DIR,
        ),
    )
    
    trainer_stats = trainer.train()
    print("✅ Training Selesai.")
    return model

def cetak_perbandingan(hasil_before, hasil_after):
    """Mencetak output side-by-side."""
    print("\n" + "="*60)
    print("📊 LAPORAN PERBANDINGAN (BEFORE vs AFTER)")
    print("="*60)
    
    laporan_str = ""
    
    for i, tanya in enumerate(DAFTAR_PERTANYAAN):
        jw_bef = hasil_before.get(tanya, "Error")
        jw_aft = hasil_after.get(tanya, "Error")
        
        chunk = f"\n[TANYA]: {tanya}\n"
        chunk += f"🔴 BEFORE : {jw_bef}\n"
        chunk += f"🟢 AFTER  : {jw_aft}\n"
        chunk += "-"*60
        
        print(chunk)
        laporan_str += chunk

    # Simpan ke file
    with open("hasil_perbandingan_final.txt", "w") as f:
        f.write(laporan_str)
    print("\nFile 'hasil_perbandingan_final.txt' berhasil disimpan.")


def is_google_colab():
    """Mengecek apakah script berjalan di lingkungan Google Colab."""
    # Cara 1: Cek module google.colab
    if 'google.colab' in sys.modules:
        return True
    # Cara 2: Cek environment variable (kadang berbeda tiap instance)
    if 'COLAB_GPU' in os.environ:
        return True
    return False

def clear_gpu_memory():
    """
    Membersihkan memori.
    """
    import gc

    # LOGIKA IF-ELSE KOLAB
    if is_google_colab():
        print("\n[Colab Detected] Membersihkan Memori GPU & RAM...")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize() 
        print("   ✅ Memori dibersihkan.")
    else:
        print("\n[Local Environment] ⏩ Skip pembersihan memori agresif.")

# ==========================================
# 3. MAIN EXECUTION (ALUR UTAMA)
# ==========================================
if __name__ == "__main__":

    # 1. Load Model & Setup
    model, tokenizer = load_base_model()
    model = apply_lora_adapters(model)

    # 2. Test Before Fine Tuning
    # Kita harus memastikan tokenizer global tersedia untuk fungsi formatting nanti
    globals()['tokenizer'] = tokenizer 
    
    hasil_before = jalankan_pengujian(model, tokenizer, "BEFORE FINE-TUNING")

    clear_gpu_memory()

    # 3. Fine Tuning
    # Kembalikan model ke mode training
    FastLanguageModel.for_training(model)
    model = train_model(model, tokenizer)

    # 4. Test After Fine Tuning
    hasil_after = jalankan_pengujian(model, tokenizer, "AFTER FINE-TUNING")

    # 5. Output Comparison
    cetak_perbandingan(hasil_before, hasil_after)
