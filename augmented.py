import json
import random

# Konfigurasi Input/Output
INPUT_FILE = "dataset_finetuning.jsonl"  
OUTPUT_FILE = "dataset_final.jsonl"     

TEMPLATES = {
    "definisi": [
        "Jelaskan definisi '{term}'.",
        "Apa yang dimaksud dengan '{term}'?",
        "Berikan pengertian dari '{term}' menurut aturan ini.",
        "Dalam konteks regulasi ini, apa arti '{term}'?",
        "Definisikan istilah '{term}'."
    ],
    "tujuan": [
        "Apa tujuan utama dari peraturan ini?",
        "Untuk alasan apa regulasi ini dibuat?",
        "Jelaskan gol/tujuan dibentuknya aturan ini.",
        "Mengapa JDIH Kemenkes perlu dibentuk?"
    ],
    "tugas": [
        "Sebutkan tugas dan tanggung jawab unit ini.",
        "Apa saja kewajiban yang harus dilakukan?",
        "Uraikan fungsi dan peran dari entitas tersebut.",
        "Jelaskan job desk yang diatur dalam pasal ini."
    ],
    "umum": [
        "Jelaskan poin utama dari bagian ini.",
        "Apa inti sari dari pasal berikut?",
        "Ringkaskan isi ketentuan ini.",
        "Informasi apa yang terkandung dalam teks ini?"
    ]
}

def detect_type(instruction):
    instruction = instruction.lower()
    if "definisi" in instruction: return "definisi"
    if "tujuan" in instruction or "bertujuan" in instruction: return "tujuan"
    if "tugas" in instruction or "fungsi" in instruction: return "tugas"
    return "umum"

def extract_term(instruction):
    # Coba ambil teks di dalam tanda kutip tunggal '...'
    import re
    match = re.search(r"'(.*?)'", instruction)
    if match:
        return match.group(1)
    return None

def augment_process():
    augmented_data = []
    
    print(f"Membaca {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            original_entry = json.loads(line)
            original_instr = original_entry['instruction']
            
            # 1. Masukkan data ASLI (Penting agar data original tetap ada)
            augmented_data.append(original_entry)
            
            # 2. Deteksi tipe pertanyaan
            q_type = detect_type(original_instr)
            
            # 3. Ambil template yang cocok
            templates = TEMPLATES.get(q_type, TEMPLATES["umum"])
            
            for temp in templates:
                new_instr = temp
                
                # Khusus Definisi: Ganti {term} dengan istilah aslinya
                if q_type == "definisi":
                    term = extract_term(original_instr)
                    if term:
                        new_instr = temp.replace("{term}", term)
                    else:
                        continue # Skip jika gagal ekstrak term
                
                # Cek agar tidak duplikat dengan aslinya
                if new_instr != original_instr:
                    new_entry = {
                        "instruction": new_instr,       
                        "input": original_entry['input'], 
                        "output": original_entry['output']
                    }
                    augmented_data.append(new_entry)

    # Simpan Hasil
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Selesai! Data bertambah dari {sum(1 for _ in open(INPUT_FILE))} menjadi {len(augmented_data)} baris.")
    print(f"File disimpan di: {OUTPUT_FILE}")

if __name__ == "__main__":
    augment_process()