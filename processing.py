import re
import json
import pymupdf4llm

def clean_and_unwrap_text_v3(text):
    # 1. Basic Cleaning
    header_pattern = r"^[\s\S]*?MENTERI KESEHATAN REPUBLIK INDONESIA,"
    text = re.sub(header_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*-\s*\d+\s*-?\s*$", "", text, flags=re.MULTILINE) # Hapus no halaman
    text = re.sub(r"Agar setiap orang mengetahuinya.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"[œдѽж\\]", "", text)

    lines = text.splitlines()
    cleaned_lines = []
    buffer = ""
    
    # KATA KUNCI PENYAMBUNG (GLUE WORDS)
    # Jika baris buffer berakhir dengan kata ini, baris berikutnya PASTI sambungan
    glue_words = ('dalam', 'pada', 'tentang', 'melalui', 'kepada', 'dan', 'atau', 'serta', 'terhadap', 'oleh', 'untuk')

    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Cek apakah buffer sebelumnya menggantung (berakhir dengan glue words)
        buffer_ends_with_glue = buffer.strip().lower().endswith(glue_words)
        
        # Deteksi Header (Pasal X, Menimbang, dll)
        is_potential_header = re.match(r'^(Pasal \d+|Menimbang|Mengingat|MEMUTUSKAN)', line, re.IGNORECASE)
        
        # LOGIKA UTAMA:
        # Jika buffer berakhir dengan glue words (misal: "...dimaksud dalam"), 
        # MAKA abaikan header detection, ini pasti sambungan kalimat.
        if buffer_ends_with_glue:
            buffer += " " + line
        
        # Jika tidak menggantung, dan ini Header, maka simpan buffer lama dan mulai baru
        elif is_potential_header:
            if buffer: cleaned_lines.append(buffer)
            buffer = line
            
        # Deteksi List item (a., b., 1., 2.) -> biasanya baris baru
        elif re.match(r'^(\d+\.|[a-z]\.)\s', line):
            if buffer: cleaned_lines.append(buffer)
            buffer = line
            
        else:
            # Jika bukan header dan bukan list, cek apakah harus digabung
            # Gabung jika buffer tidak diakhiri tanda baca penutup
            if buffer and not buffer.endswith(('.', ';', ':')):
                buffer += " " + line
            else:
                if buffer: cleaned_lines.append(buffer)
                buffer = line
    
    if buffer: cleaned_lines.append(buffer)
    
    return "\n".join(cleaned_lines)

def process_definitions(chunk):
    """Memecah Pasal 1 menjadi instruksi spesifik (Sudah Bagus)"""
    definitions = re.split(r'\n(?=\d+\.\s)', chunk)
    entries = []
    for definition in definitions:
        match = re.search(r'\d+\.\s(.*?)\s(?:adalah|disebut)', definition)
        if match:
            term = match.group(1).strip()
            entries.append({
                "instruction": f"Jelaskan definisi '{term}' berdasarkan peraturan ini.",
                "input": "Konteks: Definisi Umum Peraturan Menteri Kesehatan",
                "output": definition.strip()
            })
    return entries

def generate_final_dataset(input_pdf, output_jsonl):
    # 1. Extract & Clean
    raw_md = pymupdf4llm.to_markdown(input_pdf)
    clean_text = clean_and_unwrap_text_v3(raw_md) # Pakai versi v3
    
    # Simpan untuk pengecekan manual jika perlu
    with open("output_final_fixed.md", "w", encoding="utf-8") as f:
        f.write(clean_text)

    # 2. Split Chunks
    # Split saat ketemu Pasal baru atau Header utama
    chunks = re.split(r'\n(?=Pasal \d+|Menimbang|Mengingat|MEMUTUSKAN)', clean_text)
    
    dataset = []
    
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 20: continue

        low_chunk = chunk.lower()
        
        # HANDLING DEFINISI (PASAL 1)
        if "dimaksud dengan:" in low_chunk and "pasal 1" in low_chunk:
            dataset.extend(process_definitions(chunk))
            continue

        # HANDLING UMUM
        instruction = "Jelaskan poin utama dari bagian ini."
        if "menimbang" in low_chunk:
            instruction = "Apa pertimbangan sosiologis dan yuridis peraturan ini?"
        elif "mengingat" in low_chunk:
            instruction = "Sebutkan dasar hukum landasan peraturan ini."
        elif "bertujuan" in low_chunk:
            instruction = "Apa tujuan utama dibentuknya JDIH Kemenkes?"
        elif "tugas" in low_chunk:
            instruction = "Uraikan tugas dan tanggung jawab unit yang disebutkan."
        elif "meliputi" in low_chunk:
            instruction = "Apa saja ruang lingkup yang diatur dalam pasal ini?"
        
        # Ambil baris pertama sebagai input context (biasanya "Pasal X...")
        context_line = chunk.split('\n')[0]
        
        dataset.append({
            "instruction": instruction,
            "input": f"Konteks: {context_line}",
            "output": chunk
        })

    # 3. Save
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Dataset FINAL disimpan ke {output_jsonl}")

# Ganti path file di sini
input_pdf = "permenkes-no-10-tahun-2024.pdf"
generate_final_dataset(input_pdf, "dataset_finetuning.jsonl")