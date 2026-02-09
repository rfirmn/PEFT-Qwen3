import subprocess
import os
import sys
import shutil

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"🚀 SEDANG MENJALANKAN: {script_name}")
    print(f"{'='*50}\n")
    try:
        # Menjalankan script python menggunakan interpreter yang sama
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"\nSUKSES: {script_name} selesai dijalankan.")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Gagal menjalankan {script_name}. (Exit code: {e.returncode})")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Terjadi kesalahan tidak terduga pada {script_name}: {e}")
        sys.exit(1)

def main():
    pdf_input = "permenkes-no-10-tahun-2024.pdf"
    if not os.path.exists(pdf_input):
        print(f"ERROR: File '{pdf_input}' tidak ditemukan.")
        print("   Silakan letakkan file PDF tersebut di folder yang sama dengan script ini.")
        return

    run_script("processing.py")

    run_script("augmented.py")

    print("\nMemulai training. Pastikan GPU aktif.")
    run_script("Finetuning QLora.py")

    print(f"\n{'='*50}")
    print("🎉 SELURUH PIPELINE SELESAI!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()