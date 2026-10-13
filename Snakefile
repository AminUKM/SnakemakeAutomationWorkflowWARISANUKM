import sys, os, importlib.util

# --- Import cleaning function ---
clean_code_path = r"D:\WarisanTeamData\snake file\2.0 clean\CLEANING_FUNCTION_CODE.py"
spec = importlib.util.spec_from_file_location("CLEANING_FUNCTION_CODE", clean_code_path)
CLEANING_FUNCTION_CODE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLEANING_FUNCTION_CODE)
clean_text = CLEANING_FUNCTION_CODE.clean_text

RAW_DIR = r"D:\WarisanTeamData\snake file\0.0 raw file"

RAW_DIR   = r"D:\WarisanTeamData\snake file\0.0 raw file"
CLEAN_DIR = r"D:\WarisanTeamData\snake file\2.0 clean\clean txt"
DEDUP_DIR = r"D:\WarisanTeamData\snake file\5.0 Deduplication\removed_txt"

# --- Auto-detect the latest stage ---
pdf_files   = [f for f in os.listdir(RAW_DIR)   if f.lower().endswith(".pdf")]
txt_files   = [f for f in os.listdir(RAW_DIR)   if f.lower().endswith(".txt")]
clean_files = [f for f in os.listdir(CLEAN_DIR) if f.lower().endswith("_cleaned.txt")] if os.path.exists(CLEAN_DIR) else []
dedup_files = [f for f in os.listdir(DEDUP_DIR) if f.lower().endswith("_deduplicated.txt")] if os.path.exists(DEDUP_DIR) else []

if pdf_files or txt_files:
    all_files = pdf_files + txt_files
    basenames = [os.path.splitext(f)[0] for f in all_files]
    print(f"[INFO] Starting from raw files: {len(basenames)} found.")
elif clean_files:
    basenames = [f.replace("_cleaned.txt", "") for f in clean_files]
    print(f"[INFO] Starting from cleaned files: {len(basenames)} found.")
elif dedup_files:
    basenames = [f.replace("_deduplicated.txt", "") for f in dedup_files]
    print(f"[INFO] Starting from deduplicated files: {len(basenames)} found.")
else:
    basenames = []
    print("[INFO] No files found in any stage. Nothing to run.")

rule all:
    input:
        expand("3.0 normalize/normalized_txt/{name}_normalized.txt", name=basenames)

rule extract_text:
    input:
        lambda wildcards: [
            os.path.join(RAW_DIR, f"{wildcards.name}.pdf")
            if os.path.exists(os.path.join(RAW_DIR, f"{wildcards.name}.pdf"))
            else os.path.join(RAW_DIR, f"{wildcards.name}.txt")
        ][0]
    output:
        "1.0 extraction/extracted txt/{name}.txt"
    shell:
        "python \"1.0 extraction/PDF_TO_TXT.py\" \"{input}\" \"{output}\""

rule clean_data:
    input:
        "1.0 extraction/extracted txt/{name}.txt"
    output:
        "2.0 clean/clean txt/{name}_cleaned.txt"
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        print(f"[CLEAN] {input[0]} → {output[0]}", flush=True)
        with open(input[0], "r", encoding="utf-8") as f:
            raw_text = f.read()
        cleaned_text = clean_text(
            raw_text,
            remove_citations=True,
            min_words=5,
            english_threshold=0.2,
            uppercase_threshold=0.8,
            min_upper_run=3
        )
        with open(output[0], "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"[CLEAN] Saved {output[0]}", flush=True)

rule deduplicate_data:
    input:
        "2.0 clean/clean txt/{name}_cleaned.txt"
    output:
        "5.0 Deduplication/removed_txt/{name}_deduplicated.txt"
    shell:
        "python \"5.0 Deduplication/Deduplication.py\" \"{input}\" \"{output}\""

rule normalize_data:
    input:
        "5.0 Deduplication/removed_txt/{name}_deduplicated.txt"
    output:
        "3.0 normalize/normalized_txt/{name}_normalized.txt"
    shell:
        "python \"3.0 normalize/normalization.py\" \"{input}\" \"{output}\""






## 2 options to run snakemake:
## python -m snakemake --cores 16 --scheduler greedy (continue from last file)
## python -m snakemake --cores 16 --scheduler greedy --forceall(kalau nak mula dari awal)
## python -m snakemake --unlock
# Add this if outputs take time to appear:

##python -m snakemake --cores 20 --scheduler greedy --forceall 
#   --latency-wait 60
## Instructions to run snakemake in command prompt:
## 1. Copy raw file into C:\WARISAN_TEAMDATA\snake file\0.0 raw file.
## 2. Run the snakemake command above at the terminal.
## 3. The output files will be saved in their respective folders automatically.
## 4. Check all txt files in 5.0 Deduplication/removed_txt folder is correct.
## 5. Repeat steps 1-4 for new raw files.
## 6. Once the raw data have completed make a new folder in D:\WarisanTeamData\snake file\ called "6.0 Final Data" and transfer all deduplicated files there , please make a folder by respective domain.
## 7. Delete all files in 1.0 extraction, 2.0 clean and 5.0 Deduplication folders to save space once the final data have been transferred.


