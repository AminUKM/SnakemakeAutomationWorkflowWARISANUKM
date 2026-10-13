import os
import re
import string
import sys
from datetime import datetime
import math

# ----------------------------
# IC detection
# ----------------------------
ID_REGEX = re.compile(r'\b(\d{12}|\d{6}[- ]\d{2}[- ]\d{4})\b')

def _is_valid_yyMMdd(six_digits):
    try:
        datetime.strptime(six_digits, "%y%m%d")
        return True
    except ValueError:
        return False

def replace_ids(text, placeholder="<IC>"):
    def _repl(match):
        raw = match.group(0)
        digits = re.sub(r'\D', '', raw)
        if len(digits) == 12 and _is_valid_yyMMdd(digits[:6]):
            return placeholder
        else:
            return raw
    return ID_REGEX.sub(_repl, text)


# ----------------------------
# Date Normalization
# ----------------------------
def normalize_dates(text: str) -> str:
    def fix_date(match):
        day, month, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        try:
            dt = datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return match.group(0)
    return re.sub(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", fix_date, text)


# ----------------------------
# Malay Normalization Dictionary (unchanged)
# ----------------------------
MALAY_NORMALIZATION = {
    "sy": "saya", "i": "saya", "ko": "kau", "sye": "saya", "saye": "saya",
    "aq": "saya", "aku": "saya", "kite": "kita", "kiteorg": "kita orang",
    "korang": "kamu semua", "korg": "kamu semua", "diorg": "mereka",
    "derang": "mereka", "dorang": "mereka", "drg": "mereka", "pegi": "pergi",
    "g": "pergi", "pi": "pergi", "balek": "balik", "blk": "balik", "mkn": "makan",
    "mknn": "makanan", "minum2": "minum-minum", "tggu": "tunggu", "tgu": "tunggu",
    "tnggu": "tunggu", "nk": "nak", "xnk": "tak nak", "xmo": "tak mahu", "mo": "mahu",
    "dpt": "dapat", "smpai": "sampai", "smpi": "sampai", "bley": "boleh", "blh": "boleh",
    "leh": "boleh", "jd": "jadi", "bwk": "bawa", "ambik": "ambil", "amik": "ambil",
    "tgk": "tengok", "tngk": "tengok", "igt": "ingat", "sgt": "sangat", "truk": "teruk",
    "bz": "sibuk", "bestnyer": "seronok", "cantiknyer": "cantik", "mcm": "macam",
    "cm": "macam", "gile": "gila", "gilerr": "gila", "brani": "berani", "seronoknyer": "seronok",
    "yg": "yang", "utk": "untuk", "untok": "untuk", "dgn": "dengan", "ngn": "dengan",
    "gn": "dengan", "ng": "dengan", "kat": "dekat", "kt": "dekat", "tp": "tapi",
    "dlm": "dalam", "plk": "pula", "lak": "pula", "ade": "ada", "xda": "tiada",
    "xde": "tiada", "x": "tak", "tk": "tak", "takde": "tiada", "ape": "apa", "aper": "apa",
    "brp": "berapa", "klu": "kalau", "tq": "terima kasih", "thx": "terima kasih",
    "pls": "tolong", "okey": "okay", "oklah": "okay lah", "haha": "ketawa",
    "hehe": "ketawa kecil", "lol": "ketawa", "alhamdulilah": "alhamdulillah",
    "jgn": "jangan", "sja": "sahaja"
}

def normalize_malay(text: str) -> str:
    def repl(match):
        word = match.group(0)
        return MALAY_NORMALIZATION.get(word.lower(), word)
    text = re.sub(r"\b\w+\b", repl, text)
    text = re.sub(r"\b([a-zA-Z]+)2\b", lambda m: f"{m.group(1)}-{m.group(1)}", text)
    return text


# ----------------------------
# Punctuation Normalization (unchanged)
# ----------------------------
def normalize_repeated_punctuations(text: str) -> str:
    text = text.replace("\n", "<NEWLINE>")
    text = re.sub(r"(\.\s*){2,}", ".", text)
    text = re.sub(r"(\?\s*){2,}", "?", text)
    text = re.sub(r"(!\s*){2,}", "!", text)
    text = re.sub(r"(,\s*){2,}", ",", text)
    text = re.sub(r"(;\s*){2,}", ";", text)
    text = re.sub(r"(:\s*){2,}", ":", text)
    text = re.sub(r"([!?])(\s*[!?]){1,}", r"\1", text)
    text = re.sub(r"\s*([.,!?;:])\s*", r"\1 ", text)
    text = text.replace("<NEWLINE>", "\n")
    text = re.sub(r"[ \t]+(\n|$)", r"\1", text)
    return text.strip()


# ----------------------------
# Load MCP Malay Wordlist (unchanged)
# ----------------------------
def load_mcp_wordlist(path=r"C:\WARISAN_TEAMDATA\snake file\3.0 normalize\kamus_dewan.txt"):
    if not os.path.exists(path):
        print(f"[WARN] MCP wordlist not found at {path}")
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

MCP_WORDS = load_mcp_wordlist()


# ----------------------------
# Filter Non-Standard Sentences (unchanged)
# ----------------------------
def filter_non_standard_sentences(text, threshold=0.3, sample_size=10, return_removed=False):
    if not MCP_WORDS:
        print("[WARN] MCP wordlist not loaded, skipping filter.")
        return (text, "") if return_removed else text

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    clean_lines, removed_lines = [], []

    arabic_re = re.compile(r'[\u0600-\u06FF]')

    for line in lines:
        if arabic_re.search(line):
            removed_lines.append((line, "ARABIC"))
            continue

        tokens = re.findall(r"[a-zA-Z']+", line.lower())
        if not tokens:
            continue

        unknown_ratio = sum(1 for t in tokens if t not in MCP_WORDS) / len(tokens)
        if unknown_ratio <= threshold:
            clean_lines.append(line)
        else:
            removed_lines.append((line, f"{round(unknown_ratio, 2)} threshold"))

    if removed_lines:
        print(f"[INFO] Filtered out {len(removed_lines)} non-standard sentences ({len(clean_lines)} kept).")
        print("[INFO] Example removed sentences:")
        for sent, reason in removed_lines[:sample_size]:
            print(f"  - ({reason}): {sent[:120]}")

    clean_text = "\n".join(clean_lines)
    removed_text = "\n".join(sent for sent, _ in removed_lines)

    return (clean_text, removed_text) if return_removed else clean_text


# ----------------------------
# Double Cleaning Step (fixed & safe)
#   - DOES NOT remove valid numbers like "ke-7", "ke 7", "RM10", "tahun 2024"
#   - removes junk punctuation, empty parentheses, stuck punctuation runs
#   - fixes stuck-together capitals like FEEDBACKProgram -> Program
# ----------------------------
def double_clean_text(text: str) -> str:
    # remove empty parentheses
    text = re.sub(r"\(\s*\)", "", text)

    # change ", ." or ", ." style to "."
    text = re.sub(r",\s*\.", ".", text)

    # collapse repeated punctuation (e.g. "??!!" -> "?" ; "...." -> ".")
    text = re.sub(r"[!?]{2,}", "?", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"[,:;]{2,}", ",", text)

    # remove isolated punctuation-only tokens on their own lines/positions
    # (will be trimmed later)
    text = re.sub(r"(?m)^[\s\-\–\—\.,;:()]+$", "", text)

    # fix stuck-together capitalized words (keep acronyms if fully uppercase)
    # FEEDBACKProgram -> Program ; but keep "NASA" as NASA
    text = re.sub(r"\b([A-Z]{2,})([A-Z][a-z][a-zA-Z]*)\b", r"\2", text)

    # ensure reasonable spacing
    text = re.sub(r"\s+", " ", text).strip()

    # ensure punctuation followed by no-space+letter gets a space: "kata.Seterusnya" -> "kata. Seterusnya"
    # but don't insert space inside decimal numbers like "1.5" (we avoid numbers with digits on both sides)
    text = re.sub(r"(?P<punc>[.!?])(?P<next>(?=[A-Za-zÀ-ÖØ-öø-ÿ]))", r"\g<punc> \g<next>", text)

    return text.strip()


# ----------------------------
# Sentence splitting - bulletproof:
#    - ensures 1 sentence = 1 line
#    - handles ".Perdana" (no space) and ". Perdana"
#    - preserves numbers and abbreviations as much as possible
# ----------------------------
def split_sentences_to_lines(text: str) -> str:
    # Normalize some ellipsis to single period to avoid many short fragments:
    text = re.sub(r"\.{3,}", ".", text)

    # First, ensure punctuation followed directly by letter gets a space (handled in double_clean_text too)
    text = re.sub(r"([.!?])(?=[A-Za-zÀ-ÖØ-öø-ÿ])", r"\1 ", text)

    # Then split: put newline after sentence-ending punctuation if followed by whitespace or end-of-string
    # We treat ., ?, ! as sentence terminators.
    # Keep number decimals like "1.5" => we avoid splitting inside numbers using a negative lookbehind for digit+dot+digit.
    # Approach: split on punctuation that is not a decimal point (i.e., not between digits)
    # Replace "SENT. Next" -> "SENT.\nNext"
    def _replace_fn(m):
        punct = m.group(1)
        return punct + "\n"

    # Use regex that matches punctuation that is not between digits (so "1.5" won't split)
    text = re.sub(r"(?<!\d)([.!?])(?!\d)", _replace_fn, text)

    # Trim spaces around newlines and collapse multiple newlines
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln != ""]
    return "\n".join(lines)


# ----------------------------
# Final Anonymizer (keeps your original pipeline but with fixes)
# ----------------------------
def anonymize(text):
    # Step 1: Basic normalization
    text = replace_ids(text)
    text = normalize_dates(text)
    text = normalize_malay(text)
    text = normalize_repeated_punctuations(text)

    # Step 2: initial double cleaning (safe)
    text = double_clean_text(text)

    # Step 3: sentence splitting -> 1 sentence per line
    text = split_sentences_to_lines(text)

    # Step 4: final trim + preserve numbers (we do NOT perform aggressive numeric deletion anymore)
    # (previous aggressive numeric removals removed valid constructs like "ke-7" — so we avoid that)
    text = text.strip()

    # Step 5: Filter non-standard sentences using MCP_WORDS (if available)
    clean_text, removed_text = filter_non_standard_sentences(text, return_removed=True)
    return clean_text, removed_text


# ----------------------------
# CLI entry point for Snakemake
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python NORMALIZATION_FUNCTION_CODE.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    clean_text, filtered_text = anonymize(text)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(clean_text)

    base, ext = os.path.splitext(output_file)
    filtered_path = f"{base}_filtered{ext}"
    with open(filtered_path, "w", encoding="utf-8") as f:
        f.write(filtered_text)

    print(f"[NORMALIZE] {input_file} → {output_file}")
    print(f"[FILTERED] Saved removed sentences → {filtered_path}")
