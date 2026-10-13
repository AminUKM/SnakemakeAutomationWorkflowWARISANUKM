import fitz  # PyMuPDF
import re
import os
import sys

# --- REMOVE WATERMARK FUNCTION ---
def remove_watermark(doc):
    """
    Remove all "Artifact" and "Watermark" objects from the PDF, as well as
    any streams that contain them.
    """
    print("Removing watermarks from PDF...")
    for pnum, page in enumerate(doc, start=1):
        xref_list = page.get_contents()
        if not xref_list:
            print(f"Page {pnum}: no content streams, skipping")
            continue

        for xref in xref_list:
            try:
                raw = bytearray(doc.xref_stream(xref))
            except Exception as e:
                print(f"Page {pnum}, xref {xref}: not a stream or unreadable ({e}), skipping")
                continue
            out = bytearray()
            i = 0
            depth = 0
            while i < len(raw):
                if depth == 0 and raw[i:i+3] == b"BDC":
                    j = i + 3
                    while j < len(raw) and raw[j] in b" \t\r\n":
                        j += 1
                    if raw[j:j+9] == b"/Artifact":
                        depth = 1
                        i = j + 9
                        continue
                elif depth > 0:
                    if raw[i:i+3] == b"BDC":
                        depth += 1
                        i += 3
                        continue
                    if raw[i:i+3] == b"EMC":
                        depth -= 1
                        i += 3
                        continue
                    i += 1
                    continue

                out.append(raw[i])
                i += 1
            if out == raw:
                continue

            try:
                doc.update_stream(xref, bytes(out))
                print(f"Page {pnum}, xref {xref}: stream updated")
            except Exception as e:
                print(f"Page {pnum}, xref {xref}: failed to update ({e})")

    for xref in range(1, doc.xref_length()):
        try:
            obj = doc.xref_object(xref)
        except Exception:
            continue

        if "/Subtype" in obj and "/Form" in obj and "/Watermark" in obj:
            doc.update_object(xref, "<<>>")


def detect_exact_header_first_lines(doc, header, start=0, end=None):
    if end is None:
        end = doc.page_count
    for i in range(start, end):
        text = doc.load_page(i).get_text("text")
        for line in text.splitlines():
            if line.strip().upper().startswith(header.upper()):
                return i
    return None


def detect_exact_header_bottom_up(doc, header, start=None, end=0):
    if start is None:
        start = doc.page_count
    for i in range(start - 1, end - 1, -1):
        text = doc.load_page(i).get_text("text")
        for line in text.splitlines():
            if line.strip().upper().startswith(header.upper()):
                return i
    return None


# --- TRIM PDF PAGES FUNCTION ---
def trim_pdf_pages(doc):
    # Accept both "BAB I" and "BAB 1"
    bab_i_page = (
        detect_exact_header_first_lines(doc, "BAB I") or
        detect_exact_header_first_lines(doc, "BAB 1")
    )
    if bab_i_page is None:
        print("⚠️ Could not locate 'BAB I' or 'BAB 1', keeping all pages")
        return doc

    # Accept "RUJUKAN" as end page
    rujukan_page = detect_exact_header_bottom_up(doc, "RUJUKAN", start=doc.page_count, end=bab_i_page)
    if rujukan_page is None:
        print("⚠️ Could not locate 'RUJUKAN', keeping until last page")
        rujukan_page = doc.page_count

    # Keep ABSTRAK + PENGHARGAAN before BAB I
    keep_pages = []
    for header in ["ABSTRAK", "PENGHARGAAN"]:
        for i in range(bab_i_page):
            if detect_exact_header_first_lines(doc, header, start=i, end=i+1):
                keep_pages.append(i)
                break

    # Add BAB I → RUJUKAN range
    keep_pages.extend(range(bab_i_page, rujukan_page))
    keep_pages = sorted(set(keep_pages))

    new_doc = fitz.open()
    for pno in keep_pages:
        new_doc.insert_pdf(doc, from_page=pno, to_page=pno)
    return new_doc


# --- TEXT CLEANING HELPERS ---
no_break_patterns = [
    # Religious
    r'S\.W\.T', r'A\.S\.', r'r\.a\.', r'r\.h\.', r'h\.a\.', r'Ustaz', r'Ustazah', r'Imam', r'Sheikh',

    # Academic / Professional
    r'Prof\.', r'Assoc\. Prof\.', r'Asst\. Prof\.', r'Dr\.', r'Ph\.D\.', r'M\.D\.', 
    r'M\.Phil\.', r'M\.Sc\.', r'B\.Sc\.', r'LL\.B\.', r'LL\.M\.', r'MBA', r'CPA', r'Dr.\.'

    # Malay / Honorific titles
    r'Tun', r'Tan Sri', r'Tan Sri Dato\'?', r'Tan Sri Datuk', 
    r'Dato\'?', r'Dato\' Seri', r'Datuk', r'Datuk Seri', r'Datin', r'Datuk Seri Utama',
    r'Tunku', r'Raja', r'Ratu', r'Megat', r'Puteri', r'Che', r'Wan', r'Mohd\.', 
    r'Syed', r'Syarifah', r'Sharifah', r'Md\.', r'Pn\.', r'En\.', r'Hj\.',
    r'Haji', r'Hajah', r'Tuan Haji', r'Puan Hajah',

    # Civil / honorifics
    r'Encik', r'Cik', r'Puan', r'Tuan', r'Yang Berhormat', r'YB', r'Yang Amat Berhormat', r'YAB',
    r'Yang Berbahagia', r'YBhg\.', r'Yang Amat Mulia', r'YAM', r'Yang Mulia', r'YM',

    # Common Latin / citations
    r'et al\.', r'et\. al\.', r'et\.al\.', r'cf\.', r'vs\.', r'etc\.', r'ibid\.', r'op\. cit\.',

    # Common abbreviations
    r'pp\.', r'vol\.', r'no\.', r'fig\.', r'ed\.', r'Inc\.', r'Ltd\.', r'Co\.', r'Corp\.', 
    r'Sdn\. Bhd\.', r'Bhd\.', r'M\.',

    # Other honoraries
    r'Duli Yang Maha Mulia', r'DYMM', r'Duli Yang Teramat Mulia', r'DYTM',
    r'His Excellency', r'Her Excellency', r'H\.E\.', r'Sir', r'Dame',
    r'Tan Sri Panglima', r'Datuk Panglima', r'Dato\' Paduka', r'Datuk Paduka',
    r'Yang Arif', r'YA', r'Yang Hormat', r'YH',
]

bullet_points = [
    '•', '▪', '‣', '⁃', '○', '●', '■', '□', '◆', '▶', '➢', '⦿',
    '◯', '⭘', '〇', '◦', '⦾', '⦿', '',
    '- ', '* ', '+ '
]
remove_patterns = [
    r'^[A-Z\s]{3,}$',          # all uppercase headings
    r'^[ivxlcdm]+$',           # lowercase Roman numerals
    r'^[IVXLCDM]+$',           # uppercase Roman numerals
    r'^BAB\s+\d+.*$',          # BAB + number
    r'^\d+$',                  # numbers only
    r'^Jadual\s+\d+.*$',       # Jadual tables
    r'^\s*Bil\s*[\|\s].*$',    # Bil headers
    r'^\|.*\|$',               # table rows
]

remove_re = re.compile('|'.join(remove_patterns))
bullet_re = re.compile('|'.join([re.escape(bp) for bp in bullet_points]))
number_only_re = re.compile(r'^\s*\d+\s*$')


def process_text(raw_text):
    """Shared cleaning pipeline for both PDF pages and TXT files"""
    raw_lines = raw_text.splitlines()
    clean_lines = []
    seen_lines = set()

    j = 0
    while j < len(raw_lines):
        raw = raw_lines[j]
        line = raw.strip()

        if not line or remove_re.fullmatch(line):
            j += 1
            continue

        line = bullet_re.sub(' ', line)
        line = re.sub(r'\s+', ' ', line).strip()

        if line in seen_lines:
            j += 1
            continue
        seen_lines.add(line)

        if number_only_re.fullmatch(line):
            k = j + 1
            while k < len(raw_lines):
                nxt = raw_lines[k].strip()
                if not nxt or remove_re.fullmatch(nxt):
                    k += 1
                    continue
                nxt = bullet_re.sub(' ', nxt)
                nxt = re.sub(r'\s+', ' ', nxt).strip()
                combined = line + ' ' + nxt
                if combined not in seen_lines:
                    seen_lines.add(combined)
                    clean_lines.append(combined)
                break
            j = k + 1 if k < len(raw_lines) else j + 1
            continue

        clean_lines.append(line)
        j += 1

    page_content = ' '.join(clean_lines)
    page_content = re.sub(r'\s+', ' ', page_content).strip()

    # Protect abbreviations
    placeholder_map = {}
    for idx, pat in enumerate(no_break_patterns):
        matches = re.findall(pat, page_content, flags=re.IGNORECASE)
        for match in matches:
            placeholder = f"__ABBR_{idx}_{len(placeholder_map)}__"
            placeholder_map[placeholder] = match
            page_content = page_content.replace(match, placeholder)

    # Sentence splitting
    sentences = re.split(r'(?<=[.])\s+(?=[A-Z0-9])', page_content)

    restored_sentences = []
    for s in sentences:
        for placeholder, original in placeholder_map.items():
            s = s.replace(placeholder, original)
        restored_sentences.append(s.strip())

    return restored_sentences


def merge_sentences(pages_text):
    """Merge across pages, deduplicate, and cleanup"""
    merged_sentences = []
    for page_sents in pages_text:
        if merged_sentences and not merged_sentences[-1].endswith('.'):
            while page_sents:
                merged_sentences[-1] += ' ' + page_sents.pop(0)
                if merged_sentences[-1].endswith('.'):
                    break
        merged_sentences.extend(page_sents)

    # Deduplicate
    seen_sentences = set()
    final_lines = []
    for line in merged_sentences:
        stripped = line.strip()
        if stripped and stripped not in seen_sentences:
            seen_sentences.add(stripped)
            final_lines.append(stripped)

    # Cleanup
    extracted_text = "\n".join(final_lines)
    extracted_text = re.sub(r'[\u25CF-\u25EF\u25A0-\u25FF\u2022-\u2023]', '', extracted_text)
    extracted_text = re.sub(r'(?<!\w)-(?!\w)', ' ', extracted_text)
    extracted_text = re.sub(r' +', ' ', extracted_text)
    extracted_text = re.sub(r'\n{3,}', '\n\n', extracted_text.strip())
    return extracted_text


# --- MAIN SCRIPT ---
if len(sys.argv) != 3:
    print("Usage: python PDF_TO_TXT.py input.pdf|input.txt output.txt")
    sys.exit(1)

input_file = sys.argv[1]
output_txt = sys.argv[2]
ext = os.path.splitext(input_file)[1].lower()

print(f"Processing {os.path.basename(input_file)}...")

if ext == ".pdf":
    doc = fitz.open(input_file)
    remove_watermark(doc)
    doc = trim_pdf_pages(doc)

    pages_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
        sentences = process_text(page_text)
        pages_text.append(sentences)

    doc.close()
    extracted_text = merge_sentences(pages_text)

elif ext == ".txt":
    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()
    sentences = process_text(raw_text)
    extracted_text = merge_sentences([sentences])

else:
    print("❌ Unsupported file type. Only .pdf or .txt allowed.")
    sys.exit(1)

# --- Save TXT ---
os.makedirs(os.path.dirname(output_txt), exist_ok=True)
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(extracted_text)

print(f"✅ Saved: {output_txt}")
