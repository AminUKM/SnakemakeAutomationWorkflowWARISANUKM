import os
import re
import math
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from functools import lru_cache

# --- helpers for uppercase handling ---
CITATION_PARENS_RE = re.compile(
    r'\s*\((?=[^)]*(?:19|20)\d{2}[a-z]?)[^)]*\)',
    re.IGNORECASE
)
ROMAN_RE = re.compile(r'^[MDCLXVI]+$', re.IGNORECASE)
NUMDOT_RE = re.compile(r'^\d+(?:\.\d+)*$')
TOC_SECTION_NUM_RE = re.compile(r'\b\d+(?:\.\d+)*\b')
ENUM_START_RE = re.compile(
    r'^\s*(?:[ivxlcdm]{1,4}|[A-Za-z]|\d{1,2})[.)]\s+',
    re.IGNORECASE
)
ENUM_ONLY_RE = re.compile(r'^\s*(?:[ivxlcdm]{1,6}|[A-Za-z]|\d{1,2})[.)]\s*$', re.IGNORECASE)
ENUM_MARKER_RE_1 = re.compile(
    r'(?<!\w)(?P<open>[\(\[])?(?P<mark>(?:[ivxlcdm]{1,8}|[A-Za-z]|\d{1,2}))\s*\)\s*\.(?=\s|$)',
    re.IGNORECASE
)
ENUM_MARKER_RE_2 = re.compile(
    r'(?<!\w)(?P<open>[\(\[])?(?P<mark>(?:[ivxlcdm]{1,8}|[A-Za-z]|\d{1,2}))\s*\.\s*\)(?=\s|$)',
    re.IGNORECASE
)
EOS_RE = re.compile(r'[.!?…]["\')\]]*\s*$')
ENUM_TAIL_RE = re.compile(
    r'(?:^|[\s(])(?:[ivxlcdm]{1,8}|[a-z]|\d{1,2})(?:\)|\.)\.?\s*$',
    re.IGNORECASE
)
CAPTION_START_RE = re.compile(r'^\s*(rajah|jadual)\b', re.IGNORECASE)
NUMERIC_PARENS_RE = re.compile(r'\(\s*\d+(?:\.\d+)*\s*\)')
GREEK_BLOCK = r'\u0370-\u03FF\u1F00-\u1FFF'
BAD_SNIPPET_RE = re.compile(
    r"Kekuatan\s+alah.*?Simulasi\s+Monte\s+Carlo\s*",
    re.IGNORECASE | re.DOTALL
)

# Broad RTM noise blacklist
NOISE_TOKENS = {
    'SOUND', 'UPSOUND', 'SUPER', 'FLASH', 'WIPE', 'MUSIC', 'MUSIK', 'MASUK',
    'INTRO', 'OUTRO', 'SOT', 'VO', 'V O', 'PKG', 'BROLL', 'B-ROLL', 'GFX',
    'LOGO', 'QIN', 'QOUT', 'Q', 'TCOUT', 'READER', 'NEXT', 'CLOSING', 'TEKS',
    'BP', 'KANTA', 'KANTA744'
}

# Common instruction/note phrases to drop
NOTES_BLACKLIST = [
    r'jangan\s+skip', r'jgn\s+skip', r"don't\s+skip", r"dont\s+skip",
    r'jangan\s+lupa', r'jangan\s+terlepas', r'please\s+skip', r'please\s+dont\s+skip',
    r"don't\s+forget", r'dont\s+forget', r'ingat\s+untuk', r'ingat\s+jangan'
]

TITLEISH_OR_GREEK_RE = re.compile(
    rf'\b(?:[A-Z][a-z]{{2,}}|[A-Z]{{2,}}|[{GREEK_BLOCK}]+)\b'
)
SECTION_DOTNUM_RE = re.compile(r'\b\d+\.\d+(?:\.\d+)*\b')
GLOSSARY_PREFIX_RE = re.compile(r'^\s*kekuatan\s+alah\b', re.IGNORECASE)

AUTHOR_TOKEN_STRONG = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ''-]+,\s*(?:[A-Z]\.\s*){1,4}"
AUTHORS_LINE_STRONG_RE = re.compile(
    rf"^\s*(?:{AUTHOR_TOKEN_STRONG}\s*(?:,|\s*&\s*|\s+and\s*))+{AUTHOR_TOKEN_STRONG}"
    r"(?:\b|[\s,;:()]\b)",
    re.UNICODE
)
AUTHOR_ITEM = r"[A-ZÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ''-]{1,40}\s*,\s*(?:[A-Z]\s*\.\s*){1,4}"
AUTHORS_LIST_LINE_RE = re.compile(
    rf"^\s*(?:{AUTHOR_ITEM}\s*(?:,|;|\s+(?:and|dan)\s+|&)\s*){{2,}}{AUTHOR_ITEM}\s*\.?\s*$",
    re.IGNORECASE | re.UNICODE
)
DOI_OR_URL_RE = re.compile(r'^\s*(?:doi\s*[:\s]|https?://|www\.)', re.IGNORECASE)
AUTHOR_TOKEN_RE2 = re.compile(AUTHOR_ITEM, re.UNICODE)

# --- Abbreviation protection ---
PROTECTED_ABBREVIATIONS = [
    's.a.w.', 'r.a.', 'r.h.', 'a.s.', 'SWT', 'S.W.T.', 
    'Prof.', 'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Sr.', 'Jr.', 'En.', 'Assoc.', 'Madya.', 'Ts.',
    'etc.', 'e.g.', 'i.e.', 'vs.'
]

def protect_abbreviations(text):
    for abbr in PROTECTED_ABBREVIATIONS:
        safe = abbr.replace('.', '<<DOT>>')
        text = re.sub(re.escape(abbr), safe, text, flags=re.IGNORECASE)
    return text

def unprotect_abbreviations(text):
    return text.replace('<<DOT>>', '.')

# --- Noise detection ---
def is_noisy_sentence(sentence: str, min_words=2) -> bool:
    """Heuristic to decide if a sentence is noise (shortforms, tech cues, speaker markers)."""
    if not sentence or sentence.strip() == '':
        return True

    toks = re.findall(r"\b\w+\b", sentence)
    if not toks:
        return True
    if len(toks) < min_words:
        return True

    for t in toks:
        if t.upper() in NOISE_TOKENS:
            return True

    short_tokens = [w for w in toks if len(w) <= 3]
    mixed_ld = [w for w in toks if re.search(r'[A-Za-z]+\d|\d+[A-Za-z]+', w)]
    if len(short_tokens) / len(toks) >= 0.6:
        return True
    if mixed_ld:
        return True

    s_low = sentence.lower()
    for pat in NOTES_BLACKLIST:
        if re.search(pat, s_low):
            return True

    punct_chars = 0
    for i, ch in enumerate(sentence):
        if ch == '%':
            prev = sentence[i-1] if i > 0 else ''
            nxt = sentence[i+1] if i+1 < len(sentence) else ''
            if (prev and prev.isdigit()) or (nxt and nxt.isdigit()):
                continue
        if not ch.isalnum() and not ch.isspace():
            punct_chars += 1
    if punct_chars / max(1, len(sentence)) > 0.5:
        return True

    return False

def is_upper_like_token(tok: str) -> bool:
    t = re.sub(r'[^\w]', '', tok)
    if not t:
        return False
    if NUMDOT_RE.fullmatch(tok):
        return True
    if ROMAN_RE.fullmatch(t):
        return True
    return t.isalpha() and t.isupper() and len(t) >= 2

def remove_uppercase_runs(sentence, min_upper_run=3):
    pattern = re.compile(
        r'(?:\b(?:[A-Z]{2,}|\d+\.?\d*|[IVXLCDM]+)\b[\s]*){' + str(min_upper_run) + ',}'
    )
    cleaned = pattern.sub('', sentence)
    return re.sub(r'\s+', ' ', cleaned).strip()

# --- Content blob detection ---
def looks_like_content_blob(s: str, min_titleish=12, max_punct=2) -> bool:
    """Long glossary-like list that also contains a section token (e.g., 1.1 …)."""
    if not SECTION_DOTNUM_RE.search(s):
        return False
    titleish = len(TITLEISH_OR_GREEK_RE.findall(s))
    punct = len(re.findall(r'[.,;:!?]', s))
    for m in SECTION_DOTNUM_RE.finditer(s):
        punct -= m.group(0).count('.')
    return titleish >= min_titleish and punct <= max_punct

def extract_enum_tail(s: str):
    m = ENUM_TAIL_RE.search(s)
    if not m:
        return s, None
    body = s[:m.start()].rstrip()
    tail = s[m.start():].strip()
    return body, tail

def canonicalize_enum_tail(tail: str) -> str:
    t = tail.strip()
    t = re.sub(r'^[\s(]+', '', t)
    t = re.sub(r'\)\.?$', '.', t)
    if not t.endswith('.'):
        t += '.'
    return t

CONTENT_WORD_RE = re.compile(r'\b([A-Z][a-z]{2,}|[A-Za-z]{3,})\b', re.UNICODE)
LETTER_RE = r"[A-Za-zÀ-ÖØ-öø-ÿ]"

# --- Reference section detection ---
REFERENCE_HEADING_RE = re.compile(
    r'^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:rujukan|senarai rujukan|bibliografi|bibliography|references|daftar pustaka)\s*[:.]?\s*$',
    re.IGNORECASE
)
AUTHOR_TOKEN_RE = re.compile(
    r'\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ\'\-]+,\s*(?:[A-Z]\.\s*){1,4}',
    re.UNICODE
)
YEAR_RE = re.compile(r'\(\d{4}[a-z]?\)|\b(?:19|20)\d{2}\b')
DOI_URL_RE = re.compile(r'\b(?:doi:|https?://\S+)', re.IGNORECASE)

def split_colon_followed_by_enum(line: str):
    m = re.match(
        r'^(.*?:)\s*((?:[ivxlcdm]{1,6}|[A-Za-z]|\d{1,2})[.)](?:\s+.*)?\s*)$',
        line,
        flags=re.IGNORECASE
    )
    if m:
        return [m.group(1).rstrip(), m.group(2).lstrip()]
    return [line]

def looks_like_reference_start(s: str) -> bool:
    if not s:
        return False
    if REFERENCE_HEADING_RE.search(s):
        return True
    author_tokens = AUTHOR_TOKEN_RE.findall(s)
    if len(author_tokens) >= 2:
        return True
    if author_tokens and (YEAR_RE.search(s) or DOI_URL_RE.search(s)):
        return True
    return False

def starts_like_continuation(s: str) -> bool:
    return re.match(r"^\s*(?:\(?\d{1,3}\)?[.)]?\s+)?[a-zà-öø-ÿ]", s) is not None

def strip_sentence_leading_index(s: str) -> str:
    return re.sub(rf"^\s*(?:\(?\d{{1,3}}\)?[.)]?)\s+(?={LETTER_RE})", "", s)

def remove_inline_ocr_noise(s: str) -> str:
    s = re.sub(r'\b(?:[A-Za-z]\s+)?[A-Za-z]\s*\.\s*(?:\.\s*){2,}', ' ', s)
    s = re.sub(r'\b[A-Za-z]\s*(?:,\s*)?(?:[.,;:!?…·•]\s*){3,}', ' ', s)
    s = re.sub(r'\b[0-9A-Za-z]{1,4}\s*-\s*[A-Za-z]{1,3}\s+\d+\s*-\s*\.(?=\s|$)', ' ', s)
    s = re.sub(r'(?<=\s)(?:[.,;:!?…·•]\s*){3,}(?=\s|$)', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

def strip_scatter_prefix(s: str) -> str:
    m = CONTENT_WORD_RE.search(s)
    return s[m.start():] if m else s

def strip_leading_bullets_enums(s: str) -> str:
    s = re.sub(
        r'^(?:[•·\-\u2022\u2023\u25E6—–]+\s*)?'
        r'(?:(?:\(\s*(?:[ivxlcdm]{1,8}|[A-Za-z]|\d{1,3})\s*\)|'
        r'(?:[ivxlcdm]{1,8}|[A-Za-z]|\d{1,3})[.)])\s*){0,3}',
        '',
        s,
        flags=re.IGNORECASE
    )
    s = re.sub(r'^(?:[A-Z](?:\s+|$)){1,4}', '', s)
    s = re.sub(r'^(?:\d{1,3}(?:\s+|[.)]\s*)?){1,4}', '', s)
    return s.lstrip()

def sanitize_scatter_symbols(s: str):
    s = re.sub(r'[~`^_\\\[\]\{\}<>\|]', ' ', s)
    s = re.sub(r'["\'`]', '', s)
    s = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\/=%]+', ' ', s)
    s = re.sub(r'^(\s*)(\d+)\)', r'\1\2.', s)
    s = re.sub(r'(?<=\d)\.(?=\d)', '<<DOT>>', s)
    s = re.sub(r'(?<=\d),(?=\d)', '<<COMMA>>', s) 
    s = re.sub(r'\s+([.,;:!?])', r'\1', s)
    s = re.sub(r'([.,;:!?])(?=\S)', r'\1 ', s)
    s = s.replace('<<DOT>>', '.').replace('<<COMMA>>', ',')
    s = re.sub(r'(\d)\s*%', r'\1%', s)
    s = re.sub(r'%(?=[A-Za-z0-9])', '% ', s)
    s = re.sub(r'\s*\(\s*', ' (', s)
    s = re.sub(r'\s*\)', ')', s)
    s = re.sub(r'([.!?])(?:\s*\1)+(?=\s|$)', r'\1', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

def looks_like_allcaps_heading(s: str, max_words=8, min_words=2, upper_ratio=0.8) -> bool:
    s2 = re.sub(r'^\s*\d+(?:\.\d+)*\.?\s*', '', s).strip()
    s2 = s2[:-1] if s2.endswith(':') else s2
    tokens = re.findall(r'\b[^\W\d_]+\b', s2, flags=re.UNICODE)
    alpha = [t for t in tokens if t.isalpha()]
    if not (min_words <= len(alpha) <= max_words):
        return False
    ups = [t for t in alpha if t.isupper() and len(t) >= 2]
    return (len(ups) / len(alpha)) >= upper_ratio

def strip_leading_allcaps_heading(s: str, min_tokens=2, max_tokens=5) -> str:
    s2 = s.lstrip()
    spans = list(re.finditer(r'\b[^\W\d_]+\b', s2, flags=re.UNICODE))
    i = 0
    while i < len(spans):
        w = spans[i].group(0)
        if w.isalpha() and w.isupper() and len(w) >= 2:
            i += 1
        else:
            break
    if min_tokens <= i <= max_tokens:
        cut = spans[i-1].end()
        return s2[cut:].lstrip(' :')
    return s

def looks_like_eos(s: str) -> bool:
    return EOS_RE.search(s) is not None

def ends_with_enum_tail(s: str) -> bool:
    return ENUM_TAIL_RE.search(s) is not None

# --- Language detection ---
@lru_cache(maxsize=10000)
def safe_detect(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def is_likely_english(text: str) -> bool:
    """Check if text is likely English using multiple methods."""
    english_patterns = [
        r'\b(the|is|are|was|were|been|being|have|has|had|will|would|should|could|can)\s+\w+',
        r'\b(this|that|these|those)\s+(is|are|was|were)\b',
        r'\bwe\s+(will|would|should|could|can|are|were)\b',
        r'\bhappen\s+to\s+be\b',
        r'\bchoose\s+to\b',
        r'\b(achieve|assure|ensure|commitment|transparency|accountability)\b',
        r'\bpart\s+of\s+(our|the)\s+policy\b',
        r'\blet\s+me\s+assure\b',
        r'\bmove\s+forward\b',
        r'\btrust\s+us\b',
    ]
    
    for pattern in english_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    try:
        lang = safe_detect(text)
        return lang == 'en'
    except:
        return False

# --- RTM leftovers removal (comprehensive) ---
def remove_rtm_leftovers(text: str) -> str:
    """Remove RTM production elements while preserving actual news content."""
    
    # STEP 1: Remove Block/Technical Markers
    text = re.sub(r'%%[^\s\n\r]+', '', text)
    text = re.sub(r'%%B\[\d+\][^\n]*', '', text)
    text = re.sub(r'%%I\[(STD|VO|JINGLE|PKG)\]%%[VE]?', '', text)
    text = re.sub(r'%%E\b', '', text)
    
    # STEP 2: Remove Metadata Section
    text = re.sub(r'BES\s+ISO\s+PERDANA:[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,2}\s+(?:JANUARI|FEBRUARI|MAC|APRIL|MEI|JUN|JULAI|OGOS|SEPTEMBER|OKTOBER|NOVEMBER|DISEMBER)\s+\d{4}', '', text)
    text = re.sub(r'^PENERBIT:[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^EDITOR:[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^SD:[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^ASD:[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^SA:[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^REG\.\s*CO:[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^NOTE:[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;|&amp;|&lt;|&gt;|&quot;|&#39;|&rsquo;|&lsquo;', ' ', text)
    
    # STEP 3: Remove READER labels and Kanta camera identifiers
    text = re.sub(r'READER\s*\d*:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'KANTA\s*[\d\*\-—–_]+(?:[\s\-\*—–_]*[\d\*\-—–_]+)*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*KANTA[^\n]*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # STEP 4: Remove ALL QIN/QOUT/Q IN/Q OUT patterns
    text = re.sub(r'Q\s*IN[\s:]*(?:\(?\d{1,2}:\d{2}:\d{2}:\d{2}\)?[\s:]*)*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'QIN[\s:]*(?:\(?\d{1,2}[:\.\s]*\d{2}[:\.\s]*\d{2}[:\.\s]*\d{2}\)?[\s:]*)*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Q\s*OUT[\s:]*(?:\(?\d{1,2}:\d{2}:\d{2}:\d{2}\)?[\s:]*)*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'QOUT[\s:]*(?:\(?\d{1,2}[:\.\s]*\d{2}[:\.\s]*\d{2}[:\.\s]*\d{2}\)?[\s:]*)*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Q(?:IN|OUT)[\s:\.]*(?:\d{1,2}[\.\s:]*){0,6}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\.…]+\s*QOUT[\s:]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\.…]+\s*QIN[\s:]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'TCOUT[\s:]*(?:\d{1,2}[:\.\s]*){0,4}', '', text, flags=re.IGNORECASE)
    
    # Enhanced Timestamp Removal
    text = re.sub(r'[,\s]*[:：]\s*\d{1,2}\s*[:：]\s*(?:\d{1,2}\s*[:：]\s*)?(?:\d{1,2}\s*[:：]?\s*)?', '', text)
    text = re.sub(r'\d{1,2}\s*[:：]\s*\d{1,2}\s*[:：]\s*\d{1,2}\s*[:：]\s*[\.]*\s*[-\.]*\s*\d{1,2}\s*[:：]\s*\d{1,2}\s*[:：]\s*\d{1,2}\s*[:：]\s*\d{1,2}', '', text)
    text = re.sub(r'\(\s*\d{1,2}\s*[:：]\s*\d{1,2}\s*[:：]\s*\d{1,2}\s*[:：]\s*\d{1,2}\s*\)', '', text)
    text = re.sub(r'\b\d{1,2}\s*[:：]\s*\d{1,2}\s*[:：]\s*\d{1,2}\s*(?:[:：]\s*\d{1,2})?\b', '', text)
    text = re.sub(r'\b\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\s*(?:\.\s*\d{1,2})?\b', '', text)
    text = re.sub(r'(?<!\w)[\-—–]?\s*[+-]?\d{1,3}(?:\s*[:\.]\s*[+-]?\d{1,3}){1,3}\s*[\-—–]?(?!\w)', ' ', text)
    text = re.sub(r'\(\s*,\s*[A-Z]\s*\.\s*\(', '', text)
    text = re.sub(r'\(\s*,\s*[A-Z]\s*\.\s*', '', text)
    text = re.sub(r'SUPER\s*[:：]\s*[,\s]*[:：]\s*\d{1,2}\s*[:：]\s*\d{1,2}\s*[:：]\s*\d{1,2}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'FLASH\s*&\s*WIPE\s+\d{1,2}\s*[:：]\s*\d{1,2}\s*[:：]\s*\d{1,2}\s*[:：]\s*\d{1,2}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'FLASH\s*[&/]\s*WIPE\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+\d{2}\s*[:\.]\s*\d{2}\s*[:\.]\s*\d{2}\s*', ' ', text)
    text = re.sub(r'(?<!\w)\d{1,2}\s*[:：\.]\s*\d{2}\s*[:：\.]\s*\d{2}\s*(?:[:：\.]\s*\d{2})?(?!\w)', '', text)
    text = re.sub(r'\s*[,\.]+\s*[:：]\s*', ' ', text)
    text = re.sub(r'\s*[:：]\s*[,\.]+\s*', ' ', text)
    
    # STEP 5: Remove UPSOUND blocks
    upsound_pattern = r'UPSOUND[:\s]*[-—]?\s*.*?(?=\n\n|\n[A-Z][a-z]|\Z)'
    def remove_upsound_block(match):
        block = match.group(0)
        if is_likely_english(block):
            return ''
        block = re.sub(r'UPSOUND[:\s]*[-—]?', '', block, flags=re.IGNORECASE)
        return block.strip()
    text = re.sub(upsound_pattern, remove_upsound_block, text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'^UPSOUND[:\s]*[-—]?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # STEP 6: Remove Special Sections
    text = re.sub(r'INSERT\s+SOT[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?im)\binsert\b[^\n]{0,60}\bcutaway(?:s)?\b[^\n]{0,40}\bmv\b', '', text)
    text = re.sub(r'(?im)\binsert\b[^\n]{0,60}\bcutaway(?:s)?\b', '', text)
    text = re.sub(r'(?im)\bcutaway(?:s)?\b[^\n]{0,40}\bmv\b', '', text)
    text = re.sub(r'(?im)\b\d+\s*(?:second|sec|s)\b[^\n]{0,30}\bcutaway(?:s)?\b', '', text)
    text = re.sub(r'(?im)\bcutways\b', '', text)
    text = re.sub(r'STANDUPPER:[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'STAND\s+UPPER:[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^SUPER:[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\[\[[^\]]*\]\]', '', text)
    text = re.sub(r'VT\]\]', '', text)
    text = re.sub(r'VD\]\]', '', text)
    text = re.sub(r'\b(UPSOUND|QIN|QOUT|Q IN|Q OUT|READER|SUPER|FLASH|WIPE|NEXT|CLOSING)\b[:\s\-]*', '', text, flags=re.IGNORECASE)
    
    # STEP 6b: Remove CO: lines, religious salutations, K-codes
    text = re.sub(r'(?im)^\s*CO\s*:\s*.*$', '', text)
    text = re.sub(r'(?i)\bBISMILLAH[A-Z0-9_\-]*\b', '', text)
    text = re.sub(r'(?m)^[^\n]*\bK\d+\b[^\n]*\n?', '', text)
    text = re.sub(r'(?m)^[^\n]*\bH-\w+\b[^\n]*\n?', '', text)
    text = re.sub(r'(?im)^.*\bHand\s*Over\b.*$\n?', '', text)
    text = re.sub(r'(?m)^[^\n]*\bB-\w+\b[^\n]*\n?', '', text)
    text = re.sub(r'(?i)\bB\s*[-\s]?\s*ROLL\b', '', text)
    text = re.sub(r'(?im)^.*\bBP\s*CLOSURE\b.*$', '', text)
    text = re.sub(r'(?m)^\s*[-—–]+\s*([A-Z0-9\s]{1,80})\s*[-—–]+\s*$', '', text)
    text = re.sub(r'(?im)\b(INTRO|INTRODUCTION|OUTRO|MUSIC IN|MUSIC OUT|MUSIK|MASUK|MUSIC)\b', '', text)
    
    # STEP 6c: Remove graphics/branding/production tokens
    prod_tokens = r'(?:gfx|logo|graphic|graphics|sot|vo|pkg|jingle|b-?roll|broll|cutaway|lower|supers|super|stinger|bed(?: music)?|intro(?: music)?|outro|musik|muzik|music|masuk|masuk muzik)'
    text = re.sub(r'\(\s*' + prod_tokens + r'(?:\s*[\/\-\&]\s*' + prod_tokens + r')*\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b' + prod_tokens + r'(?:\s*[\/\-\&]\s*' + prod_tokens + r')+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[-—–]\s*' + prod_tokens + r'(?:\s*[\/\-\&]\s*' + prod_tokens + r')*\s*[-—–]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?m)^\s*(?:' + prod_tokens + r')(?:[\s\/\-\&]+(?:' + prod_tokens + r'))*\s*$','', text, flags=re.IGNORECASE)
    text = re.sub(r'(?im)\b(gfx|logo|sot|vo|pkg|jingle|b-roll|broll|cutaway|lower|supers|stinger|bed music|intro music|outro music)\b', '', text)
    text = re.sub(r'<[^>]{1,256}>', ' ', text)
    
    # STEP 6d: Remove zero-noise and mixed letter-digit tokens
    text = re.sub(r'\b[A-Za-z]+\d+[A-Za-z]+\b', '', text)
    text = re.sub(r'\b0+([A-Za-z][A-Za-z0-9_\-]*)\b', r'\1', text)
    text = re.sub(r'(?m)^\s*0{2,}\s*$', '', text)
    text = re.sub(r'\b0{2,}\b', ' ', text)
    
    # STEP 6e: Remove bracketed instrument names
    inst = r'(?:bass|guitar|drum|drums|dram|keyboard|piano|violin|sax|trumpet|cello|perc|percussion|strings|horn|flute|clarinet)'
    text = re.sub(r'\(\s*' + inst + r'\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\([^\)]*\b' + inst + r'\b[^\)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?m)^\s*[-—–]+\s*.+?\s*[-—–]+\s*$', '', text)
    text = re.sub(r'[-—–]\s*[^-\n]{1,120}?\s*[-—–]', '', text)
    text = re.sub(r'(?i)\bUP\s*SOUND\b[:\s\-]*', '', text)
    text = re.sub(r'(?i)\bUPSOUND\b[:\s\-]*', '', text)
    text = re.sub(r'(?im)^\s*teks\s*:\s*.*$', '', text)
    text = re.sub(r'(?i)\bteks\s*:\s*', '', text)
    text = re.sub(r'(?im)\bSOUND\b[:\s\-]*', '', text)
    
    for tok in NOISE_TOKENS:
        text = re.sub(r'(?im)\b' + re.escape(tok) + r'\b[:\s\-]*', '', text)
    text = re.sub(r'(?m)\b[A-Z]{2,}\b(?=[\s:;,.\-]|$)', '', text)
    
    for pat in NOTES_BLACKLIST:
        try:
            text = re.sub(r'(?im)^.*(?:' + pat + r').*$\n?', '', text)
        except re.error:
            text = re.sub(pat, '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\s{2,}', ' ', text)
    
    # STEP 7: Remove Flash/Transition markers
    text = re.sub(r'\(flash\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*\*FLASH\*\*', '', text)
    text = re.sub(r'----FLASH----', '', text)
    text = re.sub(r'-FLASH-', '', text)
    text = re.sub(r'-flash-', '', text, flags=re.IGNORECASE)
    text = re.sub(r'=+FLASH=+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*+TRANSITION\*+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bFLASH\b\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bWIPE\b\s*', '', text, flags=re.IGNORECASE)
    
    # STEP 8: Remove Timestamps in parentheses
    text = re.sub(r'\(\d{2}:\d{2}:\d{2}(?::\d{2})?\)', '', text)
    
    # STEP 9: Remove Separators
    text = re.sub(r'^=+[A-Z\s]*=+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^=+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'=+(?:NEXT|CLOSING)=+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'=+', '', text)
    text = re.sub(r'#{2,}', '', text)
    text = re.sub(r'(?m)^\s*[,\-—–\.]\s*$','', text)
    text = re.sub(r'(?m)^\s*[\,\.;:\-\*\+\=\_\|\(\)\[\]\/\\]{1,}\s*$','', text)
    text = re.sub(r'(?m)^\s*[\)\]\}\s]{1,}\s*$', '', text)
    text = re.sub(r'\)\s+\)+', ')', text)
    text = re.sub(r'\)\s+\)', ')', text)
    text = re.sub(r'(?m)^[\-\u2013\u2014]{2,}\s*$', '', text)
    text = re.sub(r'(?m)^\s*[\,\.\-\s]{1,}[\,\.\-\s]{1,}\s*$', '', text)
    text = re.sub(r'[,\.\-]{2,}', lambda m: m.group(0)[0], text)
    text = re.sub(r'[\\/]{2,}', ' ', text)
    text = re.sub(r'(?m)^\s*[\\/\-]{1,}\s*$', '', text)
    text = re.sub(r'(?m)^[\s,;:\-]*\d{4}(?:[\s,;:\-/]+\d{4}){2,}[\s,;:\-\.]*$', '', text)
    text = re.sub(r'(?m)\(\s*(?:[A-Za-z]\s*\.){1,}\s*\)', '', text)
    text = re.sub(r'(?m)^\s*[\[\]\(\)\{\}\,\.;:\-\"\'\s]{1,}\s*$', '', text)
    text = re.sub(r'\(\s*[A-Z]{2,20}\s*\)', '', text)
    text = re.sub(r'(?m)^[\s\,:\-]*[A-Z]{1,20}\s*[:\?\.]\s*.*$', '', text)
    text = re.sub(r'(?m)^\s*:\s*[\.\?\!\s]*$', '', text)
    text = re.sub(r'(?m)^\s*[A-Z]\s*\.\s*$', '', text)
    
    # STEP 10: Remove Islamic Greetings
    text = re.sub(r'^Assalamualaikum\.?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^Salam\s+Malaysia\s+Madani\.?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\bbyeeee\b', '', text, flags=re.IGNORECASE)
    
    # STEP 11: Remove Person/Segment Labels
    text = re.sub(r'\(fastcut[^\)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Subt:[^\)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^HI\.\.\s*$', '', text, flags=re.MULTILINE)
    
    # STEP 12: Remove Special Characters
    text = re.sub(r'â€"', '-', text)
    text = re.sub(r'â€™', "'", text)
    text = re.sub(r'â€˜', "'", text)
    text = re.sub(r'â€œ', '"', text)
    text = re.sub(r'â€', '"', text)
    text = re.sub(r'Â', ' ', text)
    
    # STEP 13: Clean up repeated periods and symbols
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'…+', '', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'(\d)\s+%', r'\1%', text)
    
    # STEP 14: Process lines
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue

        if re.match(r'^[,\.\s;:\(\)\-=#\*…]+$', line):
            continue
        if re.match(r'^[-—–]{1,}\s*$', line):
            continue
        if re.match(r'^[\)\]\}]{1,}\s*$', line):
            continue
        if re.match(r'^Q\.?\:?\s*$', line, re.IGNORECASE):
            continue
        if re.match(r'^H-\w+', line, re.IGNORECASE):
            continue
        if re.match(r'^B-?\s*\w+', line, re.IGNORECASE):
            if re.match(r'^B-?\s*ROLL\b', line, re.IGNORECASE) or re.match(r'^B-?\s*\d+', line, re.IGNORECASE) or re.match(r'^B-?\s*[A-Z0-9]{2,}\b', line):
                continue
        if re.match(r'^[\:\,\-\s]*[A-Z]{1,20}\s*[:\?\.]\s*.*$', line):
            continue
        if re.match(r'^\s*:\s*[\.\?\!\s]*$', line):
            continue
        if re.match(r'^\s*[A-Z]\s*\.\s*$', line):
            continue
        if re.match(r'^[\-\u2013\u2014]{2,}\s*$', line):
            continue
        if re.match(r'^[\,\.\-\s]{2,}$', line):
            continue
        if re.match(r'^[\\/\s]{1,}$', line):
            continue
        if re.match(r'^\(\s*[A-Z]{2,20}\s*\)\s*$', line):
            continue
        if re.match(r'^[=\-\*#…]+$', line):
            continue
        if re.match(r'^(NEXT|CLOSING)$', line, re.IGNORECASE):
            continue
        
        line = re.sub(r'^\d{1,2}:\d{2}:\d{2}:\d{2}\s*', '', line)
        line = re.sub(r'^\d{2}\.\d{2}\.\d{2}\.\d{2}\s*', '', line)
        line = re.sub(r'^[,\s:;\-]+', '', line)
        line = re.sub(r'[,\s:;]+$', '', line)
        
        if not line or len(line) < 2:
            continue

        if re.match(r'^[\d\s,;:\-/]{4,}$', line) and re.search(r'\d{4}', line):
            yrs = re.findall(r'\d{4}', line)
            if len(yrs) >= 3:
                continue

        words_tokens = re.findall(r"\b\w+\b", line)
        if words_tokens:
            single_tokens = [w for w in words_tokens if len(w) == 1]
            if len(single_tokens) / max(1, len(words_tokens)) >= 0.5 and len(words_tokens) >= 2:
                continue

        punct_chars = sum(1 for ch in line if not ch.isalnum() and not ch.isspace())
        if punct_chars / max(1, len(line)) > 0.5:
            if not re.search(r'\w\s+\w', line):
                continue
        
        alpha_only = re.sub(r'[^A-Za-z]', '', line)
        if alpha_only and alpha_only.isupper():
            continue

        words = re.findall(r'\b\w+\b', line)
        if words:
            upper_words = [w for w in words if w.isalpha() and w.isupper()]
            try:
                if len(upper_words) / len(words) >= 0.6:
                    continue
            except ZeroDivisionError:
                pass
        
        cleaned_lines.append(line)
    
    # STEP 15: Final cleanup
    text = '\n'.join(cleaned_lines)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    return text.strip()

# --- Universal Cleaner with Sentence Splitting ---
def clean_text(
    text,
    remove_citations=True,
    min_words=3,
    english_threshold=0.2,
    uppercase_threshold=0.9,
    min_upper_run=4,
    keep_linebreaks=True,
    one_sentence_per_line=True
):
    """Clean text while preserving actual news content.
    
    Args:
        one_sentence_per_line: If True, output one sentence per line
    """
    
    # Protect abbreviations before processing
    text = protect_abbreviations(text)
    
    text = remove_rtm_leftovers(text)

    if remove_citations:
        text = re.sub(r'\bet al\.?\s*\(\d{4}[a-z]?\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\([^)]*\b\d{4}\b[^)]*\)', '', text)

    lines = text.splitlines()
    clean_lines = []

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue

        # Drop lines that are likely English noise
        if is_likely_english(line_strip):
            continue

        m = re.match(r'^\s*\d+[\.)]\s+(.*)', line_strip)
        if m:
            line_strip = m.group(1).strip()


        # Split into sentences - use multiple delimiters
        sentences = re.split(r'(?<=[.!?])\s+', line_strip)
        filtered_sentences = []

        for sentence in sentences:
            sentence = remove_uppercase_runs(sentence, min_upper_run=min_upper_run)
            
            if not sentence:
                continue
            
            if is_noisy_sentence(sentence, min_words):
                continue
            
            alpha_letters = re.sub(r'[^A-Za-z]', '', sentence)
            if alpha_letters and alpha_letters.isupper():
                continue

            if sentence.isupper() and len(sentence) > 10:
                continue

            words = re.findall(r'\b\w+\b', sentence)
            if len(words) < min_words:
                continue

            short_tokens = [w for w in words if len(w) <= 3]
            mixed_ld = [w for w in words if re.search(r'[A-Za-z]+\d|\d+[A-Za-z]+', w)]
            if words and (len(short_tokens) / len(words) >= 0.6 or len(mixed_ld) >= 1):
                continue

            upper_words = [w for w in words if w.isalpha() and w.isupper() and len(w) >= 2]
            if words and (len(upper_words) / len(words)) >= 0.5:
                continue

            filtered_sentences.append(sentence)

        if filtered_sentences:
            clean_lines.extend(filtered_sentences) if one_sentence_per_line else clean_lines.append(" ".join(filtered_sentences))

    # Unprotect abbreviations
    cleaned_text = "\n".join(clean_lines) if (keep_linebreaks or one_sentence_per_line) else " ".join(clean_lines)
    cleaned_text = unprotect_abbreviations(cleaned_text)
    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
    
    # Ensure one sentence per line if requested
    if one_sentence_per_line:
        cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)
    
    return cleaned_text.strip()

# --- Process all TXT files in a folder ---
def clean_txt_folder(input_folder, output_folder, **kwargs):
    """Process all TXT files in input_folder and save cleaned versions to output_folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    txt_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".txt")]
    
    if not txt_files:
        print(f"⚠️  No TXT files found in {input_folder}")
        return
    
    print(f"📂 Found {len(txt_files)} TXT files to process")
    print("="*80)

    for idx, filename in enumerate(txt_files, 1):
        try:
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace(".txt", "_cleaned.txt")
            output_path = os.path.join(output_folder, output_filename)

            with open(input_path, "r", encoding="utf-8", errors='ignore') as f:
                raw_text = f.read()

            cleaned_text = clean_text(raw_text, **kwargs)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"✅ [{idx}/{len(txt_files)}] {filename} -> {output_filename}")
            
        except Exception as e:
            print(f"❌ [{idx}/{len(txt_files)}] Error processing {filename}: {str(e)}")
            continue
    
    print("="*80)
    print(f"🎉 Processing complete! Cleaned files saved to: {output_folder}")


def _process_file_tuple(args):
    """Helper to allow multiprocessing: args = (input_path, output_path, kwargs)"""
    input_path, output_path, kwargs = args
    try:
        with open(input_path, "r", encoding="utf-8", errors='ignore') as f:
            raw_text = f.read()
        cleaned_text = clean_text(raw_text, **kwargs)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        return (input_path, None)
    except Exception as e:
        return (input_path, str(e))


def clean_txt_folder_parallel(input_folder, output_folder, workers=None, chunk_size=32, show_progress=True, **kwargs):
    """Parallel version of clean_txt_folder using multiprocessing.

    - workers: number of worker processes (None => cpu_count()-1)
    - chunk_size: number of files to batch per worker submit (controls memory)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    txt_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.txt')]
    if not txt_files:
        print(f"⚠️  No TXT files found in {input_folder}")
        return

    nfiles = len(txt_files)
    workers = workers or max(1, multiprocessing.cpu_count() - 1)
    print(f"📂 Processing {nfiles} files with {workers} workers")

    tasks = []
    for filename in txt_files:
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace('.txt', '_cleaned.txt')
        output_path = os.path.join(output_folder, output_filename)
        tasks.append((input_path, output_path, kwargs))

    errors = []
    processed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_file_tuple, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            inp = futures[fut]
            inp, err = fut.result()
            processed += 1
            if err:
                errors.append((inp, err))
                if show_progress:
                    print(f"❌ [{processed}/{nfiles}] Error: {inp} -> {err}")
            else:
                if show_progress:
                    print(f"✅ [{processed}/{nfiles}] {os.path.basename(inp)}")

    print('=' * 80)
    print(f"🎉 Processing complete! {processed} files processed. Errors: {len(errors)}")
    if errors:
        print('First errors:')
        for e in errors[:5]:
            print(e)

# ---- Runner: clean the workspace raw folder ----
if __name__ == "__main__":
    """Process all raw TXT files in the workspace raw folder and save cleaned outputs.

    This runner is safe and idempotent: it writes cleaned files to
    '2.0 clean/cleaned_raw' next to this script. Filenames are preserved
    with a '_cleaned.txt' suffix.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw TXT files from the workspace")
    parser.add_argument("--input", "-i", default=r"d:\WarisanTeamData\snake file\0.0 raw file",
                        help="Path to folder containing raw TXT files")
    parser.add_argument("--output", "-o", default=r"d:\WarisanTeamData\snake file\2.0 clean\cleaned_raw",
                        help="Output folder for cleaned files")
    parser.add_argument("--min-words", type=int, default=2, help="Minimum words per sentence to keep")
    parser.add_argument("--min-upper-run", type=int, default=3, help="Minimum uppercase-run length to strip")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file progress prints")
    parser.add_argument("--workers", type=int, default=0, help="Use N worker processes for parallel cleaning (0 = disabled)")
    parser.add_argument("--one-sentence-per-line", action="store_true", default=True, 
                        help="Output one sentence per line (default: True)")

    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Use clean_txt_folder but allow quiet mode to reduce console noise
    if args.workers and args.workers > 0:
        # Run parallel cleaning
        clean_txt_folder_parallel(input_folder, output_folder, workers=args.workers, show_progress=not args.quiet,
                                 min_words=args.min_words, min_upper_run=args.min_upper_run,
                                 one_sentence_per_line=args.one_sentence_per_line)
    elif args.quiet:
        # Temporarily suppress prints
        import sys

        class _NullWriter:
            def write(self, *a, **k):
                pass

            def flush(self):
                pass

        _stdout = sys.stdout
        try:
            sys.stdout = _NullWriter()
            clean_txt_folder(input_folder, output_folder,
                             min_words=args.min_words,
                             min_upper_run=args.min_upper_run,
                             one_sentence_per_line=args.one_sentence_per_line)
        finally:
            sys.stdout = _stdout
        print(f"Processing complete (quiet). Cleaned files saved to: {output_folder}")
    else:
        clean_txt_folder(input_folder, output_folder,
                         min_words=args.min_words,
                         min_upper_run=args.min_upper_run,
                         one_sentence_per_line=args.one_sentence_per_line)