import sys, os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---- Device Set to CPU ----
# Explicitly use 'cpu' for the model
DEVICE = "cpu"
print(f"[DEDUP] Using device: {DEVICE}")

# --- Deduplication Function ---
def deduplicate(sentences, threshold=0.9, model_name="distiluse-base-multilingual-cased-v1"):
    """
    Deduplicate sentences using embeddings + cosine similarity (FAISS).
    threshold: similarity above this is considered duplicate.
    """
    print(f"[DEDUP] Loading model {model_name} on {DEVICE}...")
    model = SentenceTransformer(model_name, device=DEVICE)

    print(f"[DEDUP] Encoding {len(sentences)} sentences...")
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print("[DEDUP] Searching for duplicates...")
    similarities, neighbors = index.search(embeddings, 2)  # top-2 (itself + nearest neighbor)

    keep = []
    seen = set()
    for i, (score, neighbor) in enumerate(zip(similarities, neighbors)):
        if i in seen:
            continue
        if score[1] >= threshold:  # neighbor[0] is itself, neighbor[1] is nearest
            seen.add(neighbor[1])
        keep.append(sentences[i])

    print(f"[DEDUP] Reduced {len(sentences)} → {len(keep)} sentences")
    return keep


def read_sentences(path):
    """Read sentences line by line from input file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def save_sentences(sentences, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"[DEDUP] Saved output → {path}")


# Main Execution Block
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python Deduplication.py input.txt output.txt")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    print(f"[DEDUP] Input: {input_path}")
    print(f"[DEDUP] Output: {output_path}")

    sentences = read_sentences(input_path)
    deduped = deduplicate(sentences, threshold=0.9)
    save_sentences(deduped, output_path)

