# data_inspector.py
# Inspect and clean the scraped Bangla dataset

import os
import re

INPUT_FILE  = 'data/bangla_dataset.txt'
OUTPUT_FILE = 'data/bangla_clean.txt'

# ─────────────────────────────────────────
# READ RAW DATA
# ─────────────────────────────────────────
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    raw_text = f.read()

print("=" * 55)
print("  DATASET INSPECTION")
print("=" * 55)
print(f"  Raw file size    : {len(raw_text):,} characters")
print(f"  Raw lines        : {len(raw_text.splitlines()):,}")

# ─────────────────────────────────────────
# SHOW SAMPLE OF RAW TEXT
# ─────────────────────────────────────────
print("\n── Raw Sample (first 500 chars) ──")
print(raw_text[:500])

# ─────────────────────────────────────────
# ANALYZE CHARACTERS
# ─────────────────────────────────────────
unique_chars = sorted(set(raw_text))
print(f"\n── Character Analysis ──")
print(f"  Unique characters : {len(unique_chars)}")

# separate bangla vs non-bangla characters
bangla_chars  = [c for c in unique_chars if '\u0980' <= c <= '\u09FF']
english_chars = [c for c in unique_chars if c.isascii() and c.isprintable()]
other_chars   = [c for c in unique_chars if c not in bangla_chars
                 and c not in english_chars]

print(f"  Bangla chars      : {len(bangla_chars)}")
print(f"  English/ASCII     : {len(english_chars)}")
print(f"  Other (symbols)   : {len(other_chars)}")
print(f"\n  Other chars       : {repr(''.join(other_chars[:50]))}")

# ─────────────────────────────────────────
# DEEP CLEANING
# ─────────────────────────────────────────
def deep_clean(text):
    # remove section separators we added
    text = re.sub(r'={2,}.*?={2,}', '', text)

    # remove article title markers
    text = re.sub(r'#.*?\n', '', text)

    # remove English words (optional — keeps pure Bangla)
    # text = re.sub(r'[a-zA-Z]+', '', text)

    # remove URLs
    text = re.sub(r'http\S+', '', text)

    # remove digits (Arabic numerals) — keep Bangla numerals ০-৯
    # text = re.sub(r'[0-9]+', '', text)

    # remove extra punctuation clusters
    text = re.sub(r'[^\u0980-\u09FF\s.,;:!?()"\'\u09E6-\u09EF0-9a-zA-Z]', '', text)

    # normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # remove very short lines
    lines = text.split('\n')
    lines = [l.strip() for l in lines if len(l.strip()) > 10]
    text  = '\n'.join(lines)

    return text.strip()

cleaned_text = deep_clean(raw_text)

# ─────────────────────────────────────────
# SHOW SAMPLE OF CLEAN TEXT
# ─────────────────────────────────────────
print("\n── Cleaned Sample (first 500 chars) ──")
print(cleaned_text[:500])

# ─────────────────────────────────────────
# FINAL STATS
# ─────────────────────────────────────────
unique_after = sorted(set(cleaned_text))
print(f"\n── After Cleaning ──")
print(f"  Characters       : {len(cleaned_text):,}")
print(f"  Unique chars     : {len(unique_after)}")
print(f"  Size reduction   : {100 - len(cleaned_text)/len(raw_text)*100:.1f}%")

# ─────────────────────────────────────────
# SAVE CLEAN DATASET
# ─────────────────────────────────────────
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(cleaned_text)

print(f"\n  ✅ Clean dataset saved to: {OUTPUT_FILE}")
print(f"  Final size: {os.path.getsize(OUTPUT_FILE)/1024/1024:.3f}MB")
print("=" * 55)