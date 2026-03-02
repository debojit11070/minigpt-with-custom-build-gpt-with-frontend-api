# scraper.py
# Scrapes Bangla Wikipedia articles and saves as plain text

import requests
from bs4 import BeautifulSoup
import time
import os
import re

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
OUTPUT_FILE    = 'data/bangla_dataset.txt'
TARGET_SIZE_MB = 1.0   # stop when we reach 1MB
DELAY_SECONDS  = 1.0   # wait between requests (be respectful)

# List of Bangla Wikipedia article titles to scrape
# These are well-known topics with rich Bangla content
TOPICS = [
    'বাংলাদেশ',           # Bangladesh
    'ঢাকা',               # Dhaka
    'বাংলা_ভাষা',         # Bangla language
    'রবীন্দ্রনাথ_ঠাকুর',  # Rabindranath Tagore
    'কাজী_নজরুল_ইসলাম',  # Kazi Nazrul Islam
    'বাংলাদেশের_ইতিহাস',  # History of Bangladesh
    'মুক্তিযুদ্ধ',        # Liberation War
    'পদ্মা_নদী',          # Padma River
    'সুন্দরবন',           # Sundarbans
    'চট্টগ্রাম',          # Chittagong
    'ইসলাম',              # Islam
    'হিন্দুধর্ম',         # Hinduism
    'বিজ্ঞান',            # Science
    'পদার্থবিজ্ঞান',      # Physics
    'গণিত',               # Mathematics
    'চিকিৎসাবিজ্ঞান',    # Medicine
    'কবিতা',              # Poetry
    'বাংলা_সাহিত্য',      # Bangla literature
    'শেখ_মুজিবুর_রহমান',  # Sheikh Mujibur Rahman
    'বাংলাদেশের_অর্থনীতি', # Economy of Bangladesh
]

# ─────────────────────────────────────────
# SCRAPER FUNCTIONS
# ─────────────────────────────────────────
def clean_text(text):
    """
    Remove unwanted characters and normalize whitespace
    """
    # remove references like [1], [2], [note 1]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[note \d+\]', '', text)
    text = re.sub(r'\[তথ্যসূত্র প্রয়োজন\]', '', text)

    # remove multiple spaces and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # remove lines that are too short (likely headers/labels)
    lines = text.split('\n')
    lines = [l.strip() for l in lines if len(l.strip()) > 20]
    text  = '\n'.join(lines)

    return text.strip()


def scrape_wikipedia_article(title):
    """
    Scrape a single Bangla Wikipedia article by title
    """
    url    = f"https://bn.wikipedia.org/wiki/{title}"
    headers = {
        'User-Agent': 'MiniGPT-Dataset-Builder/1.0 (Educational Project)'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup    = BeautifulSoup(response.content, 'html.parser')

        # find the main content div
        content = soup.find('div', {'id': 'mw-content-text'})
        if not content:
            print(f"  ✗ No content found for: {title}")
            return None

        # extract all paragraphs
        paragraphs = content.find_all('p')
        text       = '\n'.join([p.get_text() for p in paragraphs])
        text       = clean_text(text)

        if len(text) < 100:
            print(f"  ✗ Too short for: {title}")
            return None

        return text

    except requests.RequestException as e:
        print(f"  ✗ Failed to fetch {title}: {e}")
        return None


def get_current_size_mb(filepath):
    """
    Returns current file size in MB
    """
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)


# ─────────────────────────────────────────
# MAIN SCRAPING LOOP
# ─────────────────────────────────────────
def main():
    os.makedirs('data', exist_ok=True)

    print("=" * 55)
    print("  Bangla Wikipedia Dataset Builder")
    print("=" * 55)
    print(f"  Target size: {TARGET_SIZE_MB}MB")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Articles to scrape: {len(TOPICS)}")
    print("=" * 55)

    total_articles = 0
    total_chars    = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:

        for i, topic in enumerate(TOPICS):

            # check if we've reached target size
            current_mb = get_current_size_mb(OUTPUT_FILE)
            if current_mb >= TARGET_SIZE_MB:
                print(f"\n✅ Reached target size of {TARGET_SIZE_MB}MB!")
                break

            print(f"\n[{i+1}/{len(TOPICS)}] Scraping: {topic}")
            print(f"  Current size: {current_mb:.3f}MB / {TARGET_SIZE_MB}MB")

            text = scrape_wikipedia_article(topic)

            if text:
                # write article with separator
                f.write(f"\n\n{'='*50}\n")
                f.write(f"# {topic}\n")
                f.write(f"{'='*50}\n\n")
                f.write(text)
                f.flush()  # write to disk immediately

                chars = len(text)
                total_chars    += chars
                total_articles += 1
                print(f"  ✓ Scraped {chars:,} characters")

            # respectful delay between requests
            if i < len(TOPICS) - 1:
                print(f"  Waiting {DELAY_SECONDS}s...")
                time.sleep(DELAY_SECONDS)

    # ─────────────────────────────────────────
    # FINAL REPORT
    # ─────────────────────────────────────────
    final_mb = get_current_size_mb(OUTPUT_FILE)
    print("\n" + "=" * 55)
    print("  SCRAPING COMPLETE")
    print("=" * 55)
    print(f"  Articles scraped: {total_articles}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Final file size:  {final_mb:.3f}MB")
    print(f"  Saved to: {OUTPUT_FILE}")
    print("=" * 55)


if __name__ == '__main__':
    main()