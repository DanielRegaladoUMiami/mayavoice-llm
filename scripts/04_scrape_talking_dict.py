#!/usr/bin/env python3
"""
Scrape the Swarthmore Talking Dictionaries for Mayan languages of Guatemala.
https://talkingdictionary.swarthmore.edu/guatemala.php

Extracts: headword (maya), IPA, part of speech, English translation,
Spanish translation, audio URL, speaker, entry ID.

Output: one CSV per language + a combined CSV.

Usage:
    # Scrape all languages
    python 04_scrape_talking_dict.py --output-dir ../data/diccionarios/scraped/

    # Scrape specific language(s)
    python 04_scrape_talking_dict.py --languages kaqchikel ixil kiche \
        --output-dir ../data/diccionarios/scraped/

    # Dry run (just count entries)
    python 04_scrape_talking_dict.py --dry-run
"""
import argparse
import csv
import re
import sys
import time
import urllib.request
from pathlib import Path
from bs4 import BeautifulSoup

BASE_URL = "https://talkingdictionary.swarthmore.edu"

# All 22 Mayan languages on the Guatemala page
LANGUAGES = [
    'achi', 'akateko', 'awakateko', 'chorti', 'chalchiteko', 'chuj',
    'itza', 'ixil', 'kiche', 'kaqchikel', 'mam', 'mopan', 'ngabere',
    'poqomam', 'poqomchi', 'qanjobal', 'qeqchi', 'sakapulteko',
    'sipakapense', 'tektiteko', 'tzutujil', 'uspanteko',
]

HEADERS = {
    'User-Agent': 'MayaVoice Research (University of Miami, dxr1491@miami.edu)',
    'Accept': 'text/html',
}

CSV_FIELDS = [
    'language', 'entry_id', 'headword', 'ipa', 'part_of_speech',
    'english', 'spanish', 'audio_url', 'image_url', 'speaker',
]


def fetch_page(url: str, retries: int = 3) -> str:
    """Fetch a URL with retries and polite delay."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode('utf-8', errors='replace')
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt+1} after error: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {e}")
                return ""


def get_entry_count(html: str) -> int:
    """Extract total entry count from page footer."""
    match = re.search(r'(\d+)\s*entries', html)
    return int(match.group(1)) if match else 0


def get_max_entry_id(html: str) -> int:
    """Find the highest entry ID referenced on the page."""
    ids = re.findall(r'\?entry=(\d+)', html)
    return max(int(i) for i in ids) if ids else 0


def parse_entries(html: str, language: str) -> list[dict]:
    """Parse all dictionary entries from an HTML page."""
    soup = BeautifulSoup(html, 'html.parser')
    entries = []

    # Each entry starts with an <h3> (skip the first one which is version info)
    h3_tags = soup.find_all('h3')

    for h3 in h3_tags:
        headword = h3.get_text(strip=True)

        # Skip non-entry headers (version info, section headers)
        if 'version' in headword.lower() or len(headword) > 100:
            continue

        insides = h3.find_next_sibling('div', class_='insides')
        if not insides:
            continue

        entry = {
            'language': language,
            'entry_id': '',
            'headword': headword,
            'ipa': '',
            'part_of_speech': '',
            'english': '',
            'spanish': '',
            'audio_url': '',
            'image_url': '',
            'speaker': '',
        }

        # Extract audio URL from <span class="audio"><a href="...mp3">
        audio_span = insides.find('span', class_='audio')
        if audio_span:
            audio_id = audio_span.get('id', '')
            if audio_id:
                entry['entry_id'] = audio_id.replace('audio_', '')
            audio_link = audio_span.find('a', class_='audio-file')
            if audio_link and audio_link.get('href'):
                entry['audio_url'] = BASE_URL + audio_link['href']

        # Extract image
        img = insides.find('img')
        if img and img.get('src'):
            entry['image_url'] = BASE_URL + img['src']

        # Parse <p> tags for IPA, POS, English, Spanish, Speaker
        paragraphs = insides.find_all('p')
        for p in paragraphs:
            p_class = ' '.join(p.get('class', []))
            text = p.get_text(strip=True)

            if 'metadata' in p_class:
                # Speaker info
                speaker_match = re.search(r'Speaker:\s*(.+)', text)
                if speaker_match:
                    entry['speaker'] = speaker_match.group(1).strip()
                continue

            if text.startswith('(Spanish)'):
                entry['spanish'] = text.replace('(Spanish)', '').strip()
            elif text.startswith('[') or text.startswith('('):
                # First p: [IPA] part_of_speech English translation
                # Parse IPA
                ipa_match = re.search(r'\[([^\]]+)\]', text)
                if ipa_match:
                    entry['ipa'] = ipa_match.group(1)

                # Find spans for POS
                spans = p.find_all('span')
                for s in spans:
                    scls = ' '.join(s.get('class', []))
                    stxt = s.get_text(strip=True)
                    if 'pos' in scls or 'part-of-speech' in scls:
                        entry['part_of_speech'] = stxt
                    elif stxt.startswith('['):
                        entry['ipa'] = stxt.strip('[]')

                # Remove known parts to get English translation
                remaining = text
                if entry['ipa']:
                    remaining = remaining.replace(f"[{entry['ipa']}]", '')
                if entry['part_of_speech']:
                    remaining = remaining.replace(entry['part_of_speech'], '')
                remaining = remaining.strip()
                if remaining and not remaining.startswith('(Spanish)'):
                    entry['english'] = remaining

            elif not entry['english'] and text and not text.startswith('('):
                # Fallback: might be English translation without IPA
                entry['english'] = text

        # If we still don't have POS, try regex on first p
        if not entry['part_of_speech'] and paragraphs:
            first_p = paragraphs[0].get_text(strip=True)
            pos_patterns = [
                'sustantivo inalienable', 'sustantivo', 'verbo transitivo',
                'verbo intransitivo', 'verbo', 'adjetivo', 'adverbio',
                'pronombre', 'preposición', 'conjunción', 'partícula',
                'noun', 'verb', 'adjective', 'adverb',
            ]
            for pat in pos_patterns:
                if pat in first_p.lower():
                    entry['part_of_speech'] = pat
                    # Clean it from english
                    entry['english'] = entry['english'].replace(pat, '').strip()
                    break

        entries.append(entry)

    return entries


def scrape_language(language: str, dry_run: bool = False) -> list[dict]:
    """Scrape all entries for a single language by paginating through ?entry=N pages."""
    url = f"{BASE_URL}/{language}/"
    print(f"\n{'='*50}")
    print(f"Scraping: {language}")
    print(f"URL: {url}")

    # First fetch main page to get reported count
    html = fetch_page(url)
    if not html:
        print(f"  ERROR: Could not fetch page for {language}")
        return []

    entry_count = get_entry_count(html)
    print(f"  Entries reported: {entry_count}")

    if dry_run:
        return []

    # Paginate through ?entry=1, ?entry=2, ... (each returns ~15-20 entries)
    # The main page (no ?entry=) shows the first 100, but paginated pages
    # use different offsets. We collect all unique entries by audio ID.
    all_entries = {}  # keyed by entry_id to deduplicate
    page = 0
    consecutive_no_new = 0
    max_pages = 100  # safety limit

    while page <= max_pages and consecutive_no_new < 3:
        if page == 0:
            page_html = html  # reuse already-fetched main page
        else:
            time.sleep(0.4)  # polite delay
            page_html = fetch_page(f"{url}?entry={page}")

        if not page_html:
            page += 1
            continue

        entries = parse_entries(page_html, language)

        new_count = 0
        for e in entries:
            eid = e['entry_id'] or e['headword']  # fallback key
            if eid not in all_entries:
                all_entries[eid] = e
                new_count += 1

        if new_count == 0:
            consecutive_no_new += 1
        else:
            consecutive_no_new = 0

        if page % 10 == 0 or new_count > 0:
            print(f"  Page {page}: {len(entries)} entries ({new_count} new), total: {len(all_entries)}")

        page += 1

    result = list(all_entries.values())
    print(f"  Final count: {len(result)} entries (from {page} pages)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Scrape Swarthmore Talking Dictionaries")
    parser.add_argument('--languages', nargs='+', default=None,
                        help='Specific languages to scrape (default: all)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for CSV files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show entry counts, do not scrape')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Delay between language requests (seconds)')
    args = parser.parse_args()

    languages = args.languages or LANGUAGES
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []
    stats = {}

    for lang in languages:
        if lang not in LANGUAGES:
            print(f"WARNING: {lang} not in known languages, skipping")
            continue

        entries = scrape_language(lang, dry_run=args.dry_run)
        all_entries.extend(entries)
        stats[lang] = len(entries)

        if not args.dry_run and entries:
            # Save per-language CSV
            lang_file = output_dir / f"{lang}_dictionary.csv"
            with open(lang_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(entries)
            print(f"  Saved: {lang_file} ({len(entries)} entries)")

        time.sleep(args.delay)

    # Save combined CSV
    if not args.dry_run and all_entries:
        combined_file = output_dir / "all_languages_dictionary.csv"
        with open(combined_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(all_entries)
        print(f"\nCombined: {combined_file} ({len(all_entries)} entries)")

    # Summary
    print(f"\n{'='*50}")
    print(f"SCRAPE SUMMARY")
    print(f"{'='*50}")
    print(f"{'Language':<15} {'Entries':<10}")
    print("-" * 25)
    total = 0
    for lang, count in sorted(stats.items()):
        print(f"  {lang:<15} {count:<10}")
        total += count
    print("-" * 25)
    print(f"  {'TOTAL':<15} {total:<10}")


if __name__ == '__main__':
    main()
