#!/usr/bin/env python3
"""
Issue #8/#9: Merge MayanV repository data into our parallel texts corpus.

Reads all splits (train/dev/test) from a local clone of transducens/mayanv,
strips #code# prefixes from ES lines, deduplicates against existing data,
and appends only genuinely new pairs.

Usage:
    python 03_merge_mayanv.py --mayanv-dir ../mayanv/MayanV \
        --data-dir ../data/textos-paralelos/ --dry-run

    python 03_merge_mayanv.py --mayanv-dir ../mayanv/MayanV \
        --data-dir ../data/textos-paralelos/
"""
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

# MayanV code → our folder name mapping
CODE_TO_FOLDER = {
    'acr': 'achi',
    'agu': 'awakateko',
    'cac': 'chuj',
    'itz': 'itza',
    'kek': 'qeqchi',
    'kjb': 'qanjobal',
    'mam': 'mam',
    'poc': 'poqomam',
    'poh': 'poqomchi',
    'quc': 'kiche',
    'qum': 'sipakapense',
    'ttc': 'tektiteko',
    'tzj': 'tzutujil',
}


def strip_code_prefix(line: str) -> str:
    """Remove #code# prefix from ES lines (e.g., '#acr# El texto' → 'El texto')."""
    return re.sub(r'^#\w+#\s*', '', line).strip()


def load_mayanv_lang(lang_dir: Path, code: str) -> list[tuple[str, str]]:
    """Load all splits (train/dev/test) for one language from MayanV."""
    pairs = []
    for split in ['train', 'dev', 'test']:
        split_dir = lang_dir / split
        es_file = split_dir / 'data.es'
        maya_file = split_dir / f'data.{code}'

        if not es_file.exists() or not maya_file.exists():
            continue

        with open(es_file, 'r', encoding='utf-8') as f:
            es_lines = [strip_code_prefix(l) for l in f]
        with open(maya_file, 'r', encoding='utf-8') as f:
            maya_lines = [l.strip() for l in f]

        # Align to shorter if mismatch
        if len(es_lines) != len(maya_lines):
            min_len = min(len(es_lines), len(maya_lines))
            es_lines = es_lines[:min_len]
            maya_lines = maya_lines[:min_len]

        for es, maya in zip(es_lines, maya_lines):
            if es and maya and len(es) >= 2 and len(maya) >= 2:
                pairs.append((es, maya))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Merge MayanV data into parallel texts corpus")
    parser.add_argument('--mayanv-dir', type=str, required=True, help='Path to MayanV/ directory')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to textos-paralelos/')
    parser.add_argument('--dry-run', action='store_true', help='Validate only, do not write')
    args = parser.parse_args()

    mayanv_dir = Path(args.mayanv_dir)
    data_dir = Path(args.data_dir)

    if not mayanv_dir.exists():
        print(f"ERROR: MayanV directory not found: {mayanv_dir}")
        sys.exit(1)

    total_new = 0
    total_existing = 0
    total_dupes = 0
    stats = {}

    for code in sorted(CODE_TO_FOLDER.keys()):
        folder = CODE_TO_FOLDER[code]
        lang_src = mayanv_dir / code
        lang_dst = data_dir / folder

        if not lang_src.exists():
            print(f"  SKIP {code}: not found in MayanV")
            continue

        # Load MayanV data
        mayanv_pairs = load_mayanv_lang(lang_src, code)
        print(f"\n{'='*50}")
        print(f"{folder} ({code}): {len(mayanv_pairs)} total pairs from MayanV")

        # Load existing corpus
        es_file = lang_dst / 'data.es'
        maya_file = lang_dst / f'data.{code}'
        existing_es = set()

        if es_file.exists():
            with open(es_file, 'r', encoding='utf-8') as f:
                existing_es = {l.strip().lower() for l in f if l.strip()}
            print(f"  Existing corpus: {len(existing_es)} pairs")
        else:
            print(f"  NEW language — will create directory: {lang_dst}")

        # Deduplicate
        new_pairs = []
        dupes = 0
        for es, maya in mayanv_pairs:
            if es.strip().lower() in existing_es:
                dupes += 1
                continue
            # Also deduplicate within the new data itself
            if es.strip().lower() not in {p[0].strip().lower() for p in new_pairs}:
                new_pairs.append((es, maya))

        print(f"  Duplicates (already in corpus): {dupes}")
        print(f"  New pairs to add: {len(new_pairs)}")

        total_new += len(new_pairs)
        total_existing += len(existing_es)
        total_dupes += dupes
        stats[folder] = {
            'code': code,
            'mayanv_total': len(mayanv_pairs),
            'existing': len(existing_es),
            'duplicates': dupes,
            'new_added': len(new_pairs),
        }

        if args.dry_run or len(new_pairs) == 0:
            continue

        # Create directory if new language
        lang_dst.mkdir(parents=True, exist_ok=True)

        # Append new pairs
        with open(es_file, 'a', encoding='utf-8') as f:
            for es, _ in new_pairs:
                f.write(es + '\n')
        with open(maya_file, 'a', encoding='utf-8') as f:
            for _, maya in new_pairs:
                f.write(maya + '\n')

        # Verify alignment
        with open(es_file) as f:
            es_count = len([l for l in f if l.strip()])
        with open(maya_file) as f:
            maya_count = len([l for l in f if l.strip()])

        if es_count != maya_count:
            print(f"  ⚠ ALIGNMENT WARNING: ES={es_count}, Maya={maya_count}")
        else:
            print(f"  ✓ Total corpus now: {es_count} pairs")

    # Summary
    print(f"\n{'='*50}")
    print(f"MERGE SUMMARY")
    print(f"{'='*50}")
    print(f"  Languages processed: {len(stats)}")
    print(f"  Existing pairs:      {total_existing}")
    print(f"  Duplicates skipped:  {total_dupes}")
    print(f"  New pairs added:     {total_new}")
    print(f"  Total corpus now:    {total_existing + total_new}")

    if args.dry_run:
        print(f"\n[DRY RUN] No files modified.")
    else:
        print(f"\nNext step: regenerate splits with 01_create_splits.py")

    # Per-language table
    print(f"\n{'Language':<15} {'Code':<6} {'Existing':<10} {'New':<8} {'Total':<10}")
    print("-" * 50)
    for folder, s in sorted(stats.items()):
        total = s['existing'] + s['new_added']
        print(f"  {folder:<15} {s['code']:<6} {s['existing']:<10} {s['new_added']:<8} {total:<10}")


if __name__ == '__main__':
    main()
