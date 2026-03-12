#!/usr/bin/env python3
"""
Issue #1: Limpiar dataset — eliminar líneas vacías en Awakateko e Itza'
 
Removes parallel text pairs where the Maya side is empty.
Also removes exact duplicate ES lines (keeping first occurrence).
Produces clean, aligned parallel text files.
"""
import os
import argparse
from pathlib import Path

LANG_CODES = {
    'awakateko': 'agu', 'chuj': 'cac', 'itza': 'itz', 'kiche': 'quc',
    'mam': 'mam', 'poqomam': 'poc', 'poqomchi': 'poh', 'qanjobal': 'kjb',
    'qeqchi': 'kek', 'sipakapense': 'qum', 'tektiteko': 'ttc', 'tzutujil': 'tzj'
}

def clean_language(lang_dir: Path, lang: str, code: str, dry_run: bool = False) -> dict:
    """Clean a single language's parallel texts."""
    es_file = lang_dir / "data.es"
    maya_file = lang_dir / f"data.{code}"
    
    with open(es_file, 'r', encoding='utf-8') as f:
        es_lines = f.readlines()
    with open(maya_file, 'r', encoding='utf-8') as f:
        maya_lines = f.readlines()
    
    assert len(es_lines) == len(maya_lines), f"{lang}: line count mismatch ({len(es_lines)} vs {len(maya_lines)})"
    
    original_count = len(es_lines)
    
    # Step 1: Remove pairs where Maya line is empty
    clean_pairs = []
    removed_empty = 0
    for es, maya in zip(es_lines, maya_lines):
        if maya.strip() == '':
            removed_empty += 1
            continue
        clean_pairs.append((es, maya))
    
    # Step 2: Remove duplicate ES lines (keep first occurrence)
    seen_es = set()
    deduped_pairs = []
    removed_dupes = 0
    for es, maya in clean_pairs:
        es_key = es.strip().lower()
        if es_key in seen_es:
            removed_dupes += 1
            continue
        seen_es.add(es_key)
        deduped_pairs.append((es, maya))
    
    final_count = len(deduped_pairs)
    
    stats = {
        'language': lang,
        'code': code,
        'original': original_count,
        'removed_empty': removed_empty,
        'removed_dupes': removed_dupes,
        'final': final_count,
    }
    
    if not dry_run:
        with open(es_file, 'w', encoding='utf-8') as f:
            for es, _ in deduped_pairs:
                f.write(es if es.endswith('\n') else es + '\n')
        with open(maya_file, 'w', encoding='utf-8') as f:
            for _, maya in deduped_pairs:
                f.write(maya if maya.endswith('\n') else maya + '\n')
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean parallel text datasets")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to textos-paralelos/')
    parser.add_argument('--dry-run', action='store_true', help='Only report, do not modify files')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print(f"{'Language':<15} {'Original':<10} {'Empty rm':<10} {'Dupes rm':<10} {'Final':<10}")
    print("-" * 55)
    
    total_removed = 0
    for lang, code in sorted(LANG_CODES.items()):
        lang_dir = data_dir / lang
        if not lang_dir.exists():
            print(f"{lang:<15} MISSING")
            continue
        
        stats = clean_language(lang_dir, lang, code, dry_run=args.dry_run)
        removed = stats['removed_empty'] + stats['removed_dupes']
        total_removed += removed
        flag = " ✓ cleaned" if removed > 0 and not args.dry_run else ""
        print(f"{stats['language']:<15} {stats['original']:<10} {stats['removed_empty']:<10} {stats['removed_dupes']:<10} {stats['final']:<10}{flag}")
    
    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"\n[{mode}] Total pairs removed: {total_removed}")


if __name__ == '__main__':
    main()
