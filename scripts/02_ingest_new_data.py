#!/usr/bin/env python3
"""
Issue #4: Pipeline de ingestión de nuevos datos.

Supports multiple input formats and validates data before adding
to the parallel texts corpus. After ingestion, regenerate splits
with 01_create_splits.py.

Usage:
    # Ingest a CSV file (columns: es, maya)
    python 02_ingest_new_data.py --format csv --lang kaqchikel --code cak \
        --input new_data.csv --data-dir data/textos-paralelos/

    # Ingest a TSV file
    python 02_ingest_new_data.py --format tsv --lang kiche --code quc \
        --input new_kiche.tsv --data-dir data/textos-paralelos/

    # Ingest a JSONL file (fields: es, maya or source, target)
    python 02_ingest_new_data.py --format jsonl --lang mam --code mam \
        --input new_mam.jsonl --data-dir data/textos-paralelos/

    # Ingest plain text pair files
    python 02_ingest_new_data.py --format parallel --lang achi --code acr \
        --input-es achi_es.txt --input-maya achi_maya.txt \
        --data-dir data/textos-paralelos/
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path


def validate_pair(es: str, maya: str, line_num: int) -> tuple[bool, str]:
    """Validate a single translation pair."""
    if not es.strip():
        return False, f"Line {line_num}: empty ES text"
    if not maya.strip():
        return False, f"Line {line_num}: empty Maya text"
    if len(es.strip()) < 2:
        return False, f"Line {line_num}: ES text too short ({es.strip()!r})"
    if len(maya.strip()) < 2:
        return False, f"Line {line_num}: Maya text too short ({maya.strip()!r})"
    # Check for leftover tags
    if re.match(r'^#\w+#', es.strip()):
        return False, f"Line {line_num}: ES has language tag prefix"
    return True, ""


def load_csv(path: Path, delimiter=',') -> list[tuple[str, str]]:
    """Load pairs from CSV/TSV. Expects columns: es, maya (or source, target)."""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = reader.fieldnames

        # Auto-detect column names
        es_col = next((c for c in fields if c.lower() in ('es', 'spanish', 'source', 'español')), None)
        maya_col = next((c for c in fields if c.lower() in ('maya', 'target', 'translation', 'output')), None)

        if not es_col or not maya_col:
            print(f"ERROR: Could not find ES/Maya columns. Found: {fields}")
            print("Expected: 'es'/'source' and 'maya'/'target'")
            sys.exit(1)

        print(f"  Using columns: ES='{es_col}', Maya='{maya_col}'")

        for i, row in enumerate(reader, 1):
            pairs.append((row[es_col].strip(), row[maya_col].strip()))

    return pairs


def load_jsonl(path: Path) -> list[tuple[str, str]]:
    """Load pairs from JSONL. Expects fields: es/source and maya/target."""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)
            es = data.get('es') or data.get('source') or data.get('input', '')
            maya = data.get('maya') or data.get('target') or data.get('output', '')
            pairs.append((es.strip(), maya.strip()))
    return pairs


def load_parallel(es_path: Path, maya_path: Path) -> list[tuple[str, str]]:
    """Load from two parallel text files."""
    with open(es_path, 'r', encoding='utf-8') as f:
        es_lines = [l.strip() for l in f.readlines()]
    with open(maya_path, 'r', encoding='utf-8') as f:
        maya_lines = [l.strip() for l in f.readlines()]

    if len(es_lines) != len(maya_lines):
        print(f"WARNING: Line count mismatch: ES={len(es_lines)}, Maya={len(maya_lines)}")
        min_len = min(len(es_lines), len(maya_lines))
        es_lines = es_lines[:min_len]
        maya_lines = maya_lines[:min_len]

    return list(zip(es_lines, maya_lines))


def deduplicate_against_existing(new_pairs: list[tuple[str, str]],
                                  existing_es: set[str]) -> list[tuple[str, str]]:
    """Remove pairs whose ES text already exists in the corpus."""
    deduped = []
    dupes = 0
    for es, maya in new_pairs:
        if es.strip().lower() in existing_es:
            dupes += 1
            continue
        deduped.append((es, maya))
    if dupes:
        print(f"  Removed {dupes} duplicates (ES already in corpus)")
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Ingest new parallel text data")
    parser.add_argument('--format', choices=['csv', 'tsv', 'jsonl', 'parallel'], required=True)
    parser.add_argument('--lang', type=str, required=True, help='Language folder name (e.g., kaqchikel)')
    parser.add_argument('--code', type=str, required=True, help='ISO 639-3 code (e.g., cak)')
    parser.add_argument('--input', type=str, help='Input file (csv/tsv/jsonl)')
    parser.add_argument('--input-es', type=str, help='ES parallel file (for --format parallel)')
    parser.add_argument('--input-maya', type=str, help='Maya parallel file (for --format parallel)')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to textos-paralelos/')
    parser.add_argument('--dry-run', action='store_true', help='Validate only, do not write')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    lang_dir = data_dir / args.lang
    es_file = lang_dir / "data.es"
    maya_file = lang_dir / f"data.{args.code}"

    # Load new data
    print(f"Loading new data ({args.format})...")
    if args.format == 'csv':
        new_pairs = load_csv(Path(args.input))
    elif args.format == 'tsv':
        new_pairs = load_csv(Path(args.input), delimiter='\t')
    elif args.format == 'jsonl':
        new_pairs = load_jsonl(Path(args.input))
    elif args.format == 'parallel':
        new_pairs = load_parallel(Path(args.input_es), Path(args.input_maya))

    print(f"  Loaded {len(new_pairs)} pairs")

    # Validate
    print("Validating...")
    valid_pairs = []
    errors = []
    for i, (es, maya) in enumerate(new_pairs, 1):
        ok, err = validate_pair(es, maya, i)
        if ok:
            valid_pairs.append((es, maya))
        else:
            errors.append(err)

    if errors:
        print(f"  {len(errors)} validation errors:")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    print(f"  {len(valid_pairs)} valid pairs")

    # Deduplicate against existing data
    if lang_dir.exists() and es_file.exists():
        with open(es_file, 'r', encoding='utf-8') as f:
            existing_es = {l.strip().lower() for l in f.readlines()}
        print(f"  Existing corpus: {len(existing_es)} pairs")
        valid_pairs = deduplicate_against_existing(valid_pairs, existing_es)
    else:
        print(f"  New language — creating directory: {lang_dir}")
        if not args.dry_run:
            lang_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Final new pairs to add: {len(valid_pairs)}")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        return

    if len(valid_pairs) == 0:
        print("Nothing to add.")
        return

    # Append to files
    with open(es_file, 'a', encoding='utf-8') as f:
        for es, _ in valid_pairs:
            f.write(es + '\n')
    with open(maya_file, 'a', encoding='utf-8') as f:
        for _, maya in valid_pairs:
            f.write(maya + '\n')

    # Verify alignment
    with open(es_file) as f:
        es_count = len(f.readlines())
    with open(maya_file) as f:
        maya_count = len(f.readlines())

    assert es_count == maya_count, f"ALIGNMENT ERROR: {es_count} vs {maya_count}"
    print(f"\n✓ Successfully added {len(valid_pairs)} pairs to {args.lang}")
    print(f"  Total corpus size: {es_count} pairs")
    print(f"\nNext step: regenerate splits with 01_create_splits.py")


if __name__ == '__main__':
    main()
