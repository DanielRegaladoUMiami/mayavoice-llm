#!/usr/bin/env python3
"""
Issue #2: Crear train/val/test split estratificado por idioma.

Reads cleaned parallel texts, converts to Alpaca format with instruction
templates (as defined in Issue #3), and creates stratified splits.

Output: train.jsonl, val.jsonl, test.jsonl in Alpaca format.
"""
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

LANG_CODES = {
    'achi': ('acr', 'Achi'),
    'awakateko': ('agu', 'Awakateko'),
    'chuj': ('cac', 'Chuj'),
    'itza': ('itz', "Itza'"),
    'kiche': ('quc', "K'iche'"),
    'mam': ('mam', 'Mam'),
    'poqomam': ('poc', 'Poqomam'),
    'poqomchi': ('poh', "Poqomchi'"),
    'qanjobal': ('kjb', "Q'anjob'al"),
    'qeqchi': ('kek', "Q'eqchi'"),
    'sipakapense': ('qum', 'Sipakapense'),
    'tektiteko': ('ttc', 'Tektiteko'),
    'tzutujil': ('tzj', "Tz'utujil"),
}

INSTRUCTION_TEMPLATES = {
    'translate_es_to_maya': [
        "Traduce del español al {lang}.",
        "Traduce la siguiente oración al {lang}.",
        "Convierte este texto del español al {lang}.",
    ],
    'translate_maya_to_es': [
        "Traduce del {lang} al español.",
        "Traduce la siguiente oración en {lang} al español.",
        "Convierte este texto del {lang} al español.",
    ],
}


def load_parallel_texts(data_dir: Path) -> list[dict]:
    """Load all cleaned parallel texts and convert to Alpaca format."""
    all_examples = []

    for lang_folder, (code, display_name) in sorted(LANG_CODES.items()):
        lang_dir = data_dir / lang_folder
        es_file = lang_dir / "data.es"
        maya_file = lang_dir / f"data.{code}"

        if not lang_dir.exists():
            print(f"  SKIP {lang_folder}: directory not found")
            continue

        with open(es_file, 'r', encoding='utf-8') as f:
            es_lines = [l.strip() for l in f.readlines()]
        with open(maya_file, 'r', encoding='utf-8') as f:
            maya_lines = [l.strip() for l in f.readlines()]

        assert len(es_lines) == len(maya_lines), f"{lang_folder}: line mismatch"

        for es, maya in zip(es_lines, maya_lines):
            if not es or not maya:
                continue

            # ES -> Maya direction
            instruction = random.choice(INSTRUCTION_TEMPLATES['translate_es_to_maya']).format(lang=display_name)
            all_examples.append({
                'instruction': instruction,
                'input': es,
                'output': maya,
                'lang': lang_folder,
                'direction': 'es_to_maya',
            })

            # Maya -> ES direction
            instruction = random.choice(INSTRUCTION_TEMPLATES['translate_maya_to_es']).format(lang=display_name)
            all_examples.append({
                'instruction': instruction,
                'input': maya,
                'output': es,
                'lang': lang_folder,
                'direction': 'maya_to_es',
            })

    return all_examples


def stratified_split(examples: list[dict], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split examples ensuring proportional representation of each language."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)

    # Group by (lang, direction) to stratify
    groups = defaultdict(list)
    for ex in examples:
        groups[(ex['lang'], ex['direction'])].append(ex)

    train, val, test = [], [], []

    for key, group_examples in groups.items():
        random.shuffle(group_examples)
        n = len(group_examples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(group_examples[:n_train])
        val.extend(group_examples[n_train:n_train + n_val])
        test.extend(group_examples[n_train + n_val:])

    # Final shuffle
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_jsonl(examples: list[dict], path: Path, include_meta: bool = False):
    """Save examples to JSONL in Alpaca format."""
    with open(path, 'w', encoding='utf-8') as f:
        for ex in examples:
            record = {
                'instruction': ex['instruction'],
                'input': ex['input'],
                'output': ex['output'],
            }
            if include_meta:
                record['_lang'] = ex['lang']
                record['_direction'] = ex['direction']
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits in Alpaca format")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to textos-paralelos/')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for splits')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--include-meta', action='store_true', help='Include _lang and _direction metadata')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and converting parallel texts...")
    examples = load_parallel_texts(data_dir)
    print(f"  Total examples (both directions): {len(examples)}")

    # Show per-language stats
    lang_counts = defaultdict(int)
    for ex in examples:
        lang_counts[ex['lang']] += 1
    print(f"\n{'Language':<15} {'Pairs (both dirs)':<20}")
    print("-" * 35)
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang:<15} {count:<20}")

    print(f"\nCreating 80/10/10 stratified split (seed={args.seed})...")
    train, val, test = stratified_split(examples, seed=args.seed)

    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")
    print(f"  Total: {len(train) + len(val) + len(test)}")

    # Verify stratification
    print(f"\nPer-language distribution in test set:")
    test_langs = defaultdict(int)
    for ex in test:
        test_langs[ex['lang']] += 1
    for lang, count in sorted(test_langs.items()):
        print(f"  {lang:<15} {count}")

    # Save
    save_jsonl(train, output_dir / "train.jsonl", include_meta=args.include_meta)
    save_jsonl(val, output_dir / "val.jsonl", include_meta=args.include_meta)
    save_jsonl(test, output_dir / "test.jsonl", include_meta=args.include_meta)

    # Save test with metadata for evaluation scripts
    save_jsonl(test, output_dir / "test_with_meta.jsonl", include_meta=True)

    # Save split stats
    stats = {
        'seed': args.seed,
        'total_examples': len(examples),
        'train': len(train),
        'val': len(val),
        'test': len(test),
        'languages': {lang: count for lang, count in sorted(lang_counts.items())},
        'format': 'alpaca',
        'directions': ['es_to_maya', 'maya_to_es'],
    }
    with open(output_dir / "split_stats.json", 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nFiles saved to {output_dir}/")
    print("  train.jsonl, val.jsonl, test.jsonl, test_with_meta.jsonl, split_stats.json")


if __name__ == '__main__':
    main()
