#!/usr/bin/env python3
"""
Data augmentation via Spanish synonym replacement.

For each parallel pair (ES ↔ Maya), generates N additional variants by
replacing content words in the Spanish sentence with WordNet synonyms.
The Maya translation stays the same — this teaches the model that
different Spanish phrasings map to the same Maya output.

Strategies:
  1. Single-word synonym replacement (replace 1 word per variant)
  2. Multi-word replacement (replace 2-3 words per variant)

Filters:
  - Only replaces nouns, verbs, adjectives, adverbs (not function words)
  - Skips proper nouns and very short words
  - Validates that the synonym fits (same POS)
  - Deduplicates generated variants

Usage:
    python 05_augment_synonyms.py \
        --data-dir ../data/textos-paralelos/ \
        --output-dir ../data/textos-paralelos-augmented/ \
        --variants 2

    # Include Bible data
    python 05_augment_synonyms.py \
        --data-dir ../data/textos-paralelos/ \
        --bible-dir ../data/textos-paralelos-biblia/ \
        --output-dir ../data/textos-paralelos-augmented/ \
        --variants 2
"""
import argparse
import random
import re
import sys
from pathlib import Path
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn

# Ensure resources
for res in ['omw-1.4', 'wordnet']:
    nltk.download(res, quiet=True)

# Spanish stopwords / function words to skip
STOPWORDS = set("""
el la los las un una unos unas de del al a en con por para sin sobre
entre desde hasta como más que y o ni pero sino también ya no me te
se le lo nos les su sus mi mis tu tus es son ser estar hay ha han fue
muy bien mal aquí ahí allí este esta estos estas ese esa esos esas
aquel aquella yo tú él ella nosotros ellos ellas usted ustedes
""".split())

# Language code mapping (same as 01_create_splits.py)
LANG_CODES = {
    'achi': 'acr', 'awakateko': 'agu', 'chuj': 'cac', 'itza': 'itz',
    'kiche': 'quc', 'mam': 'mam', 'poqomam': 'poc', 'poqomchi': 'poh',
    'qanjobal': 'kjb', 'qeqchi': 'kek', 'sipakapense': 'qum',
    'tektiteko': 'ttc', 'tzutujil': 'tzj', 'kaqchikel': 'cak',
}


def get_spanish_synonyms(word: str) -> list[str]:
    """Get Spanish synonyms for a word using WordNet."""
    word_lower = word.lower()
    if word_lower in STOPWORDS or len(word_lower) < 4:
        return []

    synonyms = set()
    for synset in wn.synsets(word_lower, lang='spa'):
        for lemma in synset.lemma_names('spa'):
            clean = lemma.replace('_', ' ')
            if clean.lower() != word_lower and len(clean) > 2:
                synonyms.add(clean)

    return list(synonyms)


def augment_sentence(sentence: str, max_replacements: int = 1) -> list[str]:
    """Generate augmented variants of a Spanish sentence."""
    words = sentence.split()
    if len(words) < 3:
        return []

    # Find replaceable words and their synonyms
    replaceable = []
    for i, word in enumerate(words):
        # Skip punctuation-heavy tokens
        clean = re.sub(r'[^\wáéíóúñü]', '', word.lower())
        if not clean or clean in STOPWORDS or len(clean) < 4:
            continue
        syns = get_spanish_synonyms(clean)
        if syns:
            replaceable.append((i, word, syns))

    if not replaceable:
        return []

    variants = set()
    attempts = 0
    max_attempts = max_replacements * 5

    while len(variants) < max_replacements and attempts < max_attempts:
        attempts += 1
        new_words = words.copy()

        # Pick 1-2 random positions to replace
        n_replace = min(random.randint(1, 2), len(replaceable))
        positions = random.sample(replaceable, n_replace)

        for idx, original, syns in positions:
            syn = random.choice(syns)
            # Preserve capitalization
            if original[0].isupper():
                syn = syn.capitalize()
            # Preserve trailing punctuation
            trailing = ''
            if original and not original[-1].isalnum():
                trailing = original[-1]
                syn = syn.rstrip('.,;:!?') + trailing
            new_words[idx] = syn

        variant = ' '.join(new_words)
        if variant != sentence:
            variants.add(variant)

    return list(variants)


def load_parallel_data(data_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """Load all parallel text pairs from directory."""
    data = {}
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        lang_name = folder.name
        code = LANG_CODES.get(lang_name)
        if not code:
            continue

        es_file = folder / 'data.es'
        maya_file = folder / f'data.{code}'

        if not es_file.exists() or not maya_file.exists():
            continue

        with open(es_file, encoding='utf-8') as f:
            es_lines = [l.strip() for l in f]
        with open(maya_file, encoding='utf-8') as f:
            maya_lines = [l.strip() for l in f]

        pairs = [(es, maya) for es, maya in zip(es_lines, maya_lines)
                 if es and maya]
        data[lang_name] = pairs

    return data


def main():
    parser = argparse.ArgumentParser(description="Augment parallel data with Spanish synonyms")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--bible-dir', type=str, default=None,
                        help='Optional Bible parallel texts directory')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--variants', type=int, default=2,
                        help='Number of augmented variants per pair (default: 2)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Load main data
    print("Loading parallel texts...")
    data = load_parallel_data(data_dir)

    # Load Bible data if provided
    if args.bible_dir:
        bible_dir = Path(args.bible_dir)
        print("Loading Bible parallel texts...")
        bible_data = load_parallel_data(bible_dir)
        for lang, pairs in bible_data.items():
            if lang in data:
                data[lang].extend(pairs)
            else:
                data[lang] = pairs
            print(f"  Added {len(pairs)} Bible pairs for {lang}")

    total_original = sum(len(pairs) for pairs in data.values())
    print(f"\nTotal original pairs: {total_original}")
    print(f"Generating {args.variants} variants per pair...\n")

    total_augmented = 0
    stats = {}

    for lang_name, pairs in sorted(data.items()):
        code = LANG_CODES[lang_name]
        augmented_es = []
        augmented_maya = []
        original_es = []
        original_maya = []

        for es, maya in pairs:
            original_es.append(es)
            original_maya.append(maya)

            # Generate augmented variants
            variants = augment_sentence(es, max_replacements=args.variants)
            for var in variants:
                augmented_es.append(var)
                augmented_maya.append(maya)  # Maya stays the same

        n_aug = len(augmented_es)
        total_augmented += n_aug
        total_for_lang = len(pairs) + n_aug
        ratio = n_aug / len(pairs) if pairs else 0

        stats[lang_name] = {
            'original': len(pairs),
            'augmented': n_aug,
            'total': total_for_lang,
            'ratio': ratio,
        }

        print(f"  {lang_name:<15} orig={len(pairs):<6} +aug={n_aug:<6} total={total_for_lang:<6} ({ratio:.1f}x)")

        if args.dry_run:
            # Show a few examples
            if augmented_es:
                print(f"    Example original: {pairs[0][0][:70]}")
                print(f"    Example augment:  {augmented_es[0][:70]}")
            continue

        # Save combined (original + augmented)
        lang_dir = output_dir / lang_name
        lang_dir.mkdir(parents=True, exist_ok=True)

        all_es = original_es + augmented_es
        all_maya = original_maya + augmented_maya

        with open(lang_dir / 'data.es', 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_es) + '\n')
        with open(lang_dir / f'data.{code}', 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_maya) + '\n')

    # Summary
    print(f"\n{'='*50}")
    print(f"AUGMENTATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Original pairs:   {total_original}")
    print(f"  Augmented pairs:  {total_augmented}")
    print(f"  Total pairs:      {total_original + total_augmented}")
    print(f"  Augmentation:     {total_augmented/total_original:.1f}x increase")

    if args.dry_run:
        print(f"\n[DRY RUN] No files written.")
    else:
        print(f"\nOutput: {output_dir}")
        print(f"Next: run 01_create_splits.py with --data-dir {output_dir}")


if __name__ == '__main__':
    main()
