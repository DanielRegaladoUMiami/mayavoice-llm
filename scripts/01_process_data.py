#!/usr/bin/env python3
"""
01_process_data.py
==================
Procesa textos paralelos y diccionarios para crear datasets de fine-tuning.

Funciones principales:
- Carga textos paralelos español-maya
- Genera pares bidireccionales (es->maya y maya->es)
- Agrega datos de diccionarios
- Crea splits train/val (90/10)
- Exporta a JSONL en formato Alpaca

Uso:
    python 01_process_data.py --source ../mayan-data-organized --output ../data/processed
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict

# Mapeo de códigos ISO a nombres de lenguas mayas
LANGUAGE_CODES = {
    'agu': 'Awakateko',
    'cac': 'Chuj',
    'itz': 'Itza',
    'quc': 'Kiche',
    'mam': 'Mam',
    'poc': 'Poqomam',
    'poh': 'Poqomchi',
    'kjb': 'Qanjobal',
    'kek': 'Qeqchi',
    'qum': 'Sipakapense',
    'ttc': 'Tektiteko',
    'tzj': 'Tzutujil'
}


def load_parallel_texts(base_path: Path) -> List[Dict]:
    """Carga archivos de textos paralelos español-maya."""
    examples = []
    parallel_dir = base_path / "01_parallel_texts"

    if not parallel_dir.exists():
        print(f"⚠️  Directorio no encontrado: {parallel_dir}")
        return examples

    stats = defaultdict(int)

    for lang_dir in sorted(parallel_dir.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name == 'unknown':
            continue

        es_file = lang_dir / "data.es"
        maya_files = list(lang_dir.glob("data.*"))
        maya_file = next((f for f in maya_files if f.suffix != '.es'), None)

        if not es_file.exists() or not maya_file:
            continue

        lang_code = maya_file.suffix[1:]  # Quita el punto
        lang_name = LANGUAGE_CODES.get(lang_code, lang_dir.name.title())

        try:
            with open(es_file, 'r', encoding='utf-8') as f:
                es_lines = [line.strip() for line in f if line.strip()]

            with open(maya_file, 'r', encoding='utf-8') as f:
                maya_lines = [line.strip() for line in f if line.strip()]

            if len(es_lines) != len(maya_lines):
                print(f"⚠️  {lang_name}: Desbalance ({len(es_lines)} es vs {len(maya_lines)} maya)")

            # Generar pares bidireccionales
            for es, maya in zip(es_lines, maya_lines):
                if len(es) < 5 or len(maya) < 3:  # Filtrar líneas muy cortas
                    continue

                # Español -> Maya
                examples.append({
                    "instruction": f"Translate the following text from Spanish to {lang_name}.",
                    "input": es,
                    "output": maya,
                    "language": lang_code,
                    "direction": "es->maya"
                })

                # Maya -> Español
                examples.append({
                    "instruction": f"Translate the following text from {lang_name} to Spanish.",
                    "input": maya,
                    "output": es,
                    "language": lang_code,
                    "direction": "maya->es"
                })

                stats[lang_name] += 2

        except Exception as e:
            print(f"❌ Error procesando {lang_name}: {e}")

    print(f"\n✅ Textos paralelos cargados:")
    for lang, count in sorted(stats.items()):
        print(f"   {lang}: {count:,} ejemplos")

    return examples


def load_dictionaries(base_path: Path) -> List[Dict]:
    """Carga diccionarios CSV y genera ejemplos palabra-nivel."""
    examples = []
    dict_dir = base_path / "02_dictionaries" / "csv"

    if not dict_dir.exists():
        print(f"⚠️  Directorio de diccionarios no encontrado: {dict_dir}")
        return examples

    # Procesar diccionario principal
    main_dict = dict_dir / "mayavoice_dictionary_full.csv"
    if main_dict.exists():
        print(f"\n📖 Procesando diccionario principal...")
        try:
            with open(main_dict, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # Skip header

            stats = defaultdict(int)
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue

                maya_word = parts[0].strip()
                es_word = parts[1].strip()
                lang_code = parts[2].strip() if len(parts) > 2 else 'unknown'

                if not maya_word or not es_word or len(maya_word) < 2:
                    continue

                lang_name = LANGUAGE_CODES.get(lang_code, lang_code.title())

                # Español -> Maya (palabra)
                examples.append({
                    "instruction": f"Translate the following word from Spanish to {lang_name}.",
                    "input": es_word,
                    "output": maya_word,
                    "language": lang_code,
                    "direction": "es->maya",
                    "type": "dictionary"
                })

                # Maya -> Español (palabra)
                examples.append({
                    "instruction": f"Translate the following word from {lang_name} to Spanish.",
                    "input": maya_word,
                    "output": es_word,
                    "language": lang_code,
                    "direction": "maya->es",
                    "type": "dictionary"
                })

                stats[lang_name] += 2

            print(f"✅ Diccionario procesado: {len(examples):,} ejemplos")
            for lang, count in sorted(stats.items())[:10]:
                print(f"   {lang}: {count:,} palabras")

        except Exception as e:
            print(f"❌ Error procesando diccionario: {e}")

    return examples


def create_splits(examples: List[Dict], val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
    """Divide dataset en train/val manteniendo balance por idioma."""
    random.seed(42)

    # Agrupar por idioma
    by_language = defaultdict(list)
    for ex in examples:
        by_language[ex['language']].append(ex)

    train_data = []
    val_data = []

    for lang, lang_examples in by_language.items():
        random.shuffle(lang_examples)
        split_idx = int(len(lang_examples) * (1 - val_ratio))

        train_data.extend(lang_examples[:split_idx])
        val_data.extend(lang_examples[split_idx:])

    random.shuffle(train_data)
    random.shuffle(val_data)

    return train_data, val_data


def save_jsonl(data: List[Dict], output_path: Path, include_metadata: bool = False):
    """Guarda datos en formato JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            if not include_metadata:
                # Solo campos necesarios para fine-tuning
                output_item = {
                    "instruction": item["instruction"],
                    "input": item["input"],
                    "output": item["output"]
                }
            else:
                output_item = item

            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    print(f"💾 Guardado: {output_path} ({len(data):,} ejemplos)")


def generate_metadata(train_data: List[Dict], val_data: List[Dict]) -> Dict:
    """Genera estadísticas del dataset."""
    stats = {
        "total_examples": len(train_data) + len(val_data),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "languages": {},
        "directions": {
            "es->maya": 0,
            "maya->es": 0
        },
        "types": {
            "parallel": 0,
            "dictionary": 0
        }
    }

    all_data = train_data + val_data

    for item in all_data:
        lang = item.get('language', 'unknown')
        if lang not in stats["languages"]:
            stats["languages"][lang] = {"count": 0, "name": LANGUAGE_CODES.get(lang, lang)}
        stats["languages"][lang]["count"] += 1

        direction = item.get('direction', 'unknown')
        if direction in stats["directions"]:
            stats["directions"][direction] += 1

        item_type = item.get('type', 'parallel')
        if item_type in stats["types"]:
            stats["types"][item_type] += 1

    # Calcular promedios de longitud
    avg_input = sum(len(item['input']) for item in all_data) / len(all_data)
    avg_output = sum(len(item['output']) for item in all_data) / len(all_data)

    stats["avg_lengths"] = {
        "input_chars": round(avg_input, 1),
        "output_chars": round(avg_output, 1)
    }

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Procesa datos para fine-tuning de lenguas mayas')
    parser.add_argument('--source', type=str, default='../mayan-data-organized',
                        help='Directorio con datos originales')
    parser.add_argument('--output', type=str, default='../data/processed',
                        help='Directorio de salida')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Proporción para validación (default: 0.1)')
    parser.add_argument('--include-dicts', action='store_true',
                        help='Incluir datos de diccionarios')

    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)

    print("=" * 60)
    print("🌎 PROCESADOR DE DATOS - LENGUAS MAYAS")
    print("=" * 60)

    # Cargar datos
    print("\n📂 Cargando textos paralelos...")
    parallel_data = load_parallel_texts(source_path)

    all_data = parallel_data.copy()

    if args.include_dicts:
        print("\n📖 Cargando diccionarios...")
        dict_data = load_dictionaries(source_path)
        all_data.extend(dict_data)

    if not all_data:
        print("\n❌ No se encontraron datos para procesar.")
        sys.exit(1)

    print(f"\n📊 Total de ejemplos: {len(all_data):,}")

    # Crear splits
    print(f"\n✂️  Creando splits (train={1-args.val_ratio:.0%}, val={args.val_ratio:.0%})...")
    train_data, val_data = create_splits(all_data, args.val_ratio)

    # Guardar archivos
    print("\n💾 Guardando archivos...")
    save_jsonl(train_data, output_path / "train.jsonl")
    save_jsonl(val_data, output_path / "val.jsonl")

    # Generar metadata
    metadata = generate_metadata(train_data, val_data)
    with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"💾 Guardado: {output_path / 'metadata.json'}")

    # Resumen final
    print("\n" + "=" * 60)
    print("✅ PROCESAMIENTO COMPLETO")
    print("=" * 60)
    print(f"\n📊 Estadísticas finales:")
    print(f"   Total: {metadata['total_examples']:,} ejemplos")
    print(f"   Train: {metadata['train_examples']:,} ejemplos")
    print(f"   Val: {metadata['val_examples']:,} ejemplos")
    print(f"\n🌍 Idiomas: {len(metadata['languages'])}")
    for lang, info in sorted(metadata['languages'].items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
        print(f"   {info['name']}: {info['count']:,} ejemplos")

    print(f"\n📁 Archivos generados en: {output_path.resolve()}")
    print("\n🚀 Siguiente paso: python 02_train_qlora.py")


if __name__ == "__main__":
    main()
