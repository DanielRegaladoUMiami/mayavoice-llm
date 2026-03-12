#!/usr/bin/env python3
"""
00_analyze_dataset.py
=====================
Analiza dataset existente y genera reporte de calidad.

Uso:
    python 00_analyze_dataset.py --data ../data/processed/training_data_v2.jsonl
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List

def load_jsonl(file_path: Path) -> List[Dict]:
    """Carga archivo JSONL."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def analyze_dataset(data: List[Dict]):
    """Genera análisis detallado del dataset."""

    print("=" * 70)
    print("📊 ANÁLISIS DEL DATASET")
    print("=" * 70)

    total = len(data)
    print(f"\n✅ Total de ejemplos: {total:,}")

    # Análisis por idioma
    languages = Counter()
    directions = Counter()
    types = Counter()

    input_lengths = []
    output_lengths = []

    for item in data:
        # Extraer idioma del instruction
        instruction = item.get('instruction', '')
        if 'Spanish to' in instruction:
            lang = instruction.split('Spanish to ')[-1].rstrip('.')
            direction = 'es->maya'
        elif 'to Spanish' in instruction:
            lang = instruction.split(' to Spanish')[0].split('from ')[-1]
            direction = 'maya->es'
        else:
            lang = 'unknown'
            direction = 'unknown'

        languages[lang] += 1
        directions[direction] += 1

        # Tipo (palabra vs oración)
        input_text = item.get('input', '')
        if ' ' in input_text.strip():
            types['sentence'] += 1
        else:
            types['word'] += 1

        input_lengths.append(len(input_text))
        output_lengths.append(len(item.get('output', '')))

    # Reportes
    print("\n🌍 Distribución por idioma:")
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"   {lang:20s}: {count:6,} ejemplos ({percentage:5.2f}%)")

    print(f"\n🔄 Distribución por dirección:")
    for dir, count in directions.items():
        percentage = (count / total) * 100
        print(f"   {dir:15s}: {count:6,} ejemplos ({percentage:5.2f}%)")

    print(f"\n📝 Tipo de ejemplo:")
    for type_name, count in types.items():
        percentage = (count / total) * 100
        print(f"   {type_name:15s}: {count:6,} ejemplos ({percentage:5.2f}%)")

    # Estadísticas de longitud
    avg_input = sum(input_lengths) / len(input_lengths)
    avg_output = sum(output_lengths) / len(output_lengths)

    print(f"\n📏 Longitud promedio (caracteres):")
    print(f"   Input:  {avg_input:.1f} chars (min: {min(input_lengths)}, max: {max(input_lengths)})")
    print(f"   Output: {avg_output:.1f} chars (min: {min(output_lengths)}, max: {max(output_lengths)})")

    # Verificar problemas
    print("\n⚠️  Verificación de problemas:")

    empty_inputs = sum(1 for item in data if not item.get('input', '').strip())
    empty_outputs = sum(1 for item in data if not item.get('output', '').strip())
    very_short = sum(1 for item in data if len(item.get('input', '')) < 3 or len(item.get('output', '')) < 2)
    duplicates = total - len(set(item.get('input', '') for item in data))

    print(f"   Inputs vacíos: {empty_inputs}")
    print(f"   Outputs vacíos: {empty_outputs}")
    print(f"   Muy cortos (<3 chars): {very_short}")
    print(f"   Duplicados (input): {duplicates}")

    if empty_inputs + empty_outputs + very_short == 0:
        print("\n   ✅ No se encontraron problemas mayores")
    else:
        print("\n   ⚠️  Se recomienda limpiar el dataset")

    # Ejemplos
    print("\n📄 Ejemplos del dataset:")
    for i, item in enumerate(data[:3], 1):
        print(f"\n   Ejemplo {i}:")
        print(f"   Instruction: {item.get('instruction', 'N/A')[:80]}...")
        print(f"   Input: {item.get('input', 'N/A')[:60]}")
        print(f"   Output: {item.get('output', 'N/A')[:60]}")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analiza dataset de entrenamiento')
    parser.add_argument('--data', type=str, required=True,
                        help='Ruta al archivo JSONL')

    args = parser.parse_args()

    data_path = Path(args.data)

    if not data_path.exists():
        print(f"❌ Error: No se encontró {data_path}")
        sys.exit(1)

    print(f"\n📂 Cargando: {data_path}")
    data = load_jsonl(data_path)

    analyze_dataset(data)


if __name__ == "__main__":
    main()
