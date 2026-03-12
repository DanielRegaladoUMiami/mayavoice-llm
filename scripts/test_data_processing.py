#!/usr/bin/env python3
"""
Script de prueba para verificar que el procesamiento de datos funciona.
Ejecuta la lógica sin Kubeflow para debugging.
"""

import json
import random
from collections import defaultdict
from google.cloud import storage

PROJECT_ID = "mayan-gptai"
BUCKET_NAME = "mayan-llm-mayan-gptai"
VAL_RATIO = 0.1

print("📂 Iniciando procesamiento de datos (modo prueba)...")
print()

try:
    # Cliente de Storage
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    # Cargar training_data_v2.jsonl
    examples = []
    print("📥 Cargando training_data_v2.jsonl desde GCS...")
    
    blob = bucket.blob("data/raw/training_data_v2.jsonl")
    if blob.exists():
        print("✅ Archivo encontrado")
        content = blob.download_as_text()
        lines = content.strip().split('\n')
        
        for i, line in enumerate(lines):
            if line.strip():
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Error en línea {i}: {e}")
                    continue
    else:
        print("❌ training_data_v2.jsonl no encontrado")
        raise FileNotFoundError(f"No se encontró data/raw/training_data_v2.jsonl en {BUCKET_NAME}")
    
    print(f"📊 Total de ejemplos cargados: {len(examples):,}")
    
    if len(examples) == 0:
        raise ValueError("No hay ejemplos para procesar")
    
    # Crear splits estratificados por idioma
    print("✂️  Dividiendo datos en train/val estratificados...")
    random.seed(42)
    by_language = defaultdict(list)
    
    for ex in examples:
        lang = ex.get('language', 'unknown')
        by_language[lang].append(ex)
    
    print(f"🌐 Idiomas encontrados: {len(by_language)}")
    
    train_data = []
    val_data = []
    
    for lang, lang_examples in by_language.items():
        random.shuffle(lang_examples)
        n = len(lang_examples)
        train_end = int(n * (1 - VAL_RATIO))
        
        train_data.extend(lang_examples[:train_end])
        val_data.extend(lang_examples[train_end:])
        print(f"  {lang}: {train_end} train, {n-train_end} val")
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print(f"\n📊 Split final:")
    print(f"   Train: {len(train_data):,}")
    print(f"   Val: {len(val_data):,}")
    
    # Guardar a GCS
    print("\n💾 Guardando archivos a GCS...")
    
    # Escribir train
    train_blob = bucket.blob("data/processed/train.jsonl")
    train_content = '\n'.join([json.dumps(ex, ensure_ascii=False) for ex in train_data])
    train_blob.upload_from_string(train_content, content_type='application/x-ndjson')
    print(f"✅ train.jsonl guardado ({len(train_data):,} ejemplos, {len(train_content) / (1024*1024):.2f} MB)")
    
    # Escribir val
    val_blob = bucket.blob("data/processed/val.jsonl")
    val_content = '\n'.join([json.dumps(ex, ensure_ascii=False) for ex in val_data])
    val_blob.upload_from_string(val_content, content_type='application/x-ndjson')
    print(f"✅ val.jsonl guardado ({len(val_data):,} ejemplos, {len(val_content) / (1024*1024):.2f} MB)")
    
    print("\n✅ Procesamiento completado exitosamente")
    
except Exception as e:
    print(f"\n❌ Error durante procesamiento: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)
