#!/usr/bin/env python3
"""
03_merge_model.py
=================
Fusiona adaptadores LoRA con el modelo base para deployment.

Uso:
    python 03_merge_model.py --adapter-path ../models/run-xxx/final_model --output ../models/merged
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_adapter(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    device: str = "auto"
):
    """Fusiona adaptadores LoRA con modelo base."""

    print("=" * 70)
    print("🔗 FUSIONANDO MODELO LORA")
    print("=" * 70)

    print(f"\n📦 Cargando modelo base: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    print(f"🔌 Cargando adaptadores: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("🔀 Fusionando adaptadores con modelo base...")
    model = model.merge_and_unload()

    print(f"\n💾 Guardando modelo fusionado en: {output_path}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)

    # Guardar tokenizer
    print("💾 Guardando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    print("\n" + "=" * 70)
    print("✅ FUSIÓN COMPLETA")
    print("=" * 70)
    print(f"\n📦 Modelo listo para deployment en: {output_path.resolve()}")
    print("\n🚀 Siguiente paso: python 04_inference.py")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fusionar adaptadores LoRA con modelo base')
    parser.add_argument('--adapter-path', type=str, required=True,
                        help='Ruta a los adaptadores LoRA entrenados')
    parser.add_argument('--base-model', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                        help='Nombre del modelo base')
    parser.add_argument('--output', type=str, default='../models/merged',
                        help='Directorio de salida')

    args = parser.parse_args()

    merge_lora_adapter(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
