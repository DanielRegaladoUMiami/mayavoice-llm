#!/usr/bin/env python3
"""
04_inference.py
===============
Script de inferencia para traducción español-lenguas mayas.

Uso:
    # Con modelo fusionado
    python 04_inference.py --model ../models/merged --text "Hola mundo" --target kiche

    # Con adaptadores LoRA
    python 04_inference.py --adapter ../models/run-xxx/final_model --text "Hola mundo" --target kiche

    # Modo interactivo
    python 04_inference.py --model ../models/merged --interactive
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

LANGUAGES = {
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


def load_model(model_path: str = None, adapter_path: str = None, base_model: str = None):
    """Carga modelo fusionado o con adaptadores."""

    if model_path:
        print(f"📦 Cargando modelo fusionado: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    elif adapter_path and base_model:
        print(f"📦 Cargando modelo base: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        print(f"🔌 Cargando adaptadores: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    else:
        raise ValueError("Debe proporcionar --model o --adapter + --base-model")

    return model, tokenizer


def generate_translation(
    model,
    tokenizer,
    text: str,
    target_lang: str,
    source_lang: str = "Spanish",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """Genera traducción usando el modelo."""

    target_name = LANGUAGES.get(target_lang, target_lang.title())

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Translate the following text from {source_lang} to {target_name}.

### Input:
{text}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer solo la respuesta
    if "### Response:" in full_output:
        translation = full_output.split("### Response:")[-1].strip()
    else:
        translation = full_output

    return translation


def interactive_mode(model, tokenizer):
    """Modo interactivo de traducción."""

    print("\n" + "=" * 70)
    print("🌎 MODO INTERACTIVO - TRADUCTOR DE LENGUAS MAYAS")
    print("=" * 70)
    print("\nIdiomas disponibles:")
    for code, name in sorted(LANGUAGES.items()):
        print(f"  {code}: {name}")
    print("\nComandos especiales:")
    print("  'exit' - Salir")
    print("  'langs' - Ver idiomas disponibles")
    print("=" * 70 + "\n")

    while True:
        try:
            text = input("\n📝 Texto a traducir (o 'exit'): ").strip()

            if text.lower() == 'exit':
                print("\n👋 ¡Hasta luego!")
                break

            if text.lower() == 'langs':
                for code, name in sorted(LANGUAGES.items()):
                    print(f"  {code}: {name}")
                continue

            if not text:
                continue

            target_lang = input("🎯 Idioma destino (código): ").strip().lower()

            if target_lang not in LANGUAGES:
                print(f"❌ Idioma '{target_lang}' no reconocido. Use 'langs' para ver opciones.")
                continue

            print(f"\n🔄 Traduciendo a {LANGUAGES[target_lang]}...")

            translation = generate_translation(
                model, tokenizer, text, target_lang
            )

            print(f"\n✅ Traducción: {translation}")

        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Inferencia para traducción de lenguas mayas')
    parser.add_argument('--model', type=str, help='Ruta al modelo fusionado')
    parser.add_argument('--adapter', type=str, help='Ruta a adaptadores LoRA')
    parser.add_argument('--base-model', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                        help='Modelo base (requerido si usa --adapter)')
    parser.add_argument('--text', type=str, help='Texto a traducir')
    parser.add_argument('--target', type=str, help='Idioma destino (código)')
    parser.add_argument('--source', type=str, default='Spanish', help='Idioma origen')
    parser.add_argument('--interactive', action='store_true', help='Modo interactivo')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--max-tokens', type=int, default=256)

    args = parser.parse_args()

    # Validar argumentos
    if not args.model and not args.adapter:
        parser.error("Debe proporcionar --model o --adapter")

    if args.adapter and not args.base_model:
        parser.error("--adapter requiere --base-model")

    # Cargar modelo
    model, tokenizer = load_model(
        model_path=args.model,
        adapter_path=args.adapter,
        base_model=args.base_model
    )

    print("✅ Modelo cargado exitosamente\n")

    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        if not args.text or not args.target:
            parser.error("Modo no-interactivo requiere --text y --target")

        if args.target not in LANGUAGES:
            print(f"❌ Idioma '{args.target}' no reconocido.")
            print("Idiomas disponibles:", ", ".join(LANGUAGES.keys()))
            return

        print(f"🔄 Traduciendo a {LANGUAGES[args.target]}...")

        translation = generate_translation(
            model, tokenizer,
            text=args.text,
            target_lang=args.target,
            source_lang=args.source,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

        print(f"\n✅ Traducción: {translation}")


if __name__ == "__main__":
    main()
