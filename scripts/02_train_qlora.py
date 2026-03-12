#!/usr/bin/env python3
"""
02_train_qlora.py
=================
Entrena modelo LLM con QLoRA para traducción español-lenguas mayas.

Características:
- Fine-tuning con QLoRA (4-bit quantization)
- Soporte para GPU única (L4, T4, A100)
- Logging a W&B
- Checkpointing automático
- Evaluación en validation set

Uso:
    python 02_train_qlora.py --config ../configs/training_config.yaml
"""

import os
import sys
import json
import torch
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


def load_config(config_path: str) -> dict:
    """Carga configuración desde YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def format_alpaca_prompt(example: dict, tokenizer) -> dict:
    """Formatea ejemplos en el estilo Alpaca."""
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

    return {"text": prompt}


def tokenize_function(examples: dict, tokenizer, max_length: int):
    """Tokeniza ejemplos para entrenamiento."""
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs


def setup_model_and_tokenizer(config: dict):
    """Configura modelo y tokenizer con QLoRA."""
    model_name = config['model']['base_model']

    print(f"📦 Cargando modelo: {model_name}")

    # Configuración de quantización 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Cargar modelo
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Preparar para k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configuración LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Aplicar LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_and_prepare_datasets(config: dict, tokenizer):
    """Carga y prepara datasets para entrenamiento."""
    data_path = Path(config['data']['processed_dir'])

    train_file = data_path / "train.jsonl"
    val_file = data_path / "val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"❌ No se encontró: {train_file}")

    print(f"\n📂 Cargando datasets...")
    print(f"   Train: {train_file}")
    print(f"   Val: {val_file}")

    # Cargar datasets
    dataset_dict = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file) if val_file.exists() else None
        }
    )

    # Formatear prompts
    print("\n🔄 Formateando prompts...")
    dataset_dict = dataset_dict.map(
        lambda x: format_alpaca_prompt(x, tokenizer),
        remove_columns=dataset_dict["train"].column_names
    )

    # Tokenizar
    print("🔤 Tokenizando...")
    max_length = config['training']['max_length']

    dataset_dict = dataset_dict.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"]
    )

    print(f"\n✅ Datasets preparados:")
    print(f"   Train: {len(dataset_dict['train']):,} ejemplos")
    if "validation" in dataset_dict:
        print(f"   Val: {len(dataset_dict['validation']):,} ejemplos")

    return dataset_dict


def setup_training_args(config: dict, output_dir: Path) -> TrainingArguments:
    """Configura argumentos de entrenamiento."""
    train_config = config['training']

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config['num_epochs'],
        per_device_train_batch_size=train_config['batch_size'],
        per_device_eval_batch_size=train_config['batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        lr_scheduler_type=train_config['lr_scheduler'],
        warmup_ratio=train_config['warmup_ratio'],
        weight_decay=train_config['weight_decay'],
        logging_steps=train_config['logging_steps'],
        save_steps=train_config['save_steps'],
        eval_steps=train_config['eval_steps'],
        save_total_limit=3,
        evaluation_strategy="steps" if config['data'].get('val_ratio', 0) > 0 else "no",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="wandb" if config['training'].get('use_wandb') else "none",
        run_name=f"mayan-llm-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Entrenar LLM para lenguas mayas con QLoRA')
    parser.add_argument('--config', type=str, default='../configs/training_config.yaml',
                        help='Ruta al archivo de configuración')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Checkpoint desde el cual resumir')

    args = parser.parse_args()

    # Cargar configuración
    config = load_config(args.config)

    print("=" * 70)
    print("🚀 ENTRENAMIENTO QLORA - LENGUAS MAYAS")
    print("=" * 70)

    # Setup W&B
    if config['training'].get('use_wandb'):
        import wandb
        wandb.init(
            project=config['training'].get('wandb_project', 'mayan-llm'),
            name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config
        )

    # Output directory
    output_dir = Path(config['training']['output_dir']) / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output: {output_dir.resolve()}")

    # Setup modelo
    model, tokenizer = setup_model_and_tokenizer(config)

    # Cargar datasets
    dataset_dict = load_and_prepare_datasets(config, tokenizer)

    # Training arguments
    training_args = setup_training_args(config, output_dir)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("validation"),
        data_collator=data_collator,
    )

    # Guardar configuración
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Entrenar
    print("\n" + "=" * 70)
    print("🏋️  INICIANDO ENTRENAMIENTO")
    print("=" * 70 + "\n")

    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Guardar modelo final
    final_model_path = output_dir / "final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETO")
    print("=" * 70)
    print(f"\n📦 Modelo guardado en: {final_model_path.resolve()}")
    print("\n🚀 Siguiente paso: python 03_merge_model.py")


if __name__ == "__main__":
    main()
