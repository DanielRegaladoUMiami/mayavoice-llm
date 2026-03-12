# Selección de Modelo Base — Decision Record

## Issue #5: Evaluar y seleccionar modelo base

### Decisión Final

**Modelo primario:** `meta-llama/Llama-3.1-8B-Instruct`
**Modelo alternativo:** `Qwen/Qwen2.5-7B-Instruct`

### Comparación de Candidatos

| Criterio | Llama 3.1-8B | Qwen 2.5-7B | Mistral-7B-v0.3 |
|----------|-------------|-------------|-----------------|
| **Parámetros** | 8B | 7B | 7B |
| **Español nativo** | ✅ Core language | ✅ 29+ langs | ✅ Soporte |
| **Context window** | 128K | 128K | 32K |
| **Unsloth QLoRA** | ✅ Excelente | ✅ Excelente | ✅ Bueno |
| **VRAM (4-bit)** | ~8GB | ~6-8GB | ~6-8GB |
| **Colab Free T4** | ✅ Cabe | ✅ Cabe | ✅ Cabe |
| **Licencia** | Community License | Apache 2.0 | Apache 2.0 |
| **Fine-tune idiomas nuevos** | ✅ Explícito en licencia | ✅ | ✅ |
| **Tokens de entrenamiento** | 15T | No divulgado | No divulgado |
| **Comunidad/docs** | Muy amplia | Amplia | Amplia |

### Justificación de Llama 3.1-8B-Instruct

1. **Español como idioma core**: Entrenado con español como uno de sus 8 idiomas principales, lo cual facilita el transfer learning hacia idiomas mayas a través del español.

2. **Licencia permite fine-tuning para idiomas nuevos**: La Llama Community License explícitamente permite desarrollar modelos para idiomas fuera de los 8 soportados.

3. **Mejor soporte de Unsloth**: Documentación extensa, múltiples ejemplos en Colab, optimizaciones específicas. 2x más rápido y 60% menos memoria.

4. **128K context window**: Permite incluir múltiples ejemplos de traducción en el prompt para few-shot learning.

5. **15 trillion tokens de entrenamiento**: La base de conocimiento más amplia entre los candidatos, lo cual beneficia el aprendizaje de patrones lingüísticos.

6. **Instruction-tuned**: Ya optimizado para seguir instrucciones, facilitando el uso en formato Alpaca para traducción y conversación.

### Por qué NO Qwen 2.5 como primario

- Performance comparable, pero menos documentación para fine-tuning de idiomas nuevos.
- Se mantiene como alternativa por si Llama presenta problemas con el tokenizer para caracteres especiales del maya (ʼ, ä, etc.).

### Por qué NO Mistral-7B

- Context window de 32K (4x menor que los otros).
- Comunidad de fine-tuning más pequeña.
- Sin ventaja clara sobre Llama o Qwen.

### Por qué NO NLLB-200

- Aunque es especializado en traducción de idiomas low-resource, es encoder-decoder.
- No compatible con el flujo instruction-based de Unsloth/QLoRA.
- No permite el modo conversación en Maya (solo traducción directa).
- Podría usarse como baseline de referencia en evaluación futura.

### Configuración de Fine-tuning

```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
quantization: 4-bit (nf4)
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
max_seq_length: 2048
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
warmup_ratio: 0.1
num_train_epochs: 3
optimizer: adamw_8bit
lr_scheduler_type: cosine
```

### Plan de Evaluación

1. **Sprint 1**: Fine-tune Llama 3.1-8B con Unsloth en Colab → medir BLEU/chrF
2. **Si resultados < aceptables**: Probar Qwen 2.5-7B con misma config
3. **Sprint 3**: Comparar ambos modelos con evaluación humana (UVG)

### Riesgos

- **Tokenizer**: Llama podría no tokenizar eficientemente caracteres mayas (ʼ, ä, ë, ö, ü). Mitigación: verificar tokenización antes de entrenar.
- **Data size**: 23K ejemplos es bajo. Mitigación: data augmentation en Sprint 2, contactar UVG.
- **Overfitting**: Alto riesgo con poco dato. Mitigación: QLoRA limita parámetros entrenables, early stopping.
