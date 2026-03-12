# MayaVoice — Documentación Técnica

## Resumen del Sistema

MayaVoice es un sistema de traducción automática neuronal para lenguas mayas de Guatemala. Utiliza fine-tuning de modelos de lenguaje grandes (LLM) con técnicas de eficiencia de parámetros (QLoRA) para adaptar Llama 3.1-8B-Instruct a la tarea de traducción bidireccional español ↔ lenguas mayas.

## Pipeline de Datos

### Fuentes

El corpus de entrenamiento se construye a partir de tres fuentes complementarias:

**1. MayanV (transducens/mayanv)** — Corpus paralelo académico bajo licencia CC0. Contiene textos paralelos español-maya en 13 idiomas: Achi, Awakateko, Chuj, Itza', K'iche', Mam, Poqomam, Poqomchi', Q'anjob'al, Q'eqchi', Sipakapense, Tektiteko y Tz'utujil. Se procesaron las splits train, dev y test, eliminando duplicados y prefijos de código (`#code#`). Total: 32,328 pares únicos.

**2. Bible Corpus (christos-c/bible-corpus)** — Nuevo Testamento paralelo alineado por versículos. Se extrajeron Cakchiquel-NT.xml (7,851 versos) y Mam-NT.xml (7,789 versos) alineados con Spanish.xml. Esto proporcionó datos de Kaqchikel donde antes no existían, y datos adicionales de Mam. Total: 15,640 pares.

**3. Augmentación por sinónimos** — Se utiliza WordNet en español (NLTK, Open Multilingual Wordnet 1.4) para reemplazar 1-2 palabras de contenido por sinónimos en las oraciones en español, manteniendo la traducción maya intacta. Se excluyen stopwords y palabras menores a 4 caracteres. Esto genera variaciones que enseñan al modelo que distintas formas de expresar lo mismo en español corresponden a la misma traducción maya. Multiplicador: ~1.5x. Total: +72,688 pares generados.

### Pipeline de Procesamiento

```
Fuentes originales
       ↓
scripts/03_merge_mayanv.py      → Merge y dedup de MayanV
scripts/05_augment_synonyms.py  → Augmentación + merge con Bible
       ↓
data/textos-paralelos-augmented/  → 14 carpetas (idioma/data.es + data.{code})
       ↓
scripts/01_create_splits.py     → Splits estratificados 80/10/10
       ↓
data/splits-v2/                 → train.jsonl, val.jsonl, test_with_meta.jsonl
```

### Formato de Datos (Alpaca)

Cada ejemplo es un JSON con tres campos:

```json
{
  "instruction": "Traduce del español al K'iche'.",
  "input": "Buenos días, ¿cómo estás?",
  "output": "Utz q'ij, jachin k'o chi awetaq"
}
```

Se generan ejemplos en ambas direcciones (ES→Maya y Maya→ES) con 3 templates de instrucción por dirección para variabilidad. El test set incluye metadata adicional (`_lang`, `_direction`) para evaluación por idioma.

### Estadísticas del Corpus

| Versión | Pares paralelos | Idiomas | Train | Val | Test | Ejemplos totales |
|---------|----------------|---------|-------|-----|------|-----------------|
| v1 | 32,328 | 13 | 51,714 | 6,456 | 6,486 | 64,656 |
| v2 | 120,656 | 14 | 193,038 | 24,116 | 24,158 | 241,312 |

Los "ejemplos totales" son 2x los pares paralelos porque cada par genera dos ejemplos (ES→Maya y Maya→ES).

## Modelo

### Arquitectura

Se utiliza Llama 3.1-8B-Instruct como modelo base, cargado en 4-bit quantization (bnb-4bit) vía Unsloth. Se aplican adaptadores LoRA (Low-Rank Adaptation) a las capas de atención y feed-forward del transformer.

### Configuración de Entrenamiento

| Parámetro | v1 | v2 |
|-----------|----|----|
| Modelo base | unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit | Mismo |
| Max sequence length | 2048 | 2048 |
| LoRA rank (r) | 16 | 32 |
| LoRA alpha | 32 | 64 |
| LoRA dropout | 0.05 | 0.05 |
| Target modules | q,k,v,o,gate,up,down_proj | Mismo |
| Batch size | 4 | 4 |
| Gradient accumulation | 4 | 4 |
| Effective batch size | 16 | 16 |
| Learning rate | 2e-4 | 1e-4 |
| Warmup ratio | 0.1 | 0.05 |
| Epochs | 3 | 2 |
| Optimizer | AdamW 8-bit | Mismo |
| LR scheduler | Cosine | Mismo |
| Precision | BF16 (A100) | Mismo |
| Parámetros entrenables | 41.9M (0.92%) | 83.9M (1.03%) |

### Justificación de Cambios v1 → v2

**LoRA r: 16 → 32.** Con 3.7x más datos de entrenamiento, el modelo necesita más capacidad de adaptación. LoRA rank 32 duplica los parámetros entrenables, permitiendo capturar patrones más complejos de 14 idiomas simultáneamente.

**Learning rate: 2e-4 → 1e-4.** Con más datos, cada epoch contiene más steps de gradiente. Un learning rate más bajo previene inestabilidad y permite una convergencia más suave.

**Warmup ratio: 0.1 → 0.05.** El dataset más grande implica más steps totales, por lo que 5% del total es suficiente warmup.

**Epochs: 3 → 2.** v2 ve 193K × 2 = 386K ejemplos vs v1 que vio 19K × 3 = 57K. Menos epochs con más datos reduce el riesgo de overfitting y el tiempo de entrenamiento.

### Formato de Chat

Los datos se formatean usando el template de chat de Llama 3.1:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Eres MayaVoice, un asistente especializado en idiomas mayas de Guatemala.
Traduces con precisión y respetas la riqueza lingüística de cada idioma.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Traduce del español al K'iche'.

Buenos días, ¿cómo estás?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Utz q'ij, jachin k'o chi awetaq<|eot_id|>
```

Se usa `train_on_responses_only` de Unsloth para que el loss solo se calcule sobre las respuestas del asistente, no sobre las instrucciones.

## Evaluación

### Métricas

Se utilizan dos métricas estándar de traducción automática via sacrebleu:

**BLEU (Bilingual Evaluation Understudy):** Mide la precisión de n-gramas (1-4 palabras) entre la traducción del modelo y la referencia. Rango 0-100, mayor es mejor.

**chrF (Character F-score):** Mide similitud a nivel de caracteres. Más apropiado para lenguas aglutinantes como las mayas, donde las palabras son compuestas y los morfemas son significativos.

### Resultados v1

Evaluado sobre una muestra de 200 ejemplos del test set:

| Idioma | N | BLEU | chrF |
|--------|---|------|------|
| Poqomam | 20 | 19.38 | 38.53 |
| Sipakapense | 8 | 16.03 | 35.86 |
| Poqomchi' | 11 | 13.09 | 31.42 |
| Tektiteko | 18 | 11.57 | 30.09 |
| Mam | 23 | 10.76 | 29.71 |
| K'iche' | 11 | 9.51 | 29.69 |
| Tz'utujil | 13 | 9.52 | 28.66 |
| Chuj | 23 | 7.30 | 29.52 |
| Q'eqchi' | 22 | 4.19 | 27.84 |
| Awakateko | 11 | 3.38 | 14.97 |
| Q'anjob'al | 21 | 3.14 | 28.90 |
| Achi | 13 | 2.48 | 16.38 |
| Itza' | 6 | 0.00 | 11.80 |
| **TOTAL** | **200** | **8.85** | **28.82** |

### Contexto de Referencia

Para lenguas de bajos recursos, estos números se comparan favorablemente con la literatura. Los mejores sistemas de AmericasNLP 2021/2024 para lenguas indígenas reportan BLEU de 5-15 con datasets comparables. MayaVoice v1 logra 8.85 BLEU con solo 19K ejemplos de entrenamiento en 13 idiomas simultáneos.

## Datos para RAG (Futuro)

Se dispone de 5,224 entradas de diccionario del Swarthmore Talking Dictionary (22 idiomas mayas), incluyendo: headword en maya, traducción al español e inglés, IPA, parte del discurso, y URL de audio MP3. Estos datos alimentarán un pipeline RAG con ChromaDB para mejorar la cobertura de vocabulario sin necesidad de reentrenar el modelo.

## Infraestructura

### Entrenamiento

Google Colab Pro con NVIDIA A100-SXM4-80GB. v1 tomó ~3 horas, v2 estimado ~6-12 horas dependiendo de configuración.

### Hosting del Modelo

HuggingFace Hub. Los adaptadores LoRA (~168MB) se almacenan separados del modelo base, lo que permite distribución eficiente.

### Reproducibilidad

Todos los scripts están versionados en GitHub. El pipeline completo es reproducible con las fuentes de datos públicas (MayanV CC0, Bible Corpus, Swarthmore). Seed fija (42) en todas las operaciones aleatorias.

## Contribuir

1. Clonar el repo: `git clone https://github.com/DanielRegaladoUMiami/mayavoice-llm`
2. Ejecutar el pipeline de datos (ver scripts/ en orden numérico)
3. Subir splits a Google Drive
4. Abrir el notebook correspondiente en Colab con A100
5. Los datos de MayanV se descargan automáticamente desde HuggingFace
6. Los datos bíblicos requieren clonar `christos-c/bible-corpus`

## Contacto

Daniel Regalado Cardoso — dxr1491@miami.edu — University of Miami
