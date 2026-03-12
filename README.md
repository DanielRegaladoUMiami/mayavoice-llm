# MayaVoice LLM

GenAI multilingüe para traducción y conversación en lenguas mayas de Guatemala.

## Objetivo

Construir un modelo de lenguaje fine-tuneado capaz de traducir entre español y 14 lenguas mayas de Guatemala, con el fin de preservar y revitalizar estos idiomas mediante tecnología de IA accesible.

## Modelos Disponibles

| Modelo | Datos | Idiomas | BLEU | chrF | Link |
|--------|-------|---------|------|------|------|
| v1 | 19K train (64K ejemplos) | 13 | 8.85 | 28.82 | [HuggingFace](https://huggingface.co/DanielRegaladoCardoso/mayavoice-llama3.1-8b-lora) |
| v2 | 193K train (241K ejemplos) | 14 | En progreso | En progreso | Próximamente |

## Idiomas Soportados

14 lenguas mayas de Guatemala, priorizadas por población de hablantes:

| Tier | Idiomas | Hablantes aprox. |
|------|---------|-----------------|
| 1 (Prioridad máxima) | K'iche', Q'eqchi', Kaqchikel, Mam | ~3.4M (80%) |
| 2 (Alta prioridad) | Q'anjob'al, Tz'utujil, Poqomchi', Achi | ~555K |
| 3 (Preservación) | Chuj, Poqomam, Awakateko, Sipakapense, Tektiteko | ~132K |
| 4 (En peligro) | Itza' | ~3.4K |

Fuente: Censo Nacional de Población 2018, INE Guatemala.

## Arquitectura

Fine-tuning de Llama 3.1-8B-Instruct con QLoRA (4-bit) usando Unsloth. El modelo recibe instrucciones en español y traduce a/desde lenguas mayas.

```
Usuario → "Traduce al K'iche': Buenos días"
                    ↓
         Llama 3.1-8B + LoRA adapters
                    ↓
         "Utz q'ij, jachin k'o chi awetaq"
```

Próximamente: RAG pipeline con diccionarios (5,224 entradas + audio) para mejorar vocabulario y contexto.

## Fuentes de Datos

| Fuente | Tipo | Pares/Entradas | Uso |
|--------|------|----------------|-----|
| [MayanV](https://github.com/transducens/mayanv) (CC0) | Textos paralelos | 32,328 pares, 13 idiomas | Fine-tuning |
| [Bible Corpus](https://github.com/christos-c/bible-corpus) | NT paralelo | 15,640 versos (Kaqchikel + Mam) | Fine-tuning |
| Augmentación por sinónimos (WordNet ES) | Generado | +72,688 pares | Fine-tuning v2 |
| [Swarthmore Talking Dictionary](https://talkingdictionary.swarthmore.edu/) | Diccionario + audio | 5,224 entradas, 22 idiomas | RAG (futuro) |

## Stack Tecnológico

| Componente | Herramienta |
|------------|-------------|
| Modelo base | Llama 3.1-8B-Instruct (4-bit quantized) |
| Fine-tuning | Unsloth + QLoRA |
| Entrenamiento | Google Colab Pro (A100 80GB) |
| Formato | Alpaca/Chat (system + user + assistant) |
| Métricas | sacrebleu (BLEU, chrF) |
| Hosting | HuggingFace Hub |
| RAG (próximo) | ChromaDB + sentence-transformers |

## Estructura del Proyecto

```
mayavoice-llm/
├── scripts/
│   ├── 00_analyze_dataset.py      # EDA del dataset original
│   ├── 00_clean_parallel_texts.py # Limpieza de textos paralelos
│   ├── 01_create_splits.py        # Train/val/test splits en formato Alpaca
│   ├── 03_merge_mayanv.py         # Merge de todas las splits de MayanV
│   ├── 04_scrape_talking_dict.py  # Scraper de Swarthmore Talking Dictionary
│   └── 05_augment_synonyms.py     # Augmentación por sinónimos (WordNet ES)
├── notebooks/
│   ├── 01_finetune_sprint1_v1.ipynb  # Fine-tuning v1 (19K train, 13 idiomas)
│   └── 02_finetune_v2.ipynb          # Fine-tuning v2 (193K train, 14 idiomas)
├── docs/
│   ├── TECHNICAL.md               # Documentación técnica detallada
│   ├── PLAN_LLMOPS.md             # Plan LLMOps completo
│   └── MODEL_SELECTION.md         # Justificación del modelo base
├── data/                          # Datos (local, no en git)
│   ├── textos-paralelos/          # 14 carpetas, una por idioma
│   ├── textos-paralelos-biblia/   # Kaqchikel + Mam (NT)
│   ├── textos-paralelos-augmented/# Con sinónimos generados
│   ├── splits/                    # v1 splits (64K ejemplos)
│   ├── splits-v2/                 # v2 splits (241K ejemplos)
│   └── diccionarios/scraped/      # 22 CSVs + audio URLs
└── configs/                       # Configuraciones de entrenamiento
```

> **Nota:** Los datos se mantienen en Google Drive y no se suben a git. Los scripts reproducen el pipeline completo desde las fuentes originales.

## Cómo Reproducir

### 1. Preparar datos
```bash
# Clonar y preparar corpus
python scripts/03_merge_mayanv.py
python scripts/05_augment_synonyms.py \
  --data-dir data/textos-paralelos/ \
  --bible-dir data/textos-paralelos-biblia/ \
  --output-dir data/textos-paralelos-augmented/

# Crear splits
python scripts/01_create_splits.py \
  --data-dir data/textos-paralelos-augmented/ \
  --output-dir data/splits-v2/
```

### 2. Entrenar modelo
Subir `train.jsonl`, `val.jsonl`, `test_with_meta.jsonl` a Google Drive y abrir el notebook correspondiente en Colab con GPU A100.

### 3. Usar modelo
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "DanielRegaladoCardoso/mayavoice-llama3.1-8b-lora"
)
```

## Resultados v1 (Baseline)

Entrenado con 19K ejemplos, 3 epochs, LoRA r=16, en A100 (~3 horas):

| Idioma | BLEU | chrF | N |
|--------|------|------|---|
| Poqomam | 19.38 | 38.53 | 20 |
| Sipakapense | 16.03 | 35.86 | 8 |
| Poqomchi' | 13.09 | 31.42 | 11 |
| Tektiteko | 11.57 | 30.09 | 18 |
| Mam | 10.76 | 29.71 | 23 |
| K'iche' | 9.51 | 29.69 | 11 |
| Tz'utujil | 9.52 | 28.66 | 13 |
| **TOTAL** | **8.85** | **28.82** | **200** |

Ejemplo de traducción v1:
- ES→K'iche': "Buenos días, ¿cómo estás?" → "Utz q'ij, jachin k'o chi awetaq"
- ES→Mam: "La tierra es sagrada para nosotros." → "Qa'nxa tx'otx' ti'j qe."

## Sprints

| Sprint | Estado | Descripción |
|--------|--------|-------------|
| 1 | ✅ Completado | MVP: Fine-tuning v1 con Unsloth, métricas baseline |
| 2 | 🔄 En progreso | Expandir datos (Bible + sinónimos), fine-tuning v2 |
| 3 | 📋 Planeado | RAG con diccionarios, test set con hablantes nativos |
| 4 | 📋 Planeado | API, demo Gradio, deployment |

## Licencia

TBD — Los datos de MayanV son CC0. Los datos bíblicos son de dominio público.

## Contacto

Daniel Regalado Cardoso — dxr1491@miami.edu
University of Miami
