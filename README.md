# MayaVoice LLM

GenAI multilingüe para traducción y conversación en lenguas mayas de Guatemala.

## Objetivo

Construir un modelo de lenguaje fine-tuneado que pueda:
- **Traducir** entre español/inglés y lenguas mayas
- **Conversar** en idiomas mayas como asistente de IA
- **Mejorar continuamente** con feedback de usuarios

## Idiomas Target

22 lenguas mayas de Guatemala, priorizadas por población de hablantes:

| Tier | Idiomas | Hablantes aprox. |
|------|---------|-----------------|
| 1 (Prioridad máxima) | K'iche', Q'eqchi', Kaqchikel, Mam | ~3.4M (80%) |
| 2 (Alta prioridad) | Q'anjob'al, Tz'utujil, Poqomchi', Achi, Ixil | ~555K |
| 3 (Preservación) | Chuj, Poqomam, Awakateko, Sipakapense, Tektiteko | ~132K |
| 4 (En peligro) | Itza', Mopan | ~3.4K |

Fuente: Censo Nacional de Población 2018, INE Guatemala.

## Arquitectura

**RAG + Fine-tuning** sobre un modelo base open-source (Qwen 2.5 / Mistral / Llama).

```
Usuario → API → Language Router → [RAG Context + LLM Fine-tuned] → Respuesta
                                         ↑
                              Diccionarios + Gramáticas
                              (ChromaDB / FAISS)
```

## Stack Tecnológico

| Componente | Herramienta |
|------------|-------------|
| Modelo base | Qwen 2.5-7B-Instruct (u otro open-source) |
| Fine-tuning | Unsloth + QLoRA |
| Entrenamiento | Google Colab → GCP Vertex AI |
| RAG | ChromaDB + sentence-transformers |
| API | FastAPI + vLLM |
| UI | Gradio |
| Hosting | HuggingFace Hub / GCP |
| Local | Ollama (GGUF) |

## Estructura del Proyecto

```
mayavoice-llm/
├── configs/              # Configuraciones de entrenamiento
├── data/                 # Datos (local, no en git)
│   ├── textos-paralelos/ # Pares ES↔Maya por idioma
│   ├── diccionarios/     # CSVs y PDFs
│   ├── entrenamiento/    # JSONLs procesados
│   ├── corpus-rag/       # Textos para RAG
│   └── audio/            # Metadata de audio
├── notebooks/            # Jupyter notebooks
├── scripts/              # Scripts del pipeline
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── rag/              # RAG pipeline
│   ├── training/         # Training pipeline
│   └── evaluation/       # Métricas y evaluación
├── tests/                # Tests
└── docs/                 # Documentación
```

> **Nota:** Los datos (~6MB+ de JSONLs, CSVs, PDFs) se mantienen localmente y no se suben a git. Ver `data/README.md` para instrucciones de setup.

## Data Sources

- [MayanV Dataset](https://github.com/transducens/mayanv) (CC0) — Textos paralelos
- [Swarthmore Talking Dictionary](https://talkingdictionary.swarthmore.edu/) — Diccionarios
- [AmericasNLP](https://github.com/AmericasNLP/americasnlp2021) — Shared tasks

## LLMOps Approach

Este proyecto sigue un framework LLMOps de 3 fases y 10 pasos:

1. **Ideation:** Data Sourcing → Base Model Selection
2. **Development:** Prompt Engineering → Chains & Agents → RAG vs Fine-tuning → Testing
3. **Operational:** Deployment → Monitoring → Cost Management → Governance

Ver `docs/PLAN_LLMOPS.md` para el plan completo.

## Sprints

| Sprint | Milestone | Descripción |
|--------|-----------|-------------|
| 1 | MVP en Colab | Fine-tuning con Unsloth, primeras pruebas |
| 2 | Más datos + RAG | Expandir dataset, implementar RAG |
| 3 | GCP + Evaluación | Entrenamiento robusto, métricas formales |
| 4 | Deployment | API, chatbot, HuggingFace, feedback loop |

## Licencia

TBD

## Contacto

Daniel Regalado Cardoso — dxr1491@miami.edu
