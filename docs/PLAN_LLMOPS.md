# Plan LLMOps — MayaVoice / Mayan GPT

**Fecha:** 11 marzo 2026
**Autor:** Daniel Regalado Cardoso
**Objetivo:** Construir un GenAI multilingüe que traduzca entre español/inglés y lenguas mayas, y que pueda conversar en idiomas mayas. Producto comercial con capacidad de mejora continua.

---

## Análisis de Prioridad por Idioma

Antes de definir el pipeline, es clave saber qué idiomas priorizar. Según el Censo 2018 de Guatemala (INE), hay ~4 millones de hablantes de idiomas mayas. La distribución no es uniforme:

### Tier 1 — Máxima prioridad (~80% de hablantes)

| Idioma | Hablantes aprox. | % de población maya | Tenemos datos paralelos? |
|--------|-----------------|--------------------|-|
| K'iche' | 900,000+ | 27.1% | ✅ 1,000 pares |
| Q'eqchi' | 800,000+ | 22.1% | ✅ 1,000 pares |
| Kaqchikel | 1,068,000+ | 17.2% | ❌ FALTA — solo diccionario y corpus |
| Mam | 640,000+ | 13.6% | ✅ 1,000 pares |

**Hallazgo crítico:** Kaqchikel es el 3er idioma más hablado y NO tenemos textos paralelos. Solo tenemos diccionario CSV y un corpus RAG de 100 líneas. Esto es una brecha que hay que cerrar.

### Tier 2 — Alta prioridad

| Idioma | Hablantes aprox. | Tenemos datos? |
|--------|-----------------|---------------|
| Q'anjob'al | ~150,000 | ✅ 1,000 pares |
| Tz'utujil | ~90,000 | ✅ 1,000 pares |
| Poqomchi' | ~115,000 | ✅ 1,000 pares |
| Achi | ~105,000 | ❌ Solo diccionario |
| Ixil | ~95,000 | ❌ Solo diccionario |

### Tier 3 — Preservación cultural

| Idioma | Hablantes aprox. | Tenemos datos? |
|--------|-----------------|---------------|
| Chuj | ~65,000 | ✅ 1,000 pares |
| Poqomam | ~40,000 | ✅ 1,000 pares |
| Awakateko | ~20,000 | ✅ 1,000 pares (49 vacías) |
| Sipakapense | ~5,000 | ✅ 1,000 pares |
| Tektiteko | ~2,000 | ✅ 1,000 pares |

### Tier 4 — En peligro de extinción

| Idioma | Hablantes | Tenemos datos? |
|--------|-----------|---------------|
| Itza' | ~406 | ✅ 1,000 pares (35 vacías) |
| Mopan | ~3,000 | ❌ Solo diccionario |

**Decisión recomendada:** Entrenar con TODOS los idiomas disponibles (la diversidad ayuda al modelo), pero priorizar evaluación y calidad en Tier 1 y 2. Buscar activamente datos de Kaqchikel y Achi.

---

## Fuentes de Datos Existentes y Potenciales

### Fuente actual: MayanV (transducens/mayanv)

- Repositorio: https://github.com/transducens/mayanv
- Paper: NAACL 2024 — "Curated Datasets and Neural Models for Machine Translation of Informal Registers between Mayan and Spanish Vernaculars"
- Licencia: CC0 (libre uso)
- Estado: 11 commits, última actualización marzo 2024, repo estático
- Idiomas: 15 lenguas mayas, el más grande es Tzeltal con ~19,846 pares
- **Acción:** Revisar si hay más datos de los que ya extrajimos (especialmente Kaqchikel)

### Fuentes potenciales adicionales

1. **Universidad del Valle de Guatemala (UVG)** — Daniel puede contactar directamente
2. **Academia de Lenguas Mayas de Guatemala (ALMG)** — publicaciones, vocabularios, gramáticas
3. **AmericasNLP Shared Tasks** — competencias anuales con nuevos datasets: https://github.com/AmericasNLP/americasnlp2021
4. **Swarthmore Talking Dictionary** — ya tenemos los CSVs de aquí
5. **RLHF del producto** — usuarios califican traducciones → datos de mejora continua

---

## Las 3 Fases y 10 Pasos del LLMOps

---

## FASE 1: IDEATION (Definición)

### Paso 1 — Data Sourcing

**Estado actual:** Tenemos ~43,900 ejemplos (training_data_v2.jsonl) compuestos de ~17,686 traducciones MayanV + ~26,225 entradas de diccionario. 12 idiomas con textos paralelos de 1,000 pares cada uno.

**Acciones definidas:**

1. **Cerrar la brecha de Kaqchikel:**
   - Revisar el repo MayanV completo para verificar si tiene datos de Kaqchikel que no extrajimos
   - Contactar UVG para pedir corpus paralelos ES↔Kaqchikel
   - Buscar en ALMG publicaciones bilingües
   - En el peor caso, usar el diccionario CSV + augmentation para generar pares sintéticos

2. **Limpiar datos existentes:**
   - Eliminar las 49 líneas vacías de Awakateko y 35 de Itza'
   - Evaluar el prefijo `#lang#` en las oraciones ES — **recomendación: mantenerlo** porque le dice al modelo qué idioma traducir

3. **Crear split formal:**
   - 90% train / 5% validation / 5% test
   - Estratificado por idioma para garantizar representación

4. **Preparar pipeline de ingestión de nuevos datos:**
   - Script que tome un archivo .es y .{lang} nuevos → los convierta a formato Alpaca JSONL
   - Que agregue al dataset existente sin duplicar
   - Versionado del dataset (v3, v4, etc.)

**Entregable:** Dataset v3 limpio, con splits, y pipeline de ingestión.

---

### Paso 2 — Base Model Selection

**Decisión: Qwen 2.5-7B-Instruct** (primera opción) o **Mistral-7B-Instruct-v0.3** (alternativa)

**Justificación de Qwen 2.5-7B:**

- Soporte multilingüe superior (100+ idiomas en pre-training)
- Excelente rendimiento en instrucción-following
- Soportado nativamente por Unsloth (2x más rápido, 60% menos VRAM)
- 7B es el sweet spot: suficiente capacidad para multilingüe, cabe en T4/L4
- Exportable a GGUF para Ollama (uso local) y a HuggingFace para API
- Comunidad activa, bien documentado

**Alternativa — Mistral 7B:**

- Más ligero en inferencia
- Buena arquitectura base
- Menos multilingüe que Qwen (peor para idiomas de bajos recursos)

**Alternativa — NLLB-200 (Meta):**

- Diseñado específicamente para traducción de 200+ idiomas
- Incluye algunos idiomas mayas
- PERO: no es un chat model, solo traduce. No cumple el requisito de "conversar en maya"

**Alternativa — Llama 3.1-8B-Instruct:**

- Buena base multilingüe (15 trillion tokens)
- Balance entre capacidad y eficiencia
- Buena opción si Qwen no rinde como esperamos

**Plataforma de entrenamiento:**

| Opción | GPU | Costo | Velocidad | Recomendación |
|--------|-----|-------|-----------|---------------|
| Google Colab Free | T4 (16GB) | $0 | ~2 hrs | Para prototipo rápido |
| Google Colab Pro | L4/A100 | ~$10/mes | ~30 min | Para iteraciones |
| GCP Vertex AI | L4/A100 | Créditos disponibles | ~30 min | Para producción |
| HuggingFace Spaces | T4/A10G | ~$0.60/hr | ~1 hr | Alternativa si GCP falla |

**Recomendación:** Empezar con Colab Free + Unsloth para validar, luego mover a GCP con créditos para entrenamiento final.

---

## FASE 2: DEVELOPMENT (Desarrollo)

### Paso 3 — Prompt Engineering

**Diseño del sistema de prompts para el modelo fine-tuneado:**

El modelo necesita entender 3 tipos de interacción:

**Tipo A — Traducción directa:**
```
Instruction: Traduce la siguiente oración del español al K'iche'
Input: Tienes indigestión
Output: Kq'ulq'ut ri apam
```

**Tipo B — Chat en idioma maya:**
```
Instruction: Responde en Q'eqchi' a la siguiente pregunta
Input: ¿Cómo estás?
Output: Us lin chaabil, b'antyox aawu. Laa'at, ¿ma us laa chaabil?
```

**Tipo C — Explicación lingüística:**
```
Instruction: Explica el significado y uso de la siguiente palabra en Mam
Input: t-xe
Output: "t-xe" significa "debajo de" o "al pie de". Es una partícula posicional que indica ubicación inferior. Ejemplo: "t-xe witz" = "al pie del cerro".
```

**Acciones:**

1. Diseñar template de prompts para cada tipo de interacción
2. Crear ~500 ejemplos manuales de Tipo B y C (actualmente solo tenemos Tipo A)
3. Usar Claude/GPT para generar ejemplos sintéticos de Tipo B y C, validados manualmente
4. Implementar versionado de prompts (prompt registry)

---

### Paso 4 — Chains & Agents

**Arquitectura del sistema completo:**

```
Usuario → API Gateway → Router de Idioma → [RAG + LLM Fine-tuned] → Respuesta
                              ↓
                    Detectar idioma de entrada
                    Seleccionar contexto RAG
                    Enviar al modelo con prompt apropiado
```

**Componentes:**

1. **Language Router:** Detecta si el input es español, inglés, o un idioma maya específico
2. **RAG Module:** Busca en el knowledge base (diccionarios + gramáticas) para dar contexto al modelo
3. **LLM Fine-tuned:** Genera la traducción/respuesta con el contexto de RAG
4. **Feedback Collector:** Recoge calificaciones de usuarios para RLHF futuro

**Herramientas:**

- LangChain o LlamaIndex para orchestración
- ChromaDB o FAISS para vector store (RAG)
- FastAPI para la API
- Gradio o Streamlit para UI inicial

---

### Paso 5 — RAG vs Fine-tuning

**Decisión: AMBOS — RAG + Fine-tuning combinados**

| Componente | Método | Por qué |
|------------|--------|---------|
| Traducción de oraciones | Fine-tuning | Necesita estar internalizado en el modelo |
| Vocabulario y definiciones | RAG | 9,000+ entradas de diccionario como contexto dinámico |
| Gramática y reglas | RAG | PDFs de gramáticas y vocabularios |
| Chat conversacional | Fine-tuning | Requiere fluidez, no lookup |
| Nuevos idiomas/datos | Fine-tuning incremental | LoRA adapters adicionales |

**Pipeline RAG:**

1. Procesar los 9 PDFs de documentos de referencia → chunks de texto
2. Procesar los 25 diccionarios CSV → entradas indexables
3. Procesar los 22 corpus RAG existentes (expandirlos primero)
4. Embeddings con modelo multilingüe (e.g., sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
5. Indexar en ChromaDB/FAISS
6. En inference: buscar contexto relevante → inyectar en prompt → generar respuesta

**Fine-tuning pipeline:**

1. Dataset v3 limpio en formato Alpaca
2. Unsloth + QLoRA (4-bit) sobre Qwen 2.5-7B-Instruct
3. Config: rank=16, alpha=32, epochs=3, lr=2e-4
4. Guardar LoRA adapter (~50-100 MB)
5. Exportar a GGUF para Ollama y/o subir a HuggingFace

**Para agregar más datos después:**

- NO reentrenar desde cero — cargar el adapter anterior y hacer fine-tuning incremental
- O entrenar un nuevo LoRA adapter y mergearlo con el anterior
- El pipeline de ingestión del Paso 1 alimenta directamente este proceso

---

### Paso 6 — Testing

**Métricas de evaluación:**

| Métrica | Qué mide | Target |
|---------|----------|--------|
| BLEU | Overlap de n-gramas con referencia | > 15 (low-resource baseline) |
| chrF | Overlap a nivel de caracteres (mejor para idiomas aglutinantes) | > 30 |
| COMET | Calidad semántica de traducción | > 0.5 |
| Perplexity | Qué tan "natural" suena el output | Menor es mejor |
| Human eval | Hablantes nativos califican 1-5 | > 3.5 promedio |

**Plan de testing:**

1. **Test set automático:** 5% del dataset (~2,200 pares) estratificado por idioma
2. **Test set manual:** 50 oraciones por idioma Tier 1 (200 total), traducidas por hablantes nativos de UVG
3. **A/B testing de prompts:** Comparar diferentes templates de instrucción
4. **Regression testing:** Cada nuevo entrenamiento debe igualar o superar métricas anteriores
5. **Adversarial testing:** Inputs con errores ortográficos, mezcla de idiomas, oraciones muy largas

**Evaluación por idioma:**
- Cada idioma se evalúa por separado
- Se espera mejor rendimiento en Tier 1 (más datos)
- Se documenta el rendimiento por idioma para saber dónde necesitamos más datos

---

## FASE 3: OPERATIONAL (Producción)

### Paso 7 — Deployment

**Estrategia de deployment en 3 niveles:**

**Nivel 1 — Demo/MVP (semana 1-2):**
- HuggingFace Space con Gradio
- Modelo GGUF en HuggingFace Hub
- Costo: $0 (free tier) o ~$0.60/hr si se necesita GPU
- URL pública para testing con usuarios

**Nivel 2 — API (semana 3-4):**
- FastAPI + vLLM en GCP Cloud Run o Compute Engine con L4
- Endpoint REST para integración con apps
- HuggingFace Inference Endpoints como alternativa (~$1.30/hr para 7B)

**Nivel 3 — Producción (mes 2+):**
- GCP Vertex AI Endpoint
- Auto-scaling basado en tráfico
- CI/CD con GitHub Actions: push nuevo adapter → test automático → deploy si pasa
- Ollama distribution para uso offline/local

**CI/CD pipeline:**
```
Nuevo dato → Validación → Agregar a dataset → Re-train LoRA → Test automático
    → Si pasa métricas → Deploy a staging → Smoke test → Deploy a producción
```

---

### Paso 8 — Monitoring

**Qué monitorear:**

1. **Calidad de traducciones:** Score automático (BLEU/chrF) en tiempo real sobre respuestas
2. **Latencia:** Tiempo de respuesta por request (target: < 3 segundos)
3. **User satisfaction:** Thumbs up/down en cada respuesta
4. **Error rate:** Respuestas vacías, timeout, crashes
5. **Drift detection:** Si la calidad baja con el tiempo (data drift)
6. **Usage por idioma:** Qué idiomas se usan más (priorizar mejora)

**Herramientas:**
- LangSmith o Weights & Biases para LLM monitoring
- GCP Cloud Monitoring para infraestructura
- Custom dashboard con métricas de negocio

**Alertas:**
- BLEU score promedio cae > 10% → alerta
- Latencia > 5 segundos → alerta
- Error rate > 5% → alerta crítica
- User satisfaction < 3.0 → revisar últimos cambios

---

### Paso 9 — Cost Management

**Estimación de costos:**

| Componente | Costo mensual estimado |
|------------|----------------------|
| Entrenamiento (GCP, ~4 hrs/mes) | ~$10-20 con créditos |
| Inference API (GCP L4, on-demand) | ~$50-150 según tráfico |
| HuggingFace Inference Endpoint | ~$50-100 (always-on) |
| HuggingFace Space (GPU) | ~$0-40 según uso |
| Storage (GCS, modelos+datos) | ~$5 |
| **Total rango** | **$15-315/mes** |

**Estrategias de ahorro:**

1. **Quantización:** GGUF Q4_K_M reduce modelo de ~14GB a ~4GB, inference más rápido
2. **Prompt compression:** Mantener prompts cortos y eficientes (menos tokens = menos costo)
3. **Caching:** Cachear traducciones frecuentes (diccionario de traducciones comunes)
4. **Batch inference:** Agrupar requests en horarios de bajo tráfico
5. **Scale to zero:** Usar Cloud Run que escala a 0 cuando no hay tráfico
6. **Modelo más pequeño:** Si 7B es overkill, probar con 3B (Qwen 2.5-3B o Llama 3.2-3B)

---

### Paso 10 — Governance & Security

**Data governance:**

1. **Licencias:** MayanV es CC0 (libre), diccionarios Swarthmore tienen licencia propia → verificar
2. **Atribución:** Citar AmericasNLP, Swarthmore Talking Dictionary, ALMG en el producto
3. **Datos de usuarios:** Si se implementa RLHF, los datos de retroalimentación son PII → GDPR/compliance
4. **Versionado:** Cada versión del modelo y dataset se versiona y se documenta

**Seguridad del modelo:**

1. **Prompt injection:** Validar inputs del usuario antes de enviar al modelo
2. **Output filtering:** Filtrar respuestas que contengan contenido inapropiado
3. **Rate limiting:** Limitar requests por usuario para evitar abuso
4. **Model access:** API keys para acceso, no modelo público sin protección

**Ética y responsabilidad cultural:**

1. **Consultar comunidades mayas** sobre el uso de sus idiomas en tecnología
2. **No reemplazar traductores humanos** — posicionar como herramienta de apoyo
3. **Transparencia:** Indicar claramente que es traducción automática, no humana
4. **Bias testing:** Verificar que el modelo no perpetúe estereotipos

---

## Feedback Loop (Mejora Continua)

El ciclo que permite agregar data y mejorar:

```
Usuarios usan el producto
    → Califican traducciones (👍/👎)
    → Datos de feedback se acumulan
    → Cada mes: revisar feedback, identificar idiomas/frases problemáticas
    → Crear nuevos datos de entrenamiento corregidos
    → Re-entrenar LoRA adapter (incremental, no desde cero)
    → Test automático contra métricas
    → Deploy si mejora
    → Repetir
```

Esto es lo que hace que el modelo mejore según el uso, como mencionaste.

---

## Roadmap de Ejecución

### Semana 1-2: Data + Modelo Base
- [ ] Limpiar dataset (vacíos, splits)
- [ ] Buscar datos de Kaqchikel (MayanV, UVG, ALMG)
- [ ] Crear pipeline de ingestión de nuevos datos
- [ ] Primer fine-tuning con Unsloth + Qwen 2.5-7B en Colab

### Semana 3-4: RAG + Evaluación
- [ ] Procesar diccionarios y PDFs para RAG
- [ ] Implementar RAG con ChromaDB
- [ ] Crear test set manual (50 oraciones × 4 idiomas Tier 1)
- [ ] Evaluar BLEU/chrF por idioma
- [ ] Iterar prompts y re-entrenar si es necesario

### Semana 5-6: Deploy MVP
- [ ] Deploy en HuggingFace Space (Gradio)
- [ ] API con FastAPI + vLLM
- [ ] UI básica de chat
- [ ] Feedback collector (thumbs up/down)

### Semana 7-8: Producción
- [ ] Mover a GCP si el tráfico lo justifica
- [ ] Configurar CI/CD
- [ ] Monitoring y alertas
- [ ] Documentación para usuarios

### Ongoing: Mejora Continua
- [ ] Revisar feedback mensualmente
- [ ] Contactar UVG para datos nuevos
- [ ] Re-entrenar con nuevos datos
- [ ] Expandir idiomas según demanda

---

## Stack Tecnológico Final

| Componente | Herramienta | Alternativa |
|------------|-------------|-------------|
| Modelo base | Qwen 2.5-7B-Instruct | Mistral 7B / Llama 3.1 8B |
| Fine-tuning | Unsloth + QLoRA | HuggingFace TRL |
| Entrenamiento | Google Colab → GCP Vertex AI | HuggingFace Training |
| RAG vectores | ChromaDB | FAISS / Pinecone |
| RAG embeddings | paraphrase-multilingual-mpnet | E5-multilingual |
| Orchestración | LangChain | LlamaIndex |
| API | FastAPI + vLLM | HuggingFace Inference |
| UI | Gradio | Streamlit |
| Hosting modelo | HuggingFace Hub | GCP GCS |
| Local inference | Ollama (GGUF) | llama.cpp |
| CI/CD | GitHub Actions | GCP Cloud Build |
| Monitoring | LangSmith / W&B | Custom + GCP |
| Dataset versioning | DVC o HuggingFace Datasets | Manual (JSONL + git) |

---

## Fuentes de Investigación

- MayanV Dataset: https://github.com/transducens/mayanv
- AmericasNLP: https://github.com/AmericasNLP/americasnlp2021
- Unsloth: https://github.com/unslothai/unsloth
- Censo Guatemala 2018 (INE): https://censo2018.ine.gob.gt
- Lenguas de Guatemala: https://es.wikipedia.org/wiki/Lenguas_de_Guatemala
- Best LLMs for Translation 2026: https://www.hakunamatatatech.com/our-resources/blog/best-llm-for-translation

---

*Plan creado: 11 marzo 2026 | Proyecto MayaVoice LLMOps*
