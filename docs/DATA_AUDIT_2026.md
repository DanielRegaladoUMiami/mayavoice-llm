# Auditoría de Datos — Mayan GPT

**Fecha:** 11 marzo 2026
**Proyecto:** Traducción Español ↔ Lenguas Mayas
**Autor:** Daniel Regalado Cardoso

---

## Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Lenguas mayas cubiertas | 22 |
| Pares paralelos usables | 12,000 |
| Ejemplos en JSONL v2 | ~43,900 |
| Diccionarios CSV | 25 archivos |
| Corpus RAG | 37 KB (muy pequeño) |
| Grabaciones de audio | 22,617 |

---

## 1. Textos Paralelos (ES ↔ Maya)

12 idiomas con 1,000 pares cada uno + 1 carpeta "unknown" sin par maya.
**Fuente:** MayanV (AmericasNLP 2021)

| Idioma | Ext | Pares | Alineado | ES vacías | Maya vacías | ES dupes | ES avg (chars) | Maya avg |
|--------|-----|-------|----------|-----------|-------------|----------|----------------|----------|
| Sipakapense | .qum | 1,000 | ✅ | 0 | 0 | 2 | 42 | 40 |
| Mam | .mam | 1,000 | ✅ | 0 | 0 | 2 | 41 | 30 |
| Tektiteko | .ttc | 1,000 | ✅ | 0 | 0 | 4 | 39 | 32 |
| Q'anjob'al | .kjb | 1,000 | ✅ | 0 | 0 | 2 | 38 | 34 |
| K'iche' | .quc | 1,000 | ✅ | 0 | 0 | 0 | 37 | 29 |
| Poqomam | .poc | 1,000 | ✅ | 0 | 0 | 2 | 37 | 29 |
| Q'eqchi' | .kek | 1,000 | ✅ | 0 | 0 | 4 | 33 | 26 |
| Chuj | .cac | 1,000 | ✅ | 0 | 0 | 1 | 31 | 25 |
| Tz'utujil | .tzj | 1,000 | ✅ | 0 | 0 | 0 | 31 | 29 |
| Awakateko | .agu | 1,000 | ✅ | 0 | **49** | 55 | 31 | 24 |
| Itza' | .itz | 1,000 | ✅ | 0 | **35** | 53 | 30 | 26 |
| Poqomchi' | .poh | 1,000 | ✅ | 0 | 0 | 4 | 29 | 22 |
| Unknown | — | 1,000 | ❌ | 0 | — | 38 | 36 | — |

**Hallazgos clave:**

- Todos los 12 idiomas están perfectamente alineados línea por línea (✅)
- 0 pares idénticos ES=Maya en ningún idioma (no hay copias accidentales)
- K'iche' y Tz'utujil son los más limpios (0 duplicados, 0 vacíos)
- Longitud promedio razonable: ~30-42 chars ES, ~22-40 chars Maya

### Muestras de traducción

```
ES: Tienes indigestión
K'iche': Kq'ulq'ut ri apam

ES: La tierra es sagrada
Poqomchi': Looq' i ak'al

ES: Le robaron la gallina de María
Q'anjob'al: Max toj Skaxhlan ix Malin yuj elq'om

ES: Nuestro perro es miedoso
Q'eqchi': Aj xiw li qatz'i'
```

### Problemas detectados

- **Awakateko:** 49 líneas maya vacías + 55 ES duplicadas — posible problema de scraping
- **Itza':** 35 líneas maya vacías + 53 ES duplicadas — mismo patrón
- **Unknown:** 1,000 líneas ES sin contraparte maya — inutilizable, descartar
- **Prefijo `#lang#` en ES:** Todas las líneas ES empiezan con `#quc#`, `#mam#`, etc. Hay que decidir si mantenerlo (útil para modelo multilingüe) o limpiarlo

---

## 2. Datos de Entrenamiento (JSONL)

Dataset principal: `training_data_v2.jsonl`

| Propiedad | Valor |
|-----------|-------|
| Ejemplos totales | ~43,911 |
| Tamaño | 5.6 MB |
| Formato | Alpaca (instruction/input/output) |
| Composición | ~17,686 traducciones MayanV + ~26,225 diccionario |

**Nota:** Los archivos JSONL tienen error "Resource deadlock avoided" en esta VM y no se pudieron analizar en detalle. Se recomienda validar en Colab o localmente.

---

## 3. Diccionarios CSV

25 archivos individuales por idioma + 3 archivos master.
**Fuente:** Swarthmore Talking Dictionary

- `mayavoice_dictionary_full.csv` — 2.7 MB, ~9,000+ entradas, 22 idiomas
- Archivos individuales: achi, akateko, awakateko, chalchiteko, chorti, chuj, itza, ixil, kaqchikel, kiche, mam, mopan, ngabere, poqomam, poqomchi, qanjobal, qeqchi, sakapulteko, sipakapense, tektiteko, tzutujil, uspanteko

**Estado:** ⚠️ No accesibles en esta VM (mismo error de filesystem). Ya integrados en `training_data_v2.jsonl`.

---

## 4. Corpus RAG

22 archivos .txt en `05_rag_knowledge_base/embeddings_ready/`. Cada uno tiene exactamente 100 líneas.

| Métrica | Valor |
|---------|-------|
| Tamaño total | 37 KB |
| Líneas totales | 2,200 |
| Idiomas | 22 |
| Contenido | Palabras sueltas del diccionario |

**Problema crítico:** 37 KB es completamente insuficiente para un RAG efectivo. Son listas de palabras, no textos contextuales. Se necesitan textos largos y variados para que el retrieval tenga sentido.

---

## 5. Datos de Audio

| Propiedad | Valor |
|-----------|-------|
| Grabaciones | 22,617 MP3 |
| Tamaño estimado | ~2-5 GB |
| Metadata | `mayavoice_audio_dataset.csv` |
| Ubicación | `MayaVoice/data/audio/` (no movido) |

Prioridad baja — enfoque actual es texto.

---

## 6. Diagnóstico General

### Fortalezas

- 12 idiomas con textos paralelos alineados y limpios
- Dataset procesado de ~43,900 ejemplos en formato Alpaca listo para fine-tuning
- Diccionarios cubriendo 22 idiomas (~9K entradas)
- UTF-8 correcto con caracteres especiales mayas (ʼ, ä, etc.)
- 0 pares idénticos ES=Maya (no hay copias accidentales)
- Longitud promedio razonable (~30-40 chars ES, ~25-40 chars Maya)

### Debilidades y Riesgos

- **Solo ~1,000 pares por idioma** — ideal sería 5,000-10,000+ para buen desempeño
- **Corpus RAG insignificante** — 37 KB no es útil para retrieval
- **Awakateko/Itza' tienen datos con vacíos** — 49 y 35 líneas maya en blanco
- **Sin test/validation split formal** — no hay forma de evaluar el modelo objetivamente
- **Dominio limitado** — oraciones cortas cotidianas/educativas, falta diversidad
- **Archivos CSV/JSONL inaccesibles en VM** — impide validación profunda en esta sesión

---

## 7. Recomendaciones Inmediatas

| # | Acción | Impacto | Esfuerzo |
|---|--------|---------|----------|
| 1 | Limpiar líneas vacías en Awakateko (49) e Itza' (35) | Alto | 5 min |
| 2 | Eliminar carpeta "unknown" (sin datos maya) | Bajo | 1 min |
| 3 | Crear split train/val/test (90/5/5) del JSONL | Alto | 10 min |
| 4 | Decidir sobre prefijo `#lang#` en ES — ¿mantener o limpiar? | Alto | 5 min |
| 5 | Expandir corpus RAG con textos de los PDFs de referencia | Medio | 30 min |
| 6 | Validar JSONL en Colab (no accesible en VM actual) | Alto | 15 min |

---

*Auditoría generada — Mayan GPT Project | 11 marzo 2026*
