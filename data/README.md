# Data Directory

Los datos no se suben a git por su tamaño. Para configurar el directorio de datos localmente:

## Estructura esperada

```
data/
├── textos-paralelos/     # 12 idiomas × ~1,000 pares cada uno
│   ├── kiche/
│   │   ├── data.es       # Español
│   │   └── data.quc      # K'iche'
│   ├── qeqchi/
│   ├── mam/
│   └── ...
├── diccionarios/
│   ├── csv/              # 25 diccionarios individuales + master
│   └── pdf/              # 4 vocabularios en PDF
├── entrenamiento/
│   ├── training_data_v2.jsonl  # ~43,900 ejemplos (Alpaca format)
│   ├── train.jsonl
│   ├── val.jsonl
│   └── metadata.json
├── corpus-rag/           # 22 corpus .txt por idioma
├── audio/                # Metadata CSV (22,617 grabaciones)
└── documentos-referencia/ # 9 PDFs y DOCXs
```

## Fuentes

- **Textos paralelos:** [MayanV](https://github.com/transducens/mayanv) (CC0)
- **Diccionarios:** [Swarthmore Talking Dictionary](https://talkingdictionary.swarthmore.edu/)
- **Documentos:** ALMG, AmericasNLP 2021

## Setup

Copiar los datos desde la carpeta local del proyecto:

```bash
# Desde la carpeta "Mayan GPT/data/" copiar a este directorio
cp -r /path/to/Mayan\ GPT/data/* ./data/
```
