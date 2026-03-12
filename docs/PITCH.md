# MayaVoice — Pitch para Inversionistas y Grants

## El Problema

Guatemala tiene 22 lenguas mayas habladas por más de 4 millones de personas (~25% de la población). Estas lenguas están subrepresentadas en la tecnología: Google Translate solo soporta 3 de ellas (K'iche', Q'eqchi', Mam) desde 2024, y no existe ninguna herramienta de IA que cubra las 22. La brecha digital lingüística excluye a millones de personas del acceso a información, servicios de salud, educación y gobierno en su idioma materno.

## La Solución

MayaVoice es un modelo de IA especializado en traducción y conversación en lenguas mayas de Guatemala. Utilizamos fine-tuning de modelos de lenguaje grandes (Llama 3.1-8B) con datos paralelos recopilados de fuentes académicas, bíblicas y diccionarios lingüísticos.

### Lo que ya funciona

Tenemos un primer modelo entrenado (v1) que traduce entre español y 13 lenguas mayas con resultados medibles. Ejemplos reales del modelo:

- "Buenos días, ¿cómo estás?" → K'iche': "Utz q'ij, jachin k'o chi awetaq"
- "La tierra es sagrada para nosotros." → Mam: "Qa'nxa tx'otx' ti'j qe."
- "Los niños están jugando en el campo." → Q'eqchi': "Yookeb' chi b'atz'ok sa' li k'al li al"

### Lo que estamos construyendo (v2)

Un modelo mejorado con 3.7x más datos de entrenamiento, 14 idiomas (agregamos Kaqchikel, el tercer idioma maya más hablado con ~1M de hablantes), y técnicas de augmentación de datos. Actualmente en entrenamiento.

## Mercado

### Tamaño

Más de 4 millones de hablantes de lenguas mayas solo en Guatemala, con comunidades adicionales en México, Belice, Honduras y la diáspora en EE.UU. (~1.5M guatemaltecos en EE.UU.).

### Segmentos de cliente

**Gobierno y sector público:** El Ministerio de Educación de Guatemala, ALMG (Academia de Lenguas Mayas), ministerios de salud y justicia requieren comunicación en lenguas mayas por mandato constitucional. Actualmente dependen de traductores humanos escasos y costosos.

**ONGs y cooperación internacional:** USAID, UNESCO, UNICEF y decenas de ONGs trabajan con comunidades maya-hablantes y necesitan herramientas de traducción para materiales educativos, de salud y derechos humanos.

**Sector educativo:** Universidades (UVG, USAC, URL) y escuelas bilingües necesitan herramientas para enseñanza de y en lenguas mayas.

**Sector privado:** Empresas que operan en áreas rurales de Guatemala (telecomunicaciones, banca, salud) necesitan atención al cliente en lenguas mayas.

## Diferenciación

### vs. Google Translate (K'iche', Q'eqchi', Mam)
MayaVoice cubre 14 idiomas vs. 3. Además, nuestro modelo es open-source y puede desplegarse localmente (sin internet), crítico para comunidades rurales con conectividad limitada.

### vs. Traductores humanos
MayaVoice no reemplaza traductores — los complementa. Ofrece traducciones instantáneas para comunicación básica y permite a traductores profesionales enfocarse en contenido de alta complejidad.

### vs. Otros proyectos académicos
MayaVoice es el primer sistema que combina fine-tuning de LLM con RAG para lenguas mayas, y el primero con datos de diccionarios con audio para futuro soporte de voz.

## Tracción

- Modelo v1 entrenado y funcional, publicado en HuggingFace
- 120,656 pares paralelos de entrenamiento (el dataset más grande conocido para lenguas mayas de Guatemala)
- 5,224 entradas de diccionario con audio (22 idiomas)
- Pipeline reproducible y código abierto en GitHub
- Modelo v2 (3.7x más datos) en entrenamiento

## Equipo

Daniel Regalado Cardoso — University of Miami. Desarrollador del pipeline completo de datos y entrenamiento.

## Financiamiento Solicitado

### Para qué se usaría

**Infraestructura de cómputo ($5K-10K):** GPUs para entrenamiento iterativo de modelos más grandes y experimentación con arquitecturas alternativas.

**Recopilación de datos ($10K-20K):** Trabajar con hablantes nativos y lingüistas para crear un test set de evaluación humana, y recopilar datos paralelos adicionales en idiomas con poca cobertura (Itza', Ixil, Mopan).

**Despliegue ($5K):** Servidor con GPU para API pública, dominio, y desarrollo de app móvil básica.

### Fuentes de financiamiento alineadas

- Google.org Impact Challenge (AI for social good)
- Mozilla Common Voice / Mozilla Foundation
- UNESCO ICT in Education Prize
- USAID Guatemala Digital Development
- Lacuna Fund (datasets para lenguas de bajos recursos)
- AI2 (Allen Institute for AI) grants para NLP de bajos recursos
- National Science Foundation (NSF) — SaTC o NRI programs

## Roadmap

| Trimestre | Milestone |
|-----------|-----------|
| Q1 2026 | ✅ Modelo v1 entrenado, métricas baseline |
| Q1 2026 | 🔄 Modelo v2 con datos expandidos |
| Q2 2026 | Demo pública (Gradio/HuggingFace Spaces) |
| Q2 2026 | RAG pipeline con diccionarios + audio |
| Q3 2026 | Evaluación con hablantes nativos (UVG) |
| Q3 2026 | API pública + app móvil MVP |
| Q4 2026 | Piloto con institución guatemalteca |

## Contacto

Daniel Regalado Cardoso
dxr1491@miami.edu
University of Miami

GitHub: [DanielRegaladoUMiami/mayavoice-llm](https://github.com/DanielRegaladoUMiami/mayavoice-llm)
HuggingFace: [DanielRegaladoCardoso/mayavoice-llama3.1-8b-lora](https://huggingface.co/DanielRegaladoCardoso/mayavoice-llama3.1-8b-lora)
