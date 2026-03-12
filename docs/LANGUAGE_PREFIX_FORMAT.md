# Formato de Prefijo de Idioma — Decision Record

## Issue #3: Decidir sobre prefijo de idioma en oraciones

### Decisión

Usamos **tags de idioma en la instrucción** del formato Alpaca, NO en el texto fuente/target.

### Formato elegido

```json
{
  "instruction": "Traduce del español al K'iche'.",
  "input": "Buenos días, ¿cómo estás?",
  "output": "Saqarik, ¿la utz awach?"
}
```

Para chat en Maya:
```json
{
  "instruction": "Responde en K'iche' como un asistente amigable.",
  "input": "¿Qué significa 'saqarik'?",
  "output": "Saqarik kub'ij 'buenos días' pa K'iche'. Are jun tzij ri kukiya' le qach'ab'äl."
}
```

### Alternativas consideradas

| Opción | Ejemplo | Pros | Contras |
|--------|---------|------|---------|
| **A) Tag en instrucción** ✅ | `"instruction": "Traduce del español al K'iche'."` | Natural, flexible, fácil de entender | Instrucciones más largas |
| B) Prefijo inline `[ES>QUC]` | `"input": "[ES>QUC] Buenos días"` | Compacto | Contamina el texto, confunde al modelo |
| C) Special tokens `<lang:quc>` | `"input": "<lang:quc> Buenos días"` | Programático | Requiere expandir tokenizer, arriesgado con QLoRA |
| D) Campo separado | `"lang_pair": "es-quc"` | Limpio | No es Alpaca estándar, Unsloth no lo soporta |

### Justificación

1. **Compatibilidad con Unsloth**: El formato Alpaca con instruction/input/output es el estándar que Unsloth optimiza.
2. **Sin contaminar texto**: Los textos fuente y target quedan limpios, mejorando la calidad de traducción.
3. **Flexibilidad**: Permite múltiples tipos de tarea (traducción ES→Maya, Maya→ES, chat en Maya) solo cambiando la instrucción.
4. **Escalable**: Agregar un nuevo idioma solo requiere nuevas instrucciones, no cambios en tokenizer.

### Nombres estándar de idiomas

| Idioma | Nombre en instrucción | ISO 639-3 |
|--------|----------------------|-----------|
| Español | español | spa |
| K'iche' | K'iche' | quc |
| Q'eqchi' | Q'eqchi' | kek |
| Kaqchikel | Kaqchikel | cak |
| Mam | Mam | mam |
| Tz'utujil | Tz'utujil | tzj |
| Poqomchi' | Poqomchi' | poh |
| Poqomam | Poqomam | poc |
| Q'anjob'al | Q'anjob'al | kjb |
| Chuj | Chuj | cac |
| Awakateko | Awakateko | agu |
| Itza' | Itza' | itz |
| Sipakapense | Sipakapense | qum |
| Tektiteko | Tektiteko | ttc |
| Achi | Achi | acr |
| Ixil | Ixil | ixl |

### Templates de instrucción

```python
INSTRUCTION_TEMPLATES = {
    'translate_es_to_maya': "Traduce del español al {lang}.",
    'translate_maya_to_es': "Traduce del {lang} al español.",
    'chat_maya': "Responde en {lang} como un asistente amigable.",
    'explain_linguistic': "Explica el significado lingüístico de la siguiente expresión en {lang}.",
}
```
