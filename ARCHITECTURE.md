# Centuri Form Translation — Architecture

## Overview

Neural machine translation for Centuri safety forms (EN ↔ ES) using Helsinki-NLP's Opus-MT model. The translation engine extracts translatable strings from structured form JSON, translates via a dedicated NMT model, and reconstructs the payload — preserving the original schema exactly.

This demo module is designed for minimal integration into the existing Adopt AI stack (ProjectA3 + adoptwebui).

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              CENTURI .NET BACKEND                               │
│                                                                                  │
│  Tablet UI: user taps EN/ES toggle → .NET fires webhook POST                    │
└──────────────────────────┬───────────────────────────────────────────────────────┘
                           │
                           │  POST /webhook/{org_id}/action/{action_id}/translate
                           │  Header: X-Webhook-Secret
                           │  Body: { form_payload, target_lang, source_lang, callback_url }
                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         ADOPT WEB UI  (FastAPI)                                  │
│                         adoptwebui/backend/app/routes/webhook.py                 │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐                │
│  │  /translate endpoint                                        │                │
│  │  • Auth validation (X-Webhook-Secret)                       │                │
│  │  • Rate limiting (per IP + org + action)                    │                │
│  │  • Language pair validation                                 │                │
│  │  • Returns 200 accepted immediately                         │                │
│  │  • Queues background task → POST to reasoning workflow API  │                │
│  └─────────────────────────────┬───────────────────────────────┘                │
│                                │                                                 │
└────────────────────────────────┼─────────────────────────────────────────────────┘
                                 │
                                 │  POST {reasoning_workflow_url}/api/v1/workflows/execute-analysis
                                 │  Body: { workflow_name, task_queue_name,
                                 │          arguments: [{ user_params, security_headers,
                                 │                        metadata: { operation: "TRANSLATE",
                                 │                                    form_payload, translate_step,
                                 │                                    callback_url } }] }
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         PROJECT A3  (Workflow Engine)                             │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────┐                    │
│  │  WDL Executor (widdle_executor.py)                      │                    │
│  │  • Dispatches TRANSLATE operation via SINGLE_OP_DISPATCH│                    │
│  │  • Calls _execute_translate(translate_step)             │                    │
│  └──────────────────────┬──────────────────────────────────┘                    │
│                         │                                                        │
│  ┌──────────────────────▼──────────────────────────────────┐                    │
│  │  TranslateOperation (operations/translate.py)           │                    │
│  │  • Reads form_payload from metadata                     │                    │
│  │  • Delegates to FormTranslator / FormTranslatorNMT      │                    │
│  └──────────────────────┬──────────────────────────────────┘                    │
│                         │                                                        │
│  ┌──────────────────────▼──────────────────────────────────┐                    │
│  │  Translation Pipeline                                    │                    │
│  │                                                          │                    │
│  │  1. extract_translatable_strings(payload)                │                    │
│  │     └─ Recursive JSON walk: label, text, notes           │                    │
│  │     └─ Context-aware skipping (answers.text, etc.)       │                    │
│  │     └─ Returns {dot_path: string, ...}                   │                    │
│  │                                                          │                    │
│  │  2. Deduplicate unique strings                           │                    │
│  │                                                          │                    │
│  │  3. Translate (engine-dependent)                         │                    │
│  │     ├─ [NMT] Opus-MT: translate_batch() — MarianMT      │                    │
│  │     └─ [LLM] _call_llm_translate() — batched prompts    │                    │
│  │                                                          │                    │
│  │  4. reconstruct_payload(original, translations)          │                    │
│  │     └─ Deep copy + inject via dot-path                   │                    │
│  └──────────────────────┬──────────────────────────────────┘                    │
│                         │                                                        │
│  ┌──────────────────────▼──────────────────────────────────┐                    │
│  │  TranslationCache (optional, two-tier)                   │                    │
│  │  • Tier 1: In-memory dict (SHA256 key)                   │                    │
│  │  • Tier 2: SingleStore (form_translation_cache table)    │                    │
│  └─────────────────────────────────────────────────────────┘                    │
│                                                                                  │
└──────────────────────────────┬───────────────────────────────────────────────────┘
                               │
                               │  POST callback_url
                               │  Body: { translated_payload, translation_map }
                               ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         CENTURI .NET BACKEND                                     │
│  Receives translated form JSON → renders Spanish form on tablet                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Translation Engine: Opus-MT

| Property | Detail |
|----------|--------|
| Model | `Helsinki-NLP/opus-mt-en-es` (EN→ES), `opus-mt-es-en` (ES→EN) |
| Framework | MarianMT via HuggingFace Transformers |
| Architecture | Transformer encoder-decoder, trained on OPUS parallel corpus |
| Model size | ~298MB |
| Inference | CPU: ~37ms/string, GPU: ~5ms/string |
| Determinism | Same input always produces same output |

**Why dedicated NMT over general-purpose LLM for this use case:**
- Translation is a well-defined task — NMT models are trained specifically for it
- Consistent output (no prompt sensitivity or temperature variance)
- Lower latency per string at high volume
- Self-contained runtime — no external API dependency during inference

---

## Module Structure

```
centuri_translation_demo/
├── ARCHITECTURE.md              ← this file
├── requirements.txt             ← transformers, torch, sentencepiece, sacremoses, streamlit
├── opus_mt_engine.py            ← Opus-MT model loader + translate_batch()
├── form_translator_nmt.py       ← FormTranslatorNMT (extract → NMT → reconstruct)
├── app.py                       ← Streamlit demo UI (paste JSON → toggle → translated JSON)
├── webhook_server.py            ← FastAPI demo server (same endpoint contract)
├── demo.py                      ← CLI: translates sample forms with timing
├── form_JHA1_translated_es.json ← sample output
├── from_JHA2_translated_es.json ← sample output
└── form_BORE1_translated_es.json← sample output
```

---

## Integration Path into Production

### What already exists

| Component | Location | Status |
|-----------|----------|--------|
| `FormTranslator` (extract, reconstruct, cache) | `ProjectA3/actionbot/operations/form_translator.py` | Production |
| `TranslationCache` (memory + SingleStore) | `ProjectA3/actionbot/operations/form_translator.py` | Production |
| Webhook auth, rate limiting, normalization | `adoptwebui/backend/app/routes/webhook.py` | Production |
| `/translate` webhook endpoint | `adoptwebui/backend/app/routes/webhook.py` | Built |
| Centuri form samples (JHA1, JHA2, BORE1) | Project root | Available |

### What this demo adds

| Component | Location | Integration effort |
|-----------|----------|--------------------|
| `OpusMTEngine` (model singleton + batch translate) | `opus_mt_engine.py` | Copy to `ProjectA3/actionbot/operations/` |
| `FormTranslatorNMT` (NMT-backed translator) | `form_translator_nmt.py` | Merge into `form_translator.py` or keep separate |

### Minimal integration steps (3 files changed in ProjectA3)

**1. Add Opus-MT engine** — copy `opus_mt_engine.py` into `ProjectA3/actionbot/operations/`

**2. Wire into FormTranslator** — add engine selection to existing `translate_strings()`:

```python
# In form_translator.py — translate_strings()
def translate_strings(pairs, target_lang, llm_instance=None, source_lang="en", engine="nmt"):
    unique_strings = list(set(pairs.values()))

    if engine == "nmt":
        from .opus_mt_engine import translate_batch
        translated_list = translate_batch(unique_strings, source_lang, target_lang)
        translated_map = dict(zip(unique_strings, translated_list))
    else:
        translated_map = _call_llm_translate(unique_strings, source_lang, target_lang, llm_instance)

    return {
        key: {"original": value, "translated": translated_map.get(value, value)}
        for key, value in pairs.items()
    }
```

**3. Register TRANSLATE operation** — 3 one-line changes:

```
widdle_model.py:     TRANSLATE = ('TRANSLATE', ExecutionBlockType.DATA_PROCESSING)
widdle_executor.py:  SINGLE_OP_DISPATCH[WorkflowOperationType.TRANSLATE] = ("_execute_translate", None)
widdle_executor.py:  elif operation == WorkflowOperationType.TRANSLATE:
                         (result, self.error) = self._execute_translate(widdle_step)
```

**4. Dependencies** — add to `ProjectA3/pyproject.toml`:

```toml
transformers = ">=4.40.0"
sentencepiece = ">=0.2.0"
torch = ">=2.2.0"
sacremoses = ">=0.1.1"
```

---

## Data Flow Detail

### Extraction (schema-agnostic)

```
form JSON
├── formId, area, workType, status, isReadOnly    ← SKIPPED (metadata)
└── sections[]
    ├── id, type, buildTime, canClose              ← SKIPPED (system keys)
    ├── label                                      ← TRANSLATED
    └── fields[]
        ├── id, type, value, rules, logic           ← SKIPPED
        ├── label                                   ← TRANSLATED
        └── options[]
            ├── value                               ← SKIPPED (machine code: "YES", "NE")
            └── text                                ← TRANSLATED (human label: "North" → "Norte")
```

### Translation map format

```json
{
  "sections.0.label": {
    "original": "Header",
    "translated": "Encabezado"
  },
  "sections.0.fields.3.label": {
    "original": "Safety Observer",
    "translated": "Observador de seguridad"
  }
}
```

The translation map is returned alongside the translated payload, enabling:
- Audit trail (what changed, original vs translated)
- Selective revert (undo individual translations)
- Cache seeding (pre-populate cache from approved translations)

---

## Webhook API Contract

### Request

```
POST /webhook/{org_id}/action/{action_id}/translate
Header: X-Webhook-Secret: <secret>
Content-Type: application/json
```

```json
{
  "form_payload": { "sections": [{ "label": "Safety Header", "fields": [...] }] },
  "target_lang": "es",
  "source_lang": "en",
  "callback_url": "https://centuri.example.com/api/translation-result"
}
```

### Response (immediate)

```json
{ "status": "accepted", "message": "Translation queued (en → es)" }
```

### Callback (async, to Centuri)

```json
{
  "translated_payload": { "sections": [{ "label": "Encabezado de Seguridad", "fields": [...] }] },
  "translation_map": { "sections.0.label": { "original": "Safety Header", "translated": "Encabezado de Seguridad" } }
}
```

---

## Engine Comparison

| Property | Opus-MT (NMT) | LLM (current) |
|----------|---------------|----------------|
| Translation accuracy (general) | High (trained on parallel corpus) | High (broad knowledge) |
| Domain-specific terms | Needs glossary override | Can be prompted with context |
| Latency per string | ~37ms (CPU) | ~200-500ms (API round-trip) |
| Determinism | Yes | No (temperature-dependent) |
| Batch efficiency | Native batching | Prompt-based batching (60/call) |
| Infrastructure | Model loaded in-process | External API call |
| Supported languages | Per model pair (EN↔ES available) | Any language |

**Recommendation:** Use Opus-MT as the primary engine for EN↔ES (the Centuri use case). Retain LLM as fallback for unsupported language pairs or domain-specific edge cases that require contextual understanding.

---

## Known Limitations & Mitigations

| Limitation | Example | Mitigation |
|-----------|---------|------------|
| Domain-specific terms | "Bore" → "aburrimiento" (boredom) instead of "perforación" (drilling) | Glossary override layer (Centuri to provide safety/utility glossary) |
| Compound proper nouns | WBS codes get minor formatting changes | Add to SKIP_KEYS or pre-filter regex patterns |
| Model startup time | ~20s cold start | Pre-load at server startup (singleton pattern) |
| Language pair coverage | Only EN↔ES models registered | Add model pairs to MODEL_REGISTRY as needed |

---

## Running the Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Streamlit UI (recommended for demo)
streamlit run app.py

# CLI — translate all sample forms
python demo.py --side-by-side

# Webhook server — same endpoint contract as production
uvicorn webhook_server:app --port 8100

# Test webhook
curl -X POST http://localhost:8100/webhook/centuri/action/jha/translate \
  -H "X-Webhook-Secret: demo-secret" \
  -H "Content-Type: application/json" \
  -d '{"form_payload": '"$(cat ../form_JHA1.json)"', "target_lang": "es"}'
```
