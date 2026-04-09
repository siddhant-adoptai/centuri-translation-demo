"""
Form translator using Opus-MT (neural machine translation).

Drop-in replacement for ProjectA3's FormTranslator — same extract/reconstruct
pipeline, but uses Helsinki-NLP/opus-mt-en-es instead of LLM calls.

Advantages of dedicated NMT:
  - Purpose-built for translation (higher accuracy for EN↔ES)
  - Deterministic output (same input → same output)
  - Low latency: ~50ms per batch on CPU, ~5ms on GPU
  - Self-contained — no external API dependency
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from opus_mt_engine import translate_batch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirrored from ProjectA3/actionbot/operations/form_translator.py
# ---------------------------------------------------------------------------
TRANSLATABLE_KEYS: set[str] = {"label", "text", "notes"}

SKIP_KEYS: set[str] = {
    "id", "value", "type", "rules", "logic", "signatures", "media",
    "url", "buildTime", "buildError", "formId", "area", "workType",
    "status", "isReadOnly", "canCopy", "canClose", "canDelete",
    "selectedTraining", "trainingInfo",
}

CONTEXT_SKIP: dict[str, set[str]] = {
    "answers": {"text", "url"},
}


# ---------------------------------------------------------------------------
# Layer 1: Extraction (identical to production)
# ---------------------------------------------------------------------------
def extract_translatable_strings(
    payload: dict,
    translatable_keys: set[str] | None = None,
    skip_keys: set[str] | None = None,
    context_skip: dict[str, set[str]] | None = None,
) -> dict[str, str]:
    """Recursively walk JSON and extract translatable strings by dot-path."""
    if translatable_keys is None:
        translatable_keys = TRANSLATABLE_KEYS
    if skip_keys is None:
        skip_keys = SKIP_KEYS
    if context_skip is None:
        context_skip = CONTEXT_SKIP

    result: dict[str, str] = {}

    def _walk(obj: Any, path: str, parent_key: str = "") -> None:
        suppressed = context_skip.get(parent_key, set())

        if isinstance(obj, dict):
            for key, val in obj.items():
                if key in skip_keys:
                    continue
                child_path = f"{path}.{key}" if path else key
                if key in translatable_keys and key not in suppressed:
                    if isinstance(val, str) and val.strip():
                        result[child_path] = val
                else:
                    _walk(val, child_path, parent_key=key)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                _walk(item, f"{path}.{idx}", parent_key=parent_key)

    _walk(payload, "")
    return result


# ---------------------------------------------------------------------------
# Layer 1: Reconstruction (identical to production)
# ---------------------------------------------------------------------------
def _set_nested(obj: Any, parts: list[str], value: str) -> None:
    """Navigate into nested dict/list by path parts and set leaf value."""
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else obj[part]
    final = parts[-1]
    if final.isdigit():
        obj[int(final)] = value
    else:
        obj[final] = value


def reconstruct_payload(
    payload: dict, translations: dict[str, dict[str, str]]
) -> dict:
    """Apply translations back into a deep-copied payload."""
    result = copy.deepcopy(payload)
    for dot_path, mapping in translations.items():
        parts = dot_path.split(".")
        try:
            _set_nested(result, parts, mapping["translated"])
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to set translation at path '{dot_path}': {e}")
    return result


# ---------------------------------------------------------------------------
# Layer 2: NMT translation core (replaces LLM calls)
# ---------------------------------------------------------------------------
def translate_strings_nmt(
    pairs: dict[str, str],
    target_lang: str = "es",
    source_lang: str = "en",
) -> dict[str, dict[str, str]]:
    """Translate extracted string pairs using Opus-MT.

    Same interface as production translate_strings() but uses NMT instead of LLM.

    Args:
        pairs: {dot_path: source_string, ...}
        target_lang: ISO 639-1 target language.
        source_lang: ISO 639-1 source language.

    Returns:
        {dot_path: {"original": source_string, "translated": translated_string}, ...}
    """
    # Deduplicate
    unique_strings = list(set(pairs.values()))
    logger.info(f"NMT Translation: {len(pairs)} paths -> {len(unique_strings)} unique strings")

    # Translate via Opus-MT (single batch call, no prompt engineering needed)
    translated_list = translate_batch(unique_strings, source_lang, target_lang)
    nmt_map = dict(zip(unique_strings, translated_list))

    # Map back to dot-paths
    return {
        key: {"original": value, "translated": nmt_map.get(value, value)}
        for key, value in pairs.items()
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class FormTranslatorNMT:
    """Centuri form translator using Opus-MT.

    Drop-in compatible with ProjectA3's FormTranslator interface.

    Usage:
        translator = FormTranslatorNMT()
        result = translator.translate_form(form_payload, target_lang="es")
    """

    def __init__(self):
        # No config_instance needed — Opus-MT is self-contained
        pass

    def translate_form(
        self, payload: dict, target_lang: str = "es", source_lang: str = "en"
    ) -> dict:
        """Translate a Centuri form payload. Returns translated deep copy."""
        pairs = extract_translatable_strings(payload)
        if not pairs:
            return copy.deepcopy(payload)

        translations = translate_strings_nmt(pairs, target_lang, source_lang)
        return reconstruct_payload(payload, translations)

    def translate_form_with_metadata(
        self, payload: dict, target_lang: str = "es", source_lang: str = "en"
    ) -> tuple[dict, dict[str, dict[str, str]]]:
        """Translate form and return (translated_payload, translation_map)."""
        pairs = extract_translatable_strings(payload)
        if not pairs:
            return copy.deepcopy(payload), {}

        translations = translate_strings_nmt(pairs, target_lang, source_lang)
        return reconstruct_payload(payload, translations), translations
