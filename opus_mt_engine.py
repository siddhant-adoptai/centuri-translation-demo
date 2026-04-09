"""
Opus-MT translation engine — Helsinki-NLP/opus-mt-en-es.

Dedicated neural machine translation for EN → ES (and reverse).
Singleton pattern: model loads once, reused across calls.

Compatible with ProjectA3's FormTranslator interface — can be swapped
into _call_llm_translate as a drop-in replacement.
"""

import logging
from typing import Optional

from transformers import MarianMTModel, MarianTokenizer

logger = logging.getLogger(__name__)

# Model registry: (source, target) → HuggingFace model name
MODEL_REGISTRY: dict[tuple[str, str], str] = {
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
}

# Singleton cache: model_name → (model, tokenizer)
_loaded_models: dict[str, tuple[MarianMTModel, MarianTokenizer]] = {}


def _get_model(source_lang: str, target_lang: str) -> tuple[MarianMTModel, MarianTokenizer]:
    """Load model + tokenizer, cached as singleton."""
    key = (source_lang, target_lang)
    model_name = MODEL_REGISTRY.get(key)
    if model_name is None:
        raise ValueError(
            f"No Opus-MT model registered for {source_lang} → {target_lang}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    if model_name not in _loaded_models:
        logger.info(f"Loading Opus-MT model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _loaded_models[model_name] = (model, tokenizer)
        logger.info(f"Opus-MT model loaded: {model_name}")

    return _loaded_models[model_name]


def translate_batch(
    strings: list[str],
    source_lang: str = "en",
    target_lang: str = "es",
    batch_size: int = 64,
) -> list[str]:
    """Translate a list of strings using Opus-MT.

    Args:
        strings: Source strings to translate.
        source_lang: ISO 639-1 source language.
        target_lang: ISO 639-1 target language.
        batch_size: Max strings per forward pass (GPU/CPU chunking).

    Returns:
        List of translated strings, same order and length as input.
    """
    if not strings:
        return []

    model, tokenizer = _get_model(source_lang, target_lang)
    all_translated: list[str] = []

    for start in range(0, len(strings), batch_size):
        batch = strings[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_translated.extend(decoded)

    return all_translated


def translate_map(
    source_map: dict[str, str],
    source_lang: str = "en",
    target_lang: str = "es",
) -> dict[str, str]:
    """Translate a {key: source_string} map, returning {source_string: translated_string}.

    This matches the interface of FormTranslator's _call_llm_translate:
    takes unique strings, returns a source→translated dict.
    """
    unique_strings = list(set(source_map.values()))
    translated = translate_batch(unique_strings, source_lang, target_lang)
    return dict(zip(unique_strings, translated))


def get_supported_pairs() -> list[tuple[str, str]]:
    """Return list of supported (source, target) language pairs."""
    return list(MODEL_REGISTRY.keys())
