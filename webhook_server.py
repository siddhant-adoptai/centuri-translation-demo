"""
Demo webhook server — simulates the production /translate endpoint.

Standalone FastAPI server that accepts the same payload format as the
production webhook and returns translated forms using Opus-MT.

Unlike production (which queues + callbacks), this returns the translation
synchronously for demo simplicity.

Usage:
    uvicorn webhook_server:app --port 8100 --reload

    # Then test:
    curl -X POST http://localhost:8100/webhook/demo-org/action/demo-action/translate \
      -H "Content-Type: application/json" \
      -H "X-Webhook-Secret: demo-secret" \
      -d @../form_JHA1.json
"""

import os
import sys
import time
import logging

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from form_translator_nmt import FormTranslatorNMT, extract_translatable_strings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Centuri Translation Demo — Opus-MT",
    description="POC webhook for EN↔ES form translation using Helsinki-NLP/opus-mt-en-es",
    version="0.1.0",
)

# Singleton translator — model loads once at startup
_translator: Optional[FormTranslatorNMT] = None


def get_translator() -> FormTranslatorNMT:
    global _translator
    if _translator is None:
        logger.info("Initializing Opus-MT translator...")
        _translator = FormTranslatorNMT()
        # Warm up
        _translator.translate_form({"sections": [{"label": "warmup"}]})
        logger.info("Opus-MT translator ready")
    return _translator


# ---------------------------------------------------------------------------
# Request/Response models (match production webhook.py contract)
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    form_payload: dict[str, Any]
    target_lang: str = "es"
    source_lang: str = "en"
    callback_url: Optional[str] = None


class TranslateResponse(BaseModel):
    status: str
    translated_payload: dict[str, Any]
    translation_map: dict[str, dict[str, str]]
    stats: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
DEMO_SECRET = os.environ.get("WEBHOOK_SECRET", "demo-secret")
SUPPORTED_LANGS = {"en", "es"}


@app.on_event("startup")
async def startup_load_model():
    """Pre-load model at server startup so first request is fast."""
    get_translator()


@app.post(
    "/webhook/{org_id}/action/{action_id}/translate",
    response_model=TranslateResponse,
)
async def translate_form(
    org_id: str,
    action_id: str,
    body: TranslateRequest,
    x_webhook_secret: str = Header(alias="X-Webhook-Secret"),
):
    """Translate a Centuri form payload using Opus-MT.

    Matches the production webhook endpoint contract:
      POST /webhook/{org_id}/action/{action_id}/translate

    Demo mode: returns translation synchronously (production uses async callback).
    """
    # Auth check
    if x_webhook_secret != DEMO_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    # Language validation
    if body.source_lang not in SUPPORTED_LANGS or body.target_lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language pair: {body.source_lang} → {body.target_lang}. "
                   f"Supported: {SUPPORTED_LANGS}",
        )
    if body.source_lang == body.target_lang:
        raise HTTPException(status_code=400, detail="Source and target language must differ")

    # Translate
    translator = get_translator()
    start = time.time()

    pairs = extract_translatable_strings(body.form_payload)
    translated_payload, translation_map = translator.translate_form_with_metadata(
        body.form_payload, body.target_lang, body.source_lang
    )

    elapsed = time.time() - start

    return TranslateResponse(
        status="translated",
        translated_payload=translated_payload,
        translation_map=translation_map,
        stats={
            "org_id": org_id,
            "action_id": action_id,
            "source_lang": body.source_lang,
            "target_lang": body.target_lang,
            "total_paths": len(pairs),
            "unique_strings": len(set(pairs.values())),
            "translation_time_ms": round(elapsed * 1000),
            "engine": "opus-mt-en-es (Helsinki-NLP)",
        },
    )


@app.post("/webhook/{org_id}/action/{action_id}/translate-raw")
async def translate_raw_form(
    org_id: str,
    action_id: str,
    request: Request,
    x_webhook_secret: str = Header(alias="X-Webhook-Secret"),
):
    """Accept a raw Centuri form JSON directly (without the wrapper envelope).

    Convenience endpoint for demo — just POST the form JSON as-is.
    """
    if x_webhook_secret != DEMO_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    import json
    raw = await request.body()
    try:
        form_payload = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    translator = get_translator()
    start = time.time()
    translated_payload, translation_map = translator.translate_form_with_metadata(
        form_payload, target_lang="es", source_lang="en"
    )
    elapsed = time.time() - start

    return JSONResponse({
        "status": "translated",
        "translated_payload": translated_payload,
        "translation_map": translation_map,
        "stats": {
            "translation_time_ms": round(elapsed * 1000),
            "strings_translated": len(translation_map),
            "engine": "opus-mt-en-es",
        },
    })


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "opus-mt-en-es", "model_loaded": _translator is not None}
