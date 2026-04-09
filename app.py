"""
Centuri Form Translation — Streamlit Demo

Interactive demo: paste or load a Centuri form JSON, toggle EN → ES,
see the translated output side-by-side with translation map and metrics.

Run:
    cd centuri_translation_demo
    streamlit run app.py
"""

import json
import os
import sys
import time

import streamlit as st

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from form_translator_nmt import (
    FormTranslatorNMT,
    extract_translatable_strings,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Centuri Form Translation",
    page_icon="🔄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sample forms — paths relative to project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_FORMS = {
    "JHA1 — Job Hazard Analysis": os.path.join(PROJECT_ROOT, "form_JHA1.json"),
    "JHA2 — Job Hazard Analysis (v2)": os.path.join(PROJECT_ROOT, "from_JHA2.json"),
    "BORE1 — Bore Report": os.path.join(PROJECT_ROOT, "form_BORE1.json"),
}


def load_sample(name: str) -> str:
    path = SAMPLE_FORMS.get(name)
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


# ---------------------------------------------------------------------------
# Model loading (cached so it only happens once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_translator() -> FormTranslatorNMT:
    translator = FormTranslatorNMT()
    # Warm up to ensure model weights are loaded
    translator.translate_form({"sections": [{"label": "warmup"}]})
    return translator


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Centuri Form Translation")
st.caption("Adopt AI — EN ↔ ES Neural Machine Translation (Opus-MT)")

# Sidebar
with st.sidebar:
    st.header("Configuration")

    source_lang = "en"
    target_lang = "es"

    st.divider()

    st.subheader("Load sample form")
    sample_choice = st.selectbox(
        "Select a Centuri form",
        ["— Paste your own —"] + list(SAMPLE_FORMS.keys()),
    )

    st.divider()

    st.subheader("About")
    st.markdown(
        """
        **Engine:** Helsinki-NLP/opus-mt-en-es
        **Type:** MarianMT (dedicated NMT)
        **Pipeline:** Extract → Deduplicate → Translate → Reconstruct
        """
    )

# ---------------------------------------------------------------------------
# Main area — two columns
# ---------------------------------------------------------------------------

# Initialize session state
if "translated_json" not in st.session_state:
    st.session_state.translated_json = None
if "translation_map" not in st.session_state:
    st.session_state.translation_map = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "_last_sample" not in st.session_state:
    st.session_state._last_sample = None

# Update text area when sample selection changes
if sample_choice != st.session_state._last_sample:
    st.session_state._last_sample = sample_choice
    if sample_choice != "— Paste your own —":
        st.session_state.json_input = load_sample(sample_choice)
    else:
        st.session_state.json_input = ""
    # Reset translation when input changes
    st.session_state.translated_json = None
    st.session_state.translation_map = None
    st.session_state.metrics = None

col_input, col_output = st.columns(2)

with col_input:
    st.subheader(f"Source ({source_lang.upper()})")
    input_json = st.text_area(
        "Paste Centuri form JSON",
        height=500,
        key="json_input",
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Translate button
# ---------------------------------------------------------------------------
translate_col, stats_col = st.columns([1, 3])

with translate_col:
    translate_clicked = st.button(
        f"Translate  {source_lang.upper()} → {target_lang.upper()}",
        type="primary",
        use_container_width=True,
    )

if translate_clicked and input_json.strip():
    # Validate JSON
    try:
        payload = json.loads(input_json)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    # Extract stats before translation
    pairs = extract_translatable_strings(payload)
    unique_count = len(set(pairs.values()))

    # Load model + translate
    with st.spinner("Loading translation model..."):
        translator = load_translator()

    with st.spinner(f"Translating {len(pairs)} strings..."):
        start = time.time()
        translated_payload, translation_map = translator.translate_form_with_metadata(
            payload, target_lang=target_lang, source_lang=source_lang
        )
        elapsed = time.time() - start

    # Store results
    st.session_state.translated_json = json.dumps(translated_payload, indent=2, ensure_ascii=False)
    st.session_state.translation_map = translation_map
    st.session_state.metrics = {
        "total_paths": len(pairs),
        "unique_strings": unique_count,
        "translation_time_s": round(elapsed, 2),
        "avg_per_string_ms": round(elapsed / max(len(pairs), 1) * 1000, 1),
    }

# Show metrics
if st.session_state.metrics:
    m = st.session_state.metrics
    with stats_col:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Strings", m["total_paths"])
        mc2.metric("Unique", m["unique_strings"])
        mc3.metric("Time", f"{m['translation_time_s']}s")
        mc4.metric("Per string", f"{m['avg_per_string_ms']}ms")

# Show translated output
with col_output:
    st.subheader(f"Translated ({target_lang.upper()})")
    if st.session_state.translated_json:
        st.text_area(
            "Translated JSON",
            value=st.session_state.translated_json,
            height=500,
            label_visibility="collapsed",
        )
    else:
        st.text_area(
            "Translated JSON",
            value="← Paste form JSON and click Translate",
            height=500,
            label_visibility="collapsed",
            disabled=True,
        )

# ---------------------------------------------------------------------------
# Translation map — expandable detail
# ---------------------------------------------------------------------------
if st.session_state.translation_map:
    st.divider()
    st.subheader("Translation Map")

    tmap = st.session_state.translation_map

    # Build table data
    table_data = []
    for path, mapping in tmap.items():
        original = mapping["original"]
        translated = mapping["translated"]
        changed = original != translated
        table_data.append({
            "Path": path,
            "Original": original,
            "Translated": translated,
            "Changed": "Yes" if changed else "—",
        })

    st.dataframe(
        table_data,
        use_container_width=True,
        height=400,
        column_config={
            "Path": st.column_config.TextColumn("JSON Path", width="medium"),
            "Original": st.column_config.TextColumn("Original", width="medium"),
            "Translated": st.column_config.TextColumn("Translated", width="medium"),
            "Changed": st.column_config.TextColumn("Changed", width="small"),
        },
    )

    # Download buttons
    dl_col1, dl_col2, _ = st.columns([1, 1, 4])
    with dl_col1:
        st.download_button(
            "Download translated JSON",
            data=st.session_state.translated_json,
            file_name="form_translated.json",
            mime="application/json",
        )
    with dl_col2:
        map_json = json.dumps(tmap, indent=2, ensure_ascii=False)
        st.download_button(
            "Download translation map",
            data=map_json,
            file_name="translation_map.json",
            mime="application/json",
        )
