"""
Centuri Form Translation Demo — Opus-MT (EN → ES)

Translates real Centuri safety forms using Helsinki-NLP/opus-mt-en-es.
Run standalone — no ProjectA3 or adoptwebui dependencies needed.

Usage:
    python demo.py                          # translate all sample forms
    python demo.py ../form_JHA1.json        # translate a specific form
    python demo.py --side-by-side           # show EN vs ES comparison
"""

import json
import sys
import time
import os

# Add demo dir to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from form_translator_nmt import FormTranslatorNMT, extract_translatable_strings


def load_form(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def print_separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_translation_map(translation_map: dict[str, dict[str, str]]) -> None:
    """Print side-by-side EN → ES comparison."""
    max_en = max((len(m["original"]) for m in translation_map.values()), default=20)
    max_en = min(max_en, 50)  # cap column width

    print(f"  {'English':<{max_en}}  →  Spanish")
    print(f"  {'─' * max_en}     {'─' * max_en}")

    for path, mapping in translation_map.items():
        en = mapping["original"]
        es = mapping["translated"]
        # Truncate long strings for display
        en_display = (en[:max_en - 3] + "...") if len(en) > max_en else en
        print(f"  {en_display:<{max_en}}  →  {es}")


def demo_single_form(form_path: str, translator: FormTranslatorNMT, side_by_side: bool = False) -> None:
    form_name = os.path.basename(form_path)
    form_payload = load_form(form_path)

    # Show extraction stats
    pairs = extract_translatable_strings(form_payload)
    unique_count = len(set(pairs.values()))
    print(f"  Form: {form_name}")
    print(f"  Translatable strings: {len(pairs)} paths, {unique_count} unique")

    # Translate with timing
    start = time.time()
    translated_payload, translation_map = translator.translate_form_with_metadata(
        form_payload, target_lang="es", source_lang="en"
    )
    elapsed = time.time() - start

    print(f"  Translation time: {elapsed:.2f}s")
    print(f"  Strings translated: {len(translation_map)}")

    if side_by_side:
        print()
        print_translation_map(translation_map)

    # Save translated output
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_name = form_name.replace(".json", "_translated_es.json")
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(translated_payload, f, indent=2, ensure_ascii=False)
    print(f"  Output saved: {out_name}")

    return elapsed, len(translation_map)


def main():
    args = sys.argv[1:]
    side_by_side = "--side-by-side" in args or "-s" in args
    form_paths = [a for a in args if not a.startswith("-")]

    # Default: use all sample forms from parent directory
    if not form_paths:
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = ["form_JHA1.json", "from_JHA2.json", "form_BORE1.json"]
        form_paths = [os.path.join(parent, f) for f in candidates if os.path.exists(os.path.join(parent, f))]

    if not form_paths:
        print("No form files found. Pass form JSON path as argument.")
        sys.exit(1)

    print_separator("Centuri Form Translation Demo — Opus-MT (EN → ES)")
    print("  Model: Helsinki-NLP/opus-mt-en-es")
    print("  Engine: MarianMT (purpose-built neural machine translation)")
    print(f"  Forms to translate: {len(form_paths)}")

    # Initialize translator (triggers model download on first run)
    print("\n  Loading Opus-MT model...")
    load_start = time.time()
    translator = FormTranslatorNMT()
    # Warm up the model with a dummy call
    translator.translate_form({"sections": [{"label": "test"}]})
    load_time = time.time() - load_start
    print(f"  Model ready ({load_time:.1f}s)\n")

    total_strings = 0
    total_time = 0.0

    for form_path in form_paths:
        print_separator(os.path.basename(form_path))
        elapsed, count = demo_single_form(form_path, translator, side_by_side)
        total_strings += count
        total_time += elapsed

    # Summary
    print_separator("Summary")
    print(f"  Forms translated: {len(form_paths)}")
    print(f"  Total strings: {total_strings}")
    print(f"  Total translation time: {total_time:.2f}s")
    if total_strings > 0:
        print(f"  Avg per string: {total_time / total_strings * 1000:.1f}ms")
    print(f"\n  Engine: Opus-MT (Helsinki-NLP) — dedicated neural machine translation\n")


if __name__ == "__main__":
    main()
