# src/nithinlang/dicts/__init__.py
"""
Language dictionary package for NithinLang.
Provides loaders for every supported language JSON dictionary.
"""

from __future__ import annotations

import json
import os
from typing import Dict

_DICT_DIR = os.path.dirname(os.path.abspath(__file__))

_CACHE: Dict[str, Dict[str, str]] = {}


def load_language(language: str) -> Dict[str, str]:
    """
    Load and cache a language keyword dictionary.

    Args:
        language: Language name (e.g., "telugu", "hindi", "english").

    Returns:
        A dict mapping native-language keywords → NithinLang internal keywords.

    Raises:
        FileNotFoundError: If no dictionary exists for the requested language.
        json.JSONDecodeError: If the dictionary file is malformed.
    """
    lang_key = language.lower().strip()

    if lang_key in _CACHE:
        return _CACHE[lang_key]

    # English is identity — no translation needed
    if lang_key == "english":
        _CACHE[lang_key] = {}
        return {}

    dict_path = os.path.join(_DICT_DIR, f"{lang_key}.json")
    if not os.path.isfile(dict_path):
        raise FileNotFoundError(
            f"No language dictionary found for '{language}'. "
            f"Expected file: {dict_path}"
        )

    with open(dict_path, "r", encoding="utf-8") as fh:
        data: Dict[str, str] = json.load(fh)

    _CACHE[lang_key] = data
    return data


def list_languages() -> list[str]:
    """Return all available language names (from JSON files + 'english')."""
    langs = ["english"]
    for fname in os.listdir(_DICT_DIR):
        if fname.endswith(".json"):
            langs.append(fname[:-5])
    return sorted(set(langs))


__all__ = ["load_language", "list_languages"]