# src/nithinlang/parser.py
"""
NithinLang Multi-Lingual AST Parser
====================================
Responsible for:
  1. Validating the mandatory file envelope  (nithin ... end nithin)
  2. Extracting the language declaration     (lang+ <language>)
  3. Loading the appropriate keyword dictionary
  4. Performing token-level keyword substitution
  5. Producing a clean Python-compatible source string for the compiler

Architecture notes:
  - All keyword translation happens at the TOKEN level (not line-by-line
    substring replacement) to avoid partial matches inside string literals
    or identifiers.
  - String literals (single/double/triple-quoted) are preserved verbatim.
  - Comments (# …) are preserved verbatim.
  - The parser outputs a ParseResult dataclass consumed by NithinCompiler.
"""

from __future__ import annotations

import re
import tokenize
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nithinlang.dicts import load_language


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParseError:
    """Represents a single parse error with location information."""
    line_no  : int
    col_no   : int
    message  : str
    source   : str = ""

    def __str__(self) -> str:
        return (
            f"  ParseError at line {self.line_no}, col {self.col_no}: "
            f"{self.message}\n"
            f"  → {self.source}"
        )


@dataclass
class ParseResult:
    """Holds the complete output of a successful parse operation."""
    language       : str
    raw_source     : str
    translated_src : str
    keyword_map    : Dict[str, str]
    errors         : List[ParseError] = field(default_factory=list)
    warnings       : List[str]        = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Regex that matches the mandatory opening line (allows leading/trailing spaces)
_RE_NITHIN_START = re.compile(r"^\s*nithin\s*$", re.IGNORECASE)
# Regex for the mandatory closing line
_RE_NITHIN_END   = re.compile(r"^\s*end\s+nithin\s*$", re.IGNORECASE)
# Regex for the lang+ declaration: lang+ <word>
_RE_LANG_DECL    = re.compile(r"^\s*lang\+\s+(\w+)\s*$", re.IGNORECASE)

# Python tokenize types we want to translate
_TRANSLATE_TYPES = {tokenize.NAME, tokenize.OP}


def _extract_envelope(raw: str) -> Tuple[Optional[str], List[ParseError]]:
    """
    Validate the nithin / end nithin envelope and extract the inner body.

    Returns:
        (body_text, errors)  — body_text is None on fatal error.
    """
    errors: List[ParseError] = []
    lines  = raw.splitlines()

    if not lines:
        errors.append(ParseError(1, 0, "File is empty.", ""))
        return None, errors

    if not _RE_NITHIN_START.match(lines[0]):
        errors.append(ParseError(
            1, 0,
            "File must start with 'nithin' on the first line.",
            lines[0]
        ))
        return None, errors

    if not _RE_NITHIN_END.match(lines[-1]):
        errors.append(ParseError(
            len(lines), 0,
            "File must end with 'end nithin' on the last line.",
            lines[-1]
        ))
        return None, errors

    body = "\n".join(lines[1:-1])
    return body, errors


def _extract_language(body: str) -> Tuple[Optional[str], str, List[ParseError]]:
    """
    Extract the language declaration from the first non-empty body line.

    Returns:
        (language, remaining_body, errors)
    """
    errors: List[ParseError] = []
    lines  = body.splitlines()

    # Find first non-empty line
    first_content_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_content_idx = idx
            break

    if first_content_idx is None:
        errors.append(ParseError(
            2, 0,
            "Body is empty. Expected 'lang+ <language>' as the first statement.",
            ""
        ))
        return None, body, errors

    first_line = lines[first_content_idx]
    m = _RE_LANG_DECL.match(first_line)
    if not m:
        errors.append(ParseError(
            first_content_idx + 2,   # +2 because line 1 was 'nithin'
            0,
            (
                f"Expected 'lang+ <language>' but got: '{first_line.strip()}'. "
                "Supported languages: telugu, hindi, english"
            ),
            first_line,
        ))
        return None, body, errors

    language      = m.group(1).lower()
    remaining     = "\n".join(
        lines[:first_content_idx] + lines[first_content_idx + 1:]
    )
    return language, remaining, errors


def _tokenize_and_translate(
    source        : str,
    keyword_map   : Dict[str, str],
    language      : str,
    line_offset   : int = 2,
) -> Tuple[str, List[ParseError]]:
    """
    Walk through the Python tokenizer output for `source` and replace
    every NAME token that appears as a key in `keyword_map` with its value.

    String literals, comments, numbers, and operators pass through unchanged
    (unless they happen to be in keyword_map — operators can be remapped too).

    Args:
        source      : Source code after envelope/lang stripping.
        keyword_map : {native_keyword: python_keyword} mapping.
        language    : Language name (for error messages).
        line_offset : How many lines were stripped before this source
                      (used for accurate error reporting).

    Returns:
        (translated_source, errors)
    """
    errors      : List[ParseError] = []
    result_parts: List[str]        = []

    if not source.strip():
        return source, errors

    # We need to handle cases where the source might contain syntax that
    # Python's tokenizer doesn't understand (pre-translation keywords).
    # Strategy: perform a simple regex-based word-boundary substitution
    # for NAME tokens, which is safe because:
    #   a) we only replace whole words (\\b boundaries)
    #   b) we skip content inside string literals
    #   c) we skip content inside comments

    if not keyword_map:
        # English or empty map — no translation needed
        return source, errors

    # Build an alternation regex of all keywords, longest first (greedy)
    # so "not in" is matched before "not".
    sorted_keys = sorted(keyword_map.keys(), key=len, reverse=True)

    # We'll use a stateful scanner that's aware of string/comment contexts.
    translated = _safe_keyword_replace(source, keyword_map, sorted_keys, errors, line_offset)
    return translated, errors


def _safe_keyword_replace(
    source      : str,
    keyword_map : Dict[str, str],
    sorted_keys : List[str],
    errors      : List[ParseError],
    line_offset : int,
) -> str:
    """
    Replace keywords safely, skipping string literals and comments.

    Uses a single-pass regex that matches:
      - Triple-quoted strings (preserves verbatim)
      - Double-quoted strings (preserves verbatim)
      - Single-quoted strings (preserves verbatim)
      - Comments              (preserves verbatim)
      - Whole-word keyword    (replaces with mapped value)
      - Any other character   (preserves verbatim)
    """

    # Build the pattern for keywords (multi-word first, then single-word)
    kw_pattern_parts = []
    for kw in sorted_keys:
        # Escape special regex chars in the keyword, then add word boundaries
        escaped = re.escape(kw)
        # For multi-word keywords like "not in", allow flexible whitespace
        escaped_flexible = re.sub(r"\\ ", r"\\s+", escaped)
        kw_pattern_parts.append(r"\b" + escaped_flexible + r"\b")

    kw_pattern = "|".join(kw_pattern_parts)

    # Master pattern: literals/comments take priority over keywords
    master_pattern = re.compile(
        r'("""[\s\S]*?"""'           # triple double-quoted string
        r"|'''[\s\S]*?'''"           # triple single-quoted string
        r'|"(?:[^"\\]|\\.)*"'        # double-quoted string
        r"|'(?:[^'\\]|\\.)*'"        # single-quoted string
        r"|#[^\n]*"                  # comment
        r"|" + kw_pattern + ")"     # keyword to replace
        ,
        flags=re.UNICODE,
    )

    def replacer(m: re.Match) -> str:
        token = m.group(0)
        # If it's a string literal or comment, return unchanged
        if (
            token.startswith('"""')
            or token.startswith("'''")
            or token.startswith('"')
            or token.startswith("'")
            or token.startswith("#")
        ):
            return token

        # Normalise internal whitespace for lookup
        normalised = re.sub(r"\s+", " ", token).strip()
        replacement = keyword_map.get(normalised)
        if replacement is not None:
            return replacement
        return token

    return master_pattern.sub(replacer, source)


def _translate_nithin_builtins(source: str) -> str:
    """
    Map NithinLang built-in function names that are not Python keywords
    but ARE NithinLang standard names, so they survive the translation
    step unchanged (they are handled by the runtime environment).

    This is effectively a no-op translation — the names pass through
    to the compiler which injects them into the execution namespace.
    """
    # Built-ins that need NO translation — they are already valid Python
    # identifiers and will be injected into __builtins__ by the compiler.
    NITHIN_BUILTINS = {
        "data_chudu", "model_train", "model_test", "model_predict",
        "game_start",  "game_draw",  "game_loop",  "game_stop",
        "ai_adugu",    "ai_chudu",
        "f_open",      "f_read",     "f_write",    "f_close",
        "f_append",    "f_exists",   "f_delete",
        "vec_add",     "vec_dot",    "mat_mul",
        "nl_print",    "nl_input",
    }
    # These are already in the source verbatim — nothing to do.
    return source


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NithinParser:
    """
    Top-level parser for NithinLang source files.

    Usage:
        parser = NithinParser()
        result = parser.parse(source_code)
        if result.success:
            compiler.compile(result)
    """

    def parse(self, raw_source: str, filename: str = "<unknown>") -> ParseResult:
        """
        Parse a complete NithinLang source string.

        Args:
            raw_source : Full text of the .nl file.
            filename   : Used only for error messages.

        Returns:
            ParseResult — check .success and .errors.
        """
        all_errors: List[ParseError] = []

        # ── Step 1: Validate envelope ──────────────────────────────────────
        body, env_errors = _extract_envelope(raw_source)
        all_errors.extend(env_errors)
        if body is None:
            return ParseResult(
                language="unknown",
                raw_source=raw_source,
                translated_src="",
                keyword_map={},
                errors=all_errors,
            )

        # ── Step 2: Extract language declaration ───────────────────────────
        language, code_body, lang_errors = _extract_language(body)
        all_errors.extend(lang_errors)
        if language is None:
            return ParseResult(
                language="unknown",
                raw_source=raw_source,
                translated_src="",
                keyword_map={},
                errors=all_errors,
            )

        # ── Step 3: Load keyword dictionary ───────────────────────────────
        try:
            keyword_map = load_language(language)
        except FileNotFoundError as exc:
            all_errors.append(ParseError(2, 0, str(exc), ""))
            return ParseResult(
                language=language,
                raw_source=raw_source,
                translated_src="",
                keyword_map={},
                errors=all_errors,
            )

        # ── Step 4: Keyword translation ────────────────────────────────────
        translated, trans_errors = _tokenize_and_translate(
            source      = code_body,
            keyword_map = keyword_map,
            language    = language,
            line_offset = 3,   # line 1 = nithin, line 2 = lang+, line 3+ = code
        )
        all_errors.extend(trans_errors)

        # ── Step 5: Post-process NithinLang built-in names ─────────────────
        translated = _translate_nithin_builtins(translated)

        return ParseResult(
            language       = language,
            raw_source     = raw_source,
            translated_src = translated,
            keyword_map    = keyword_map,
            errors         = all_errors,
        )

    def parse_file(self, filepath: str) -> ParseResult:
        """Convenience method: read a .nl file and parse it."""
        with open(filepath, "r", encoding="utf-8") as fh:
            raw = fh.read()
        return self.parse(raw, filename=filepath)