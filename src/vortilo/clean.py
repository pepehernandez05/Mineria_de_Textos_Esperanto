"""Limpieza de texto crudo (web y Wikipedia) y división en párrafos.

Se usa en dos sitios:

* El *script* del dataset, para cumplir el contrato de datos: párrafos sin
  HTML, enlaces, referencias ni el carácter ``;``.
* El cuaderno, para limpiar el texto de las páginas web antes de clasificarlo
  **igual** que se limpió el de entrenamiento (si no, el modelo vería datos con
  otra distribución).
"""

from __future__ import annotations

import html
import re
from typing import List

__all__ = [
    "strip_html",
    "remove_urls",
    "remove_references",
    "normalize_whitespace",
    "clean_text",
    "split_paragraphs",
]

_TAG = re.compile(r"<[^>]+>")
_BLOCK_TAG = re.compile(
    r"</?(?:p|div|br|li|ul|ol|h[1-6]|tr|td|th|table|section|article|blockquote|header|footer)\b[^>]*>",
    re.IGNORECASE,
)
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\w+")
#: [1], [12], [a], [noto 3], [bezonata citaĵo], [citation needed]…
_BRACKET_REF = re.compile(r"\[(?:\d+|[a-z]|noto\s*\d+|[^\]]{0,30}(?:citaĵo|fonto|citation)[^\]]{0,10})\]", re.IGNORECASE)
#: Plantillas de wikitexto que a veces se cuelan: {{...}}
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
#: Paréntesis vacíos que quedan al borrar cosas: "( )" o "(, )".
_EMPTY_PARENS = re.compile(r"\(\s*[,;:]?\s*\)")
_SPACES = re.compile(r"[ \t\r\f\v]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)])")


def strip_html(text: str) -> str:
    """Quita etiquetas HTML (y el contenido de ``<script>``/``<style>``) y decodifica entidades.

    Examples:
        >>> strip_html("<p>La <b>hundo</b> &amp; la kato</p>")
        'La hundo & la kato'
    """
    text = _SCRIPT_STYLE.sub(" ", text)
    # Las etiquetas de bloque separan párrafos; las de línea (<b>, <a>) no.
    text = _BLOCK_TAG.sub("\n", text)
    text = _TAG.sub("", text)
    return html.unescape(text).strip()


def remove_urls(text: str) -> str:
    """Quita URLs y correos.

    Examples:
        >>> remove_urls("Vidu https://eo.wikipedia.org/wiki/Hundo por pli.")
        'Vidu  por pli.'
    """
    return _EMAIL.sub("", _URL.sub("", text))


def remove_references(text: str) -> str:
    """Quita referencias tipo Wikipedia (``[1]``, ``[noto 2]``) y plantillas ``{{…}}``.

    Examples:
        >>> remove_references("Ĝi estis fondita en 1950[1][2].")
        'Ĝi estis fondita en 1950.'
    """
    text = _TEMPLATE.sub("", text)
    return _BRACKET_REF.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Colapsa espacios dentro de cada línea (respeta los saltos de línea)."""
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(lines)


def clean_text(text: str, semicolon: str = ",") -> str:
    """Limpieza completa: HTML, URLs, referencias, ``;`` y espacios.

    Args:
        text: Texto crudo.
        semicolon: Por qué se sustituye ``;`` (el contrato de datos lo
            prohíbe porque es el separador del CSV).

    Returns:
        Texto limpio. Conserva los saltos de línea para poder dividir en párrafos.

    Examples:
        >>> clean_text("<p>Unu; du [3] https://x.org</p>")
        'Unu, du'
    """
    text = strip_html(text)
    text = remove_urls(text)
    text = remove_references(text)
    text = text.replace(";", semicolon)
    text = _EMPTY_PARENS.sub("", text)
    text = normalize_whitespace(text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()


def split_paragraphs(text: str, min_words: int = 1) -> List[str]:
    """Divide un texto en párrafos (uno por línea no vacía).

    Args:
        text: Texto (idealmente ya pasado por :func:`clean_text`).
        min_words: Descarta párrafos con menos palabras (títulos, pies de foto…).

    Examples:
        >>> split_paragraphs("Titolo\\n\\nUnua alineo kun kvin vortoj.\\nDua.", min_words=3)
        ['Unua alineo kun kvin vortoj.']
    """
    paragraphs = []
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line.split()) >= min_words:
            paragraphs.append(line)
    return paragraphs
