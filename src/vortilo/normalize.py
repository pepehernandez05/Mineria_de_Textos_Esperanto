"""Normalización de texto en Esperanto.

Muchos textos en Esperanto (sobre todo en internet) no usan las letras con
diacrítico ``ĉ ĝ ĥ ĵ ŝ ŭ``, sino uno de estos dos sistemas sustitutos:

* **Sistema X** (``x-sistemo``): ``cx gx hx jx sx ux`` → ``ĉ ĝ ĥ ĵ ŝ ŭ``.
  Es inequívoco porque la letra ``x`` no existe en Esperanto.
* **Sistema H** (``h-sistemo``) de Zamenhof: ``ch gh hh jh sh u`` →
  ``ĉ ĝ ĥ ĵ ŝ ŭ``. Es **ambiguo** (``flughaveno`` = *flug-haveno* contiene ``gh``
  sin ser ``ĝ``), así que no se aplica por defecto.

Normalizar evita que ``ĉambro`` y ``cxambro`` cuenten como dos palabras
distintas en el modelo de clasificación.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["x_to_unicode", "h_to_unicode", "normalize_text", "normalize_word"]

_X_MAP = {
    "cx": "ĉ", "gx": "ĝ", "hx": "ĥ", "jx": "ĵ", "sx": "ŝ", "ux": "ŭ",
    "Cx": "Ĉ", "Gx": "Ĝ", "Hx": "Ĥ", "Jx": "Ĵ", "Sx": "Ŝ", "Ux": "Ŭ",
    "CX": "Ĉ", "GX": "Ĝ", "HX": "Ĥ", "JX": "Ĵ", "SX": "Ŝ", "UX": "Ŭ",
}
_X_RE = re.compile(r"([cghjsuCGHJSU])([xX])")

_H_MAP = {
    "ch": "ĉ", "gh": "ĝ", "hh": "ĥ", "jh": "ĵ", "sh": "ŝ",
    "Ch": "Ĉ", "Gh": "Ĝ", "Hh": "Ĥ", "Jh": "Ĵ", "Sh": "Ŝ",
    "CH": "Ĉ", "GH": "Ĝ", "HH": "Ĥ", "JH": "Ĵ", "SH": "Ŝ",
}
_H_RE = re.compile(r"([cghjsCGHJS])([hH])")

#: Signos tipográficos que se unifican con su versión ASCII.
_TYPOGRAPHIC = str.maketrans(
    {
        "’": "'",  # ’
        "‘": "'",  # ‘
        "ʼ": "'",  # ʼ
        "“": '"',  # “
        "”": '"',  # ”
        "„": '"',  # „
        "«": '"',  # «
        "»": '"',  # »
        "–": "-",  # –
        "—": "-",  # —
        " ": " ",  # espacio duro
    }
)


def x_to_unicode(text: str) -> str:
    """Convierte el sistema X a letras con diacrítico.

    Examples:
        >>> x_to_unicode("Cxu vi parolas Esperanton? Jes, cxiutage!")
        'Ĉu vi parolas Esperanton? Jes, ĉiutage!'
    """
    return _X_RE.sub(lambda m: _X_MAP[m.group(0)], text)


def h_to_unicode(text: str) -> str:
    """Convierte el sistema H a letras con diacrítico.

    Advertencia:
        Es ambiguo (``flughaveno``) y no convierte ``au`` → ``aŭ``.
        Úsalo solo si sabes que el texto está escrito en sistema H.

    Examples:
        >>> h_to_unicode("chambro")
        'ĉambro'
    """
    return _H_RE.sub(lambda m: _H_MAP[m.group(0)], text)


def normalize_text(text: str, x_system: bool = True, h_system: bool = False) -> str:
    """Normaliza un texto completo.

    Ojo: el sistema X acorta el texto (``cx`` → ``ĉ``), así que las posiciones
    de caracteres (``Token.idx``) se refieren siempre al texto **normalizado**,
    que es el que guarda ``Doc.text``.

    Pasos:

    1. Unicode NFC (une ``c`` + ``̂`` en un solo carácter ``ĉ``).
    2. Comillas, apóstrofos, guiones y espacios tipográficos → ASCII.
    3. Sistema X (y opcionalmente H) → diacríticos.

    No pasa a minúsculas: las mayúsculas sirven para detectar nombres propios
    y comienzos de frase.

    Args:
        text: Texto de entrada.
        x_system: Convertir el sistema X. Por defecto ``True``.
        h_system: Convertir el sistema H. Por defecto ``False``.

    Returns:
        El texto normalizado.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TYPOGRAPHIC)
    if x_system:
        text = x_to_unicode(text)
    if h_system:
        text = h_to_unicode(text)
    return text


def normalize_word(word: str) -> str:
    """Forma normalizada de una palabra: :func:`normalize_text` + minúsculas.

    Es el valor de ``Token.norm_``.

    Examples:
        >>> normalize_word("CXAMBRO")
        'ĉambro'
    """
    return normalize_text(word).lower()
