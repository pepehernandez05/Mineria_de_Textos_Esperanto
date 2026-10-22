"""Tokenizador: divide un texto en tokens.

El tokenizador recorre el texto con **una única expresión regular** formada
por varias alternativas. En cada posición gana la primera alternativa que
encaja, así que el orden expresa la prioridad:

1. URL (``https://…``, ``www.…``)
2. Correo electrónico
3. Abreviatura conocida (``ekz.``, ``k.t.p.``, ``d-ro``) → no corta la frase
4. Ordinal con cifras (``20-a``, ``1-an``)
5. Número (``1887``, ``3,14``, ``1.000.000``, ``50%``)
6. Palabra, con elisión opcional (``l'``, ``dom'``)
7. Puntos suspensivos (``...``)
8. Cualquier otro carácter que no sea espacio → puntuación/símbolo

Los espacios no generan tokens; se recuperan con ``Token.whitespace_``.

Las palabras con guion (``ĉi-jare``) se separan en ``ĉi`` ``-`` ``jare``, como
hace spaCy con los infijos.
"""

from __future__ import annotations

import re
from typing import Iterator, List, Optional, Tuple

from vortilo import lexicon
from vortilo.normalize import normalize_text
from vortilo.tokens import ABBR, EMAIL, NUM, ORDINAL, PUNCT, SYM, URL, WORD, Doc

__all__ = ["Tokenizer", "sentence_tokenizer", "word_tokenizer"]

#: Letra: cualquier carácter alfabético Unicode (incluye ĉ ĝ ĥ ĵ ŝ ŭ).
_LETTER = r"[^\W\d_]"

#: Signos que se consideran puntuación (el resto de no-alfanuméricos son símbolos).
_PUNCT_CHARS = set(".,;:!?¿¡\"'()[]{}-–—…/«»“”‘’")


def _build_regex() -> re.Pattern:
    abbrevs = sorted(lexicon.ABBREVIATIONS, key=len, reverse=True)
    # Las de una sola letra (n., m., j.) solo en minúscula: "M. Boulton" es una
    # inicial, no "mortis". Las demás, sin distinguir mayúsculas (Ekz., S-ro).
    long_alt = "|".join(re.escape(a) for a in abbrevs if len(a) > 2)
    short_alt = "|".join(re.escape(a) for a in abbrevs if len(a) <= 2)
    parts = [
        (URL, r"(?i:(?:https?://|www\.)[^\s<>\"]+[^\s<>\".,;:!?)\]])"),
        (EMAIL, r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
        # Abreviatura: no puede ir pegada a letras por ninguno de los lados.
        (ABBR, rf"(?<!{_LETTER})(?:(?i:{long_alt})|{short_alt})(?!{_LETTER})"),
        (ORDINAL, r"\d+-a(?:j)?(?:n)?(?!\w)"),
        (NUM, r"\d+(?:[.,]\d+)*%?"),
        # Palabra; el apóstrofo final solo si NO va seguido de otra letra.
        (WORD, rf"{_LETTER}+(?:'(?!{_LETTER}))?"),
        (PUNCT, r"\.\.\.|…"),
        ("OTHER", r"\S"),
    ]
    pattern = "|".join(f"(?P<{name}>{rx})" for name, rx in parts)
    return re.compile(pattern)


_TOKEN_RE = _build_regex()


class Tokenizer:
    """Tokenizador de Esperanto.

    Es el primer paso del pipeline: convierte un ``str`` en un :class:`Doc`.

    Args:
        x_system: Convertir el sistema X (``cx`` → ``ĉ``) antes de tokenizar.
        h_system: Convertir el sistema H (ambiguo, desactivado por defecto).

    Examples:
        >>> tok = Tokenizer()
        >>> [t.text for t in tok("Ĉu vi venos, ekz. morgaŭ?")]
        ['Ĉu', 'vi', 'venos', ',', 'ekz.', 'morgaŭ', '?']
        >>> [t.text for t in tok("Cxu l' urbo?")]
        ['Ĉu', "l'", 'urbo', '?']
    """

    def __init__(self, x_system: bool = True, h_system: bool = False):
        self.x_system = x_system
        self.h_system = h_system

    def spans(self, text: str) -> Iterator[Tuple[int, str, str]]:
        """Genera ``(posición, texto, tipo)`` para cada token de ``text``.

        ``text`` debe estar ya normalizado.
        """
        for match in _TOKEN_RE.finditer(text):
            kind = match.lastgroup or "OTHER"
            value = match.group()
            if kind == "OTHER":
                kind = PUNCT if value in _PUNCT_CHARS else SYM
            yield match.start(), value, kind

    def __call__(self, text: str) -> Doc:
        """Tokeniza ``text`` y devuelve un :class:`Doc` sin anotar."""
        text = normalize_text(text, x_system=self.x_system, h_system=self.h_system)
        tokens = [(idx, value, kind, value.lower()) for idx, value, kind in self.spans(text)]
        return Doc(text, tokens)


def word_tokenizer(text: str) -> List[str]:
    """Atajo: lista con el texto de cada token.

    Examples:
        >>> word_tokenizer("La 20-a jarcento.")
        ['La', '20-a', 'jarcento', '.']
    """
    return [t.text for t in Tokenizer()(text)]


# --------------------------------------------------------------------------- #
# Segmentación en frases
# --------------------------------------------------------------------------- #

#: Signos que cierran frase.
SENTENCE_END = frozenset({".", "!", "?", "...", "…"})

#: Signos que pueden ir justo detrás del final de frase y pertenecen a ella.
_CLOSERS = frozenset({'"', "'", ")", "]", "»", "”"})


def mark_sentence_starts(doc: Doc) -> Doc:
    """Marca ``Token.is_sent_start`` en un documento ya tokenizado.

    Una frase termina en ``. ! ? …`` (más comillas o paréntesis de cierre) y
    la siguiente empieza **solo** si el siguiente token empieza por mayúscula,
    por cifra o es un signo de apertura. Así ``ekz. la hundo`` o ``3.14`` no
    cortan la frase (las abreviaturas y números ya son un único token).

    Tampoco cortan las iniciales de nombres (``L. L. Zamenhof``): una sola
    letra mayúscula pegada al punto. Los saltos de línea dobles (párrafos)
    siempre cortan.
    """
    tokens = list(doc)
    if not tokens:
        return doc
    for tok in tokens:
        tok.is_sent_start = False
    tokens[0].is_sent_start = True

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Párrafo nuevo: el espacio previo contiene una línea en blanco.
        if i > 0 and "\n\n" in tokens[i - 1].whitespace_:
            tok.is_sent_start = True
        if tok.text in SENTENCE_END:
            # Inicial de un nombre: "L. L. Zamenhof" no son tres frases.
            if tok.text == "." and i > 0:
                prev = tokens[i - 1]
                if len(prev.text) == 1 and prev.text.isupper() and prev.whitespace_ == "":
                    i += 1
                    continue
            j = i + 1
            while j < len(tokens) and tokens[j].text in _CLOSERS:
                j += 1
            if j < len(tokens):
                nxt = tokens[j]
                first = nxt.text[0]
                if first.isupper() or first.isdigit() or nxt.text in {"(", '"', "«", "¿", "¡", "-"}:
                    nxt.is_sent_start = True
            i = j
            continue
        i += 1
    return doc


def sentence_tokenizer(text: str) -> List[str]:
    """Atajo: divide un texto en frases (lista de ``str``).

    Examples:
        >>> sentence_tokenizer("Mi lernas, ekz. Esperanton. Ĝi estas facila! Ĉu vere?")
        ['Mi lernas, ekz. Esperanton.', 'Ĝi estas facila!', 'Ĉu vere?']
    """
    doc = mark_sentence_starts(Tokenizer()(text))
    return [s.text for s in doc.sents]
