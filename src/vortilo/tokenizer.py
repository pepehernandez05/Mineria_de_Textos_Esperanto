"""Tokenizador: divide un texto en tokens.


El tokenizador recorre el texto de izquierda a derecha. En cada posición prueba
estos tipos de token, en orden, y se queda con el primero que encaja:

1. URL (``https://…``, ``www.…``)
2. Correo electrónico
3. Abreviatura conocida (``ekz.``, ``k.t.p.``, ``d-ro``)
4. Ordinal con cifras (``20-a``, ``1-an``)
5. Número (``1887``, ``3,14``, ``1.000.000``, ``50%``)
6. Palabra (``hundo``, ``H2O``), con elisión opcional (``l'``, ``dom'``)
7. Puntos suspensivos (``...``)
8. Cualquier otro carácter que no sea espacio → puntuación/símbolo

Los espacios no generan tokens; se recuperan con ``Token.whitespace_``.

Las palabras con guion (``ĉi-jare``) se separan en ``ĉi`` ``-`` ``jare``, como
hace spaCy con los infijos.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

from vortilo import lexicon
from vortilo.normalize import normalize_text
from vortilo.tokens import ABBR, EMAIL, NUM, ORDINAL, PUNCT, SYM, URL, WORD, Doc

__all__ = ["Tokenizer", "sentence_tokenizer", "word_tokenizer"]

#: Signos que se consideran puntuación (el resto de no-alfanuméricos son símbolos).
_PUNCT_CHARS = set(".,;:!?¿¡\"'()[]{}-–—…/«»“”‘’")

_URL_PREFIXES = ("https://", "http://", "www.")

#: De más larga a más corta, para que ``ktp.`` gane a ``ktp``.
_ABBREVIATIONS = sorted(lexicon.ABBREVIATIONS, key=len, reverse=True)

_ORDINAL_ENDINGS = ("-ajn", "-aj", "-an", "-a")


def _url_end(text: str, i: int) -> Optional[int]:
    """Fin de la URL que empieza en ``i``, o ``None`` si no hay ninguna."""
    start = text[i : i + 8].lower()
    prefix = next((p for p in _URL_PREFIXES if start.startswith(p)), None)
    if prefix is None:
        return None
    end = i
    while end < len(text) and not text[end].isspace() and text[end] not in '<>"':
        end += 1
    # La puntuación final pertenece a la frase, no a la URL.
    end = i + len(text[i:end].rstrip(".,;:!?)]"))
    return end if end > i + len(prefix) else None


def _email_end(text: str, i: int) -> Optional[int]:
    """Fin del correo que empieza en ``i``, o ``None`` si no hay ninguno."""
    at = i
    while at < len(text) and (text[at].isalnum() or text[at] in "._+-"):
        at += 1
    if at == i or at == len(text) or text[at] != "@":
        return None
    end = at + 1
    while end < len(text) and (text[end].isalnum() or text[end] in "._-"):
        end += 1
    domain = text[at + 1 : end].rstrip(".")
    return at + 1 + len(domain) if "." in domain else None


def _abbreviation_end(text: str, i: int) -> Optional[int]:
    """Fin de la abreviatura que empieza en ``i``, o ``None`` si no hay ninguna.

    Las de una sola letra (``n.``, ``m.``, ``j.``) solo valen en minúscula:
    ``M. Boulton`` es una inicial, no "mortis". Las demás no distinguen
    mayúsculas (``Ekz.``, ``S-ro``).
    """
    for abbreviation in _ABBREVIATIONS:
        end = i + len(abbreviation)
        piece = text[i:end]
        if len(abbreviation) > 2:
            piece = piece.lower()
        if piece == abbreviation and not (end < len(text) and text[end].isalpha()):
            return end
    return None


def _digits_end(text: str, i: int) -> int:
    """Posición del primer carácter desde ``i`` que no es una cifra."""
    while i < len(text) and text[i].isdecimal():
        i += 1
    return i


def _ordinal_end(text: str, i: int) -> Optional[int]:
    """Fin del ordinal (``20-a``, ``1-ajn``) que empieza en ``i``, o ``None``."""
    digits = _digits_end(text, i)
    if digits == i:
        return None
    for ending in _ORDINAL_ENDINGS:
        end = digits + len(ending)
        if text.startswith(ending, digits) and not (end < len(text) and text[end].isalnum()):
            return end
    return None


def _number_end(text: str, i: int) -> Optional[int]:
    """Fin del número (``1887``, ``3,14``, ``50%``) que empieza en ``i``, o ``None``."""
    end = _digits_end(text, i)
    if end == i:
        return None
    # Un punto o una coma solo forman parte del número si les sigue una cifra.
    while end + 1 < len(text) and text[end] in ".," and text[end + 1].isdecimal():
        end = _digits_end(text, end + 1)
    if end < len(text) and text[end] == "%":
        end += 1
    return end


def _word_end(text: str, i: int, quote_open: bool) -> int:
    """Fin de la palabra que empieza en ``i``, incluida su posible elisión.

    Un apóstrofo final se queda en la palabra en dos casos:

    * Es ``l'``, aunque le siga otra letra (``l'urbo``).
    * No le sigue una letra y no cierra una comilla simple (``dom'``).
    """
    end = i + 1
    while end < len(text) and text[end].isalnum():
        end += 1
    if end == len(text) or text[end] != "'":
        return end
    followed_by_letter = end + 1 < len(text) and text[end + 1].isalpha()
    if followed_by_letter:
        keep_apostrophe = text[i:end].lower() == "l"
    else:
        keep_apostrophe = not quote_open
    return end + 1 if keep_apostrophe else end


def _read_token(text: str, i: int, quote_open: bool) -> Tuple[int, str]:
    """Devuelve ``(fin, tipo)`` del token que empieza en la posición ``i``."""
    readers = (
        (_url_end, URL),
        (_email_end, EMAIL),
        (_abbreviation_end, ABBR),
        (_ordinal_end, ORDINAL),
        (_number_end, NUM),
    )
    for reader, kind in readers:
        end = reader(text, i)
        if end is not None:
            return end, kind
    if text[i].isalpha():
        return _word_end(text, i, quote_open), WORD
    if text.startswith("...", i):
        return i + 3, PUNCT
    return i + 1, PUNCT if text[i] in _PUNCT_CHARS else SYM


def _opens_quote(text: str, i: int) -> bool:
    """¿El apóstrofo de la posición ``i`` abre una comilla simple (``'hundo``)?"""
    after_word = i > 0 and text[i - 1].isalnum()
    before_letter = i + 1 < len(text) and text[i + 1].isalpha()
    return before_letter and not after_word


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
        i = 0
        quote_open = False  # Para distinguir la comilla de cierre de una elisión.
        while i < len(text):
            if text[i].isspace():
                if text[i] == "\n":
                    quote_open = False
                i += 1
                continue
            end, kind = _read_token(text, i, quote_open)
            if text[i:end] == "'":
                quote_open = not quote_open and _opens_quote(text, i)
            yield i, text[i:end], kind
            i = end

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

#: Abreviaturas que suelen cerrar una enumeración y, por tanto, pueden cerrar
#: frase. Las demás (``ekz.``, ``d-ro``…) van seguidas de lo que introducen.
_CLOSING_ABBREVIATIONS = frozenset({"k.t.p.", "ktp.", "k.a.", "k.s."})

#: Signos que pueden ir justo detrás del final de frase y pertenecen a ella.
_CLOSERS = frozenset({'"', "'", ")", "]", "»", "”"})


def mark_sentence_starts(doc: Doc) -> Doc:
    """Marca ``Token.is_sent_start`` en un documento ya tokenizado.

    Una frase termina en ``. ! ? …`` (más comillas o paréntesis de cierre) y
    la siguiente empieza **solo** si el siguiente token empieza por mayúscula,
    por cifra o es un signo de apertura. Así ``ekz. la hundo`` o ``3.14`` no
    cortan la frase (las abreviaturas y números ya son un único token). Las
    abreviaturas que cierran enumeraciones (``ktp.``) siguen la misma regla.

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
        if tok.text in SENTENCE_END or tok.norm_ in _CLOSING_ABBREVIATIONS:
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
