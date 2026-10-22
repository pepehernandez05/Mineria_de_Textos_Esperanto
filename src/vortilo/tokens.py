"""Objetos ``Doc``, ``Token``, ``Span`` y ``MorphAnalysis``.

Imitan la API de spaCy para que quien la conozca pueda usar la librería sin
leer la documentación:

>>> import vortilo
>>> nlp = vortilo.load()
>>> doc = nlp("La hundoj kuris.")
>>> [(t.text, t.lemma_, t.pos_) for t in doc]
[('La', 'la', 'DET'), ('hundoj', 'hundo', 'NOUN'), ('kuris', 'kuri', 'VERB'), ('.', '.', 'PUNCT')]

Diferencias con spaCy (simplificaciones deliberadas):

* No hay ``Vocab`` ni ``StringStore``: los atributos son ``str`` normales.
* Los atributos sin guion bajo (``token.pos``, ``token.lemma``) que en spaCy son
  hashes enteros no existen; solo están las versiones ``*_`` de texto.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

__all__ = ["MorphAnalysis", "Token", "Span", "Doc"]

#: Tipos de token que produce el tokenizador.
WORD, NUM, PUNCT, URL, EMAIL, ABBR, ORDINAL, SYM = (
    "WORD", "NUM", "PUNCT", "URL", "EMAIL", "ABBR", "ORDINAL", "SYM",
)


class MorphAnalysis:
    """Rasgos morfológicos de un token, en formato Universal Dependencies.

    Examples:
        >>> m = MorphAnalysis((("Case", "Acc"), ("Number", "Plur")))
        >>> str(m)
        'Case=Acc|Number=Plur'
        >>> m.get("Case")
        ['Acc']
    """

    __slots__ = ("_feats",)

    def __init__(self, feats: Sequence[Tuple[str, str]] = ()):
        self._feats = tuple(sorted(feats))

    def get(self, field: str) -> List[str]:
        """Valores de un rasgo como lista (igual que spaCy). Vacía si no está."""
        for key, value in self._feats:
            if key == field:
                return value.split(",")
        return []

    def to_dict(self) -> Dict[str, str]:
        """Rasgos como diccionario ``{"Case": "Acc", ...}``."""
        return dict(self._feats)

    def __contains__(self, item: str) -> bool:
        # Admite tanto "Case" como "Case=Acc".
        if "=" in item:
            key, value = item.split("=", 1)
            return value in self.get(key)
        return bool(self.get(item))

    def __iter__(self) -> Iterator[str]:
        return (f"{k}={v}" for k, v in self._feats)

    def __len__(self) -> int:
        return len(self._feats)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MorphAnalysis):
            return self._feats == other._feats
        if isinstance(other, str):
            return str(self) == other
        return NotImplemented

    def __str__(self) -> str:
        return "|".join(f"{k}={v}" for k, v in self._feats)

    def __repr__(self) -> str:
        return f"MorphAnalysis({str(self)!r})"


class Token:
    """Una palabra, número o signo de puntuación dentro de un :class:`Doc`.

    Los atributos lingüísticos (``pos_``, ``lemma_``, ``is_stop``…) empiezan
    vacíos y los rellenan los componentes del pipeline. Si un componente está
    desactivado, su atributo queda vacío (``""``) o a ``False``.

    Attributes:
        doc: El documento al que pertenece.
        i: Posición del token en el documento.
        idx: Posición del primer carácter en ``doc.text``.
        text: Texto literal del token.
        kind: Tipo léxico asignado por el tokenizador (``WORD``, ``NUM``,
            ``PUNCT``, ``URL``, ``EMAIL``, ``ABBR``, ``ORDINAL``, ``SYM``).
        norm_: Forma normalizada (minúsculas, sistema X convertido).
        lemma_: Lema.
        pos_: Categoría gramatical UPOS.
        tag_: Etiqueta fina (terminación reconocida, p. ej. ``-ojn``).
        morph: Rasgos morfológicos (:class:`MorphAnalysis`).
        root_: Raíz léxica sin afijos (``lernejo`` → ``lern``).
        prefixes_: Prefijos detectados.
        suffixes_: Sufijos detectados.
        is_stop: ``True`` si es *stop-word*.
        is_sent_start: ``True`` si empieza frase, ``False`` si no, ``None``
            si no se ha segmentado en frases.
    """

    __slots__ = (
        "doc", "i", "idx", "text", "kind", "norm_", "lemma_", "pos_", "tag_",
        "morph", "root_", "prefixes_", "suffixes_", "is_stop", "is_sent_start",
    )

    def __init__(self, doc: "Doc", i: int, idx: int, text: str, kind: str, norm: str):
        self.doc = doc
        self.i = i
        self.idx = idx
        self.text = text
        self.kind = kind
        self.norm_ = norm
        self.lemma_ = ""
        self.pos_ = ""
        self.tag_ = ""
        self.morph = MorphAnalysis()
        self.root_ = ""
        self.prefixes_: Tuple[str, ...] = ()
        self.suffixes_: Tuple[str, ...] = ()
        self.is_stop = False
        self.is_sent_start: Optional[bool] = None

    # -- texto -------------------------------------------------------------- #

    @property
    def whitespace_(self) -> str:
        """Espacio que sigue al token en el texto original (puede ser ``""``)."""
        end = self.idx + len(self.text)
        nxt = self.doc[self.i + 1].idx if self.i + 1 < len(self.doc) else len(self.doc.text)
        return self.doc.text[end:nxt]

    @property
    def text_with_ws(self) -> str:
        """Texto del token seguido de su espacio. Concatenarlos da ``doc.text``."""
        return self.text + self.whitespace_

    @property
    def lower_(self) -> str:
        """Texto en minúsculas (sin normalizar)."""
        return self.text.lower()

    @property
    def shape_(self) -> str:
        """Forma ortográfica: ``Xxxx``, ``dd``, ``xxxx'``… (máx. 4 repeticiones)."""
        out: List[str] = []
        for ch in self.text:
            c = "X" if ch.isupper() else "x" if ch.isalpha() else "d" if ch.isdigit() else ch
            if len(out) >= 4 and all(o == c for o in out[-4:]):
                continue
            out.append(c)
        return "".join(out)

    # -- flags léxicos ------------------------------------------------------ #

    @property
    def is_alpha(self) -> bool:
        """Solo letras."""
        return self.text.isalpha()

    @property
    def is_digit(self) -> bool:
        """Solo dígitos."""
        return self.text.isdigit()

    @property
    def is_punct(self) -> bool:
        """Signo de puntuación."""
        return self.kind == PUNCT

    @property
    def is_space(self) -> bool:
        """Siempre ``False``: el tokenizador no crea tokens de espacio."""
        return False

    @property
    def like_num(self) -> bool:
        """Parece un número (cifra, ordinal o numeral escrito: ``du``, ``dudek``)."""
        return self.kind in (NUM, ORDINAL) or self.pos_ == "NUM"

    @property
    def like_url(self) -> bool:
        """Parece una URL."""
        return self.kind == URL

    @property
    def like_email(self) -> bool:
        """Parece un correo electrónico."""
        return self.kind == EMAIL

    @property
    def is_title(self) -> bool:
        """Empieza por mayúscula y sigue en minúsculas."""
        return self.text.istitle()

    @property
    def is_upper(self) -> bool:
        """Todo en mayúsculas."""
        return self.text.isupper()

    # -- navegación --------------------------------------------------------- #

    def nbor(self, offset: int = 1) -> "Token":
        """Token vecino a ``offset`` posiciones. Lanza ``IndexError`` si no hay."""
        j = self.i + offset
        if j < 0 or j >= len(self.doc):
            raise IndexError(f"No hay token en la posición {j}")
        return self.doc[j]

    @property
    def sent(self) -> "Span":
        """La frase que contiene este token."""
        for sent in self.doc.sents:
            if sent.start <= self.i < sent.end:
                return sent
        return self.doc[:]  # pragma: no cover - sents siempre cubre el doc

    def to_dict(self) -> Dict[str, Any]:
        """Atributos principales como diccionario (útil para pandas)."""
        return {
            "i": self.i,
            "text": self.text,
            "norm": self.norm_,
            "lemma": self.lemma_,
            "pos": self.pos_,
            "tag": self.tag_,
            "morph": str(self.morph),
            "root": self.root_,
            "is_stop": self.is_stop,
            "is_punct": self.is_punct,
            "is_sent_start": self.is_sent_start,
        }

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text


class Span:
    """Un trozo contiguo de un :class:`Doc` (``doc[2:5]`` o una frase)."""

    __slots__ = ("doc", "start", "end")

    def __init__(self, doc: "Doc", start: int, end: int):
        self.doc = doc
        self.start = start
        self.end = end

    @property
    def text(self) -> str:
        """Texto del fragmento, sin el espacio final."""
        if self.start >= self.end:
            return ""
        first, last = self.doc[self.start], self.doc[self.end - 1]
        return self.doc.text[first.idx : last.idx + len(last.text)]

    @property
    def lemma_(self) -> str:
        """Lemas de los tokens unidos por espacios."""
        return " ".join(t.lemma_ for t in self)

    def __iter__(self) -> Iterator[Token]:
        return (self.doc[i] for i in range(self.start, self.end))

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, key: Union[int, slice]) -> Union[Token, "Span"]:
        if isinstance(key, slice):
            start, stop, _ = key.indices(len(self))
            return Span(self.doc, self.start + start, self.start + max(start, stop))
        if key < 0:
            key += len(self)
        if not 0 <= key < len(self):
            raise IndexError("Índice fuera del span")
        return self.doc[self.start + key]

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text


class Doc:
    """Un texto procesado: la secuencia de :class:`Token` y sus frases.

    Normalmente no se construye a mano, sino llamando a ``nlp(texto)``.

    Args:
        text: Texto (ya normalizado) del documento.
        tokens: Lista de ``(idx, texto, tipo, norma)`` producida por el
            tokenizador.
    """

    def __init__(self, text: str, tokens: Sequence[Tuple[int, str, str, str]] = ()):
        self.text = text
        self._tokens: List[Token] = [
            Token(self, i, idx, tok_text, kind, norm)
            for i, (idx, tok_text, kind, norm) in enumerate(tokens)
        ]
        #: Atributos de usuario, como ``doc.user_data`` en spaCy.
        self.user_data: Dict[str, Any] = {}

    # -- secuencia ---------------------------------------------------------- #

    def __iter__(self) -> Iterator[Token]:
        return iter(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, key: Union[int, slice]) -> Union[Token, Span]:
        if isinstance(key, slice):
            start, stop, _ = key.indices(len(self))
            return Span(self, start, max(start, stop))
        return self._tokens[key]

    # -- frases ------------------------------------------------------------- #

    @property
    def has_sents(self) -> bool:
        """``True`` si algún componente ha marcado los inicios de frase."""
        return any(t.is_sent_start is not None for t in self._tokens)

    @property
    def sents(self) -> Iterator[Span]:
        """Frases del documento.

        Si no se ha ejecutado el segmentador, devuelve el documento entero
        como una única frase.
        """
        if not self._tokens:
            return
        start = 0
        for tok in self._tokens[1:]:
            if tok.is_sent_start:
                yield Span(self, start, tok.i)
                start = tok.i
        yield Span(self, start, len(self))

    # -- utilidades --------------------------------------------------------- #

    def count_by(self, attr: str, exclude_punct: bool = True) -> Counter:
        """Cuenta los valores de un atributo de token.

        Examples:
            >>> nlp = __import__("vortilo").load()
            >>> nlp("La hundo kaj la kato.").count_by("pos_")["DET"]
            2
        """
        return Counter(
            getattr(t, attr) for t in self._tokens if not (exclude_punct and t.is_punct)
        )

    def to_records(self) -> List[Dict[str, Any]]:
        """Lista de diccionarios, uno por token: ``pd.DataFrame(doc.to_records())``."""
        return [t.to_dict() for t in self._tokens]

    def to_json(self) -> Dict[str, Any]:
        """Representación serializable del documento."""
        return {
            "text": self.text,
            "tokens": self.to_records(),
            "sents": [[s.start, s.end] for s in self.sents],
        }

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text

    def _repr_json(self) -> str:  # pragma: no cover - ayuda en notebooks
        return json.dumps(self.to_json(), ensure_ascii=False, indent=2)
