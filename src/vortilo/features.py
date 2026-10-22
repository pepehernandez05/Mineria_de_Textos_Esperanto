"""Puente entre la librería y scikit-learn.

``TfidfVectorizer`` y ``CountVectorizer`` aceptan un ``analyzer``: una función
``str -> list[str]`` que convierte cada documento en sus *features*. Aquí se
construye esa función con nuestro pipeline, de modo que el modelo aprende sobre
**lemas** (o raíces) sin *stop-words* ni puntuación, en vez de sobre palabras
crudas::

    from sklearn.feature_extraction.text import TfidfVectorizer
    import vortilo

    analyzer = vortilo.make_analyzer(attr="lemma_", ngram_range=(1, 2))
    vec = TfidfVectorizer(analyzer=analyzer, min_df=2)
    X = vec.fit_transform(df["text"])

Al lematizar, ``hundo``, ``hundoj`` y ``hundojn`` son una sola *feature*: el
vocabulario se reduce y cada *feature* tiene más ejemplos de los que aprender.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from vortilo.pipeline import Language, load

__all__ = ["Analyzer", "make_analyzer"]

#: Atributos de token que se pueden usar como *feature*.
_ATTRS = ("lemma_", "root_", "norm_", "text", "lower_")


class Analyzer:
    """Convierte un texto en la lista de *features* para scikit-learn.

    Es una clase (y no una función anidada) para que se pueda serializar con
    ``pickle``/``joblib`` junto al modelo.

    Args:
        nlp: Pipeline a usar. Por defecto :func:`vortilo.load`.
        attr: Atributo del token: ``"lemma_"`` (por defecto), ``"root_"``,
            ``"norm_"``, ``"lower_"`` o ``"text"``.
        remove_stop: Quitar *stop-words*.
        remove_punct: Quitar puntuación y símbolos.
        remove_numbers: Quitar números (``1950`` no ayuda a distinguir temas
            y dispara el tamaño del vocabulario).
        keep_pos: Si se indica, solo se conservan tokens con esas categorías,
            p. ej. ``("NOUN", "PROPN", "ADJ", "VERB")``.
        min_length: Longitud mínima de la *feature*.
        lowercase: Pasar a minúsculas (afecta a los nombres propios).
        ngram_range: Rango de n-gramas ``(min_n, max_n)`` sobre los tokens ya
            filtrados.
        add_pos: Añadir la categoría a la *feature* (``hundo/NOUN``).
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        attr: str = "lemma_",
        remove_stop: bool = True,
        remove_punct: bool = True,
        remove_numbers: bool = True,
        keep_pos: Optional[Sequence[str]] = None,
        min_length: int = 2,
        lowercase: bool = True,
        ngram_range: Tuple[int, int] = (1, 1),
        add_pos: bool = False,
    ):
        if attr not in _ATTRS:
            raise ValueError(f"attr debe ser uno de {_ATTRS}")
        if ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
            raise ValueError("ngram_range inválido")
        self.nlp = nlp or load()
        self.attr = attr
        self.remove_stop = remove_stop
        self.remove_punct = remove_punct
        self.remove_numbers = remove_numbers
        self.keep_pos = frozenset(keep_pos) if keep_pos else None
        self.min_length = min_length
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.add_pos = add_pos

    def tokens(self, text: str) -> List[str]:
        """Unigramas filtrados de ``text``."""
        out: List[str] = []
        for tok in self.nlp(text):
            if self.remove_punct and (tok.is_punct or tok.pos_ in ("PUNCT", "SYM")):
                continue
            if self.remove_stop and tok.is_stop:
                continue
            if self.remove_numbers and tok.like_num:
                continue
            if self.keep_pos is not None and tok.pos_ not in self.keep_pos:
                continue
            value = getattr(tok, self.attr)
            if self.lowercase:
                value = value.lower()
            if len(value) < self.min_length:
                continue
            if self.add_pos:
                value = f"{value}/{tok.pos_}"
            out.append(value)
        return out

    def __call__(self, text: str) -> List[str]:
        unigrams = self.tokens(text)
        min_n, max_n = self.ngram_range
        if max_n == 1:
            return unigrams
        features: List[str] = unigrams[:] if min_n == 1 else []
        for n in range(max(2, min_n), max_n + 1):
            features.extend(" ".join(unigrams[i : i + n]) for i in range(len(unigrams) - n + 1))
        return features

    def __repr__(self) -> str:
        return (
            f"Analyzer(attr={self.attr!r}, remove_stop={self.remove_stop}, "
            f"keep_pos={sorted(self.keep_pos) if self.keep_pos else None}, ngram_range={self.ngram_range})"
        )


def make_analyzer(nlp: Optional[Language] = None, **kwargs) -> Analyzer:
    """Atajo para crear un :class:`Analyzer`.

    Examples:
        >>> an = make_analyzer()
        >>> an("La hundoj kuris en la parko.")
        ['hundo', 'kuri', 'parko']
    """
    return Analyzer(nlp, **kwargs)
