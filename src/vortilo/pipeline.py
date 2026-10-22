"""El objeto ``nlp`` y los componentes del pipeline.

Igual que en spaCy, procesar un texto es pasarlo por una cadena de pasos::

    texto ──► Tokenizer ──► senter ──► tagger ──► lemmatizer ──► stopwords ──► Doc

* ``Tokenizer``: crea el :class:`~vortilo.tokens.Doc` (siempre se ejecuta).
* ``senter``: marca inicios de frase (``Token.is_sent_start``).
* ``tagger``: categoría gramatical (``pos_``), etiqueta fina (``tag_``) y
  rasgos (``morph``). Decide también los nombres propios (``PROPN``).
* ``lemmatizer``: lema (``lemma_``) y raíz (``root_``).
* ``stopwords``: marca ``is_stop``.

Cada componente es simplemente una función ``Doc -> Doc``, así que se pueden
quitar, desactivar o añadir los tuyos::

    nlp = vortilo.load()
    nlp.pipe_names                       # ['senter', 'tagger', 'lemmatizer', 'stopwords']
    doc = nlp(texto, disable=["senter"]) # sin segmentar en frases
    nlp.add_pipe(mi_componente, name="mio", last=True)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from vortilo.morphology import analyze
from vortilo.stopwords import STOP_WORDS
from vortilo.tokenizer import SENTENCE_END, Tokenizer, mark_sentence_starts
from vortilo.tokens import ABBR, EMAIL, NUM, ORDINAL, PUNCT, SYM, URL, WORD, Doc, MorphAnalysis, Token

__all__ = ["Language", "load", "blank", "SentenceSegmenter", "Tagger", "Lemmatizer", "StopWordMarker"]

Component = Callable[[Doc], Doc]

#: Categorías que nunca se convierten en nombre propio aunque vayan en mayúscula
#: (``La``, ``Kaj``, ``En`` en un título siguen siendo lo que son).
_CLOSED_POS = frozenset({"DET", "PRON", "ADP", "CCONJ", "SCONJ", "PART", "INTJ", "NUM", "AUX"})


def _is_sentence_initial(tok: Token) -> bool:
    """¿Es la primera palabra de su frase?

    Usa ``is_sent_start`` si el segmentador se ha ejecutado; si no, mira si el
    token anterior cierra frase. Se saltan comillas/paréntesis de apertura.
    """
    j = tok.i - 1
    doc = tok.doc
    while j >= 0 and doc[j].kind == PUNCT and doc[j].text not in SENTENCE_END:
        j -= 1
    if j < 0:
        return True
    if tok.is_sent_start is not None:
        # Si hay segmentación, la frase empieza en el token tras la apertura.
        k = j + 1
        return bool(doc[k].is_sent_start)
    return doc[j].text in SENTENCE_END


def _restore_case(original: str, lemma: str) -> str:
    """Recupera las mayúsculas del original en el lema: ``Esperanton`` → ``Esperanto``."""
    if original.lower().startswith(lemma):
        return original[: len(lemma)]
    return lemma[:1].upper() + lemma[1:]


# --------------------------------------------------------------------------- #
# Componentes
# --------------------------------------------------------------------------- #


class SentenceSegmenter:
    """Componente ``senter``: segmenta en frases. Ver
    :func:`vortilo.tokenizer.mark_sentence_starts`."""

    name = "senter"

    def __call__(self, doc: Doc) -> Doc:
        return mark_sentence_starts(doc)


class Tagger:
    """Componente ``tagger``: asigna ``pos_``, ``tag_`` y ``morph``.

    Reglas, en orden:

    1. Tokens no-palabra según su tipo léxico: puntuación → ``PUNCT``,
       símbolo → ``SYM``, número → ``NUM``, ordinal con cifras → ``ADJ``,
       URL/correo → ``X``.
    2. Palabras: :func:`vortilo.morphology.analyze` (sin contexto).
    3. Contexto: una palabra en mayúscula que **no** es de clase cerrada pasa a
       ``PROPN`` si está en mitad de frase, si está toda en mayúsculas (siglas)
       o si su terminación no es esperanta (``Madrid``).
    """

    name = "tagger"

    def __call__(self, doc: Doc) -> Doc:
        for tok in doc:
            if tok.kind == PUNCT:
                tok.pos_, tok.tag_ = "PUNCT", "PUNCT"
                continue
            if tok.kind == SYM:
                tok.pos_, tok.tag_ = "SYM", "SYM"
                continue
            if tok.kind == NUM:
                tok.pos_, tok.tag_ = "NUM", "NUM"
                tok.morph = MorphAnalysis((("NumType", "Card"),))
                continue
            if tok.kind == ORDINAL:
                tok.pos_, tok.tag_ = "ADJ", "ORD"
                tok.morph = MorphAnalysis((("NumType", "Ord"),))
                continue
            if tok.kind in (URL, EMAIL):
                tok.pos_, tok.tag_ = "X", tok.kind
                continue

            analysis = analyze(tok.norm_)
            pos = analysis.pos
            if tok.text[:1].isupper() and pos not in _CLOSED_POS and tok.kind == WORD:
                acronym = tok.is_upper and len(tok.text) > 1
                if acronym or pos == "X" or not _is_sentence_initial(tok):
                    pos = "PROPN"
            tok.pos_ = pos
            tok.tag_ = analysis.tag
            tok.morph = MorphAnalysis(analysis.feats)
        return doc


class Lemmatizer:
    """Componente ``lemmatizer``: asigna ``lemma_``, ``root_``, ``prefixes_`` y ``suffixes_``.

    El lema sale de :func:`vortilo.morphology.analyze`. Para los nombres propios
    (si el ``tagger`` los ha marcado) se conservan las mayúsculas:
    ``Esperanton`` → ``Esperanto``.
    """

    name = "lemmatizer"

    def __call__(self, doc: Doc) -> Doc:
        for tok in doc:
            if tok.kind in (PUNCT, SYM, NUM, URL, EMAIL):
                tok.lemma_ = tok.root_ = tok.text
                continue
            if tok.kind == ORDINAL:
                lemma = tok.norm_.rstrip("n").rstrip("j")  # 20-ajn -> 20-a
                tok.lemma_ = tok.root_ = lemma
                continue

            analysis = analyze(tok.norm_)
            if tok.pos_ == "PROPN":
                tok.lemma_ = _restore_case(tok.text, analysis.lemma)
                tok.root_ = analysis.root or analysis.lemma
            else:
                tok.lemma_ = analysis.lemma
                tok.root_ = analysis.root or analysis.lemma
            tok.prefixes_ = analysis.prefixes
            tok.suffixes_ = analysis.suffixes
        return doc


class StopWordMarker:
    """Componente ``stopwords``: marca ``is_stop``.

    Un token es *stop-word* si su forma normalizada **o** su lema están en la
    lista. La lista es la de ``nlp.stop_words`` (modificable en caliente).
    """

    name = "stopwords"

    def __init__(self, stop_words: set):
        self.stop_words = stop_words

    def __call__(self, doc: Doc) -> Doc:
        sw = self.stop_words
        for tok in doc:
            if tok.kind in (PUNCT, SYM):
                tok.is_stop = False
                continue
            tok.is_stop = tok.norm_ in sw or tok.lemma_.lower() in sw
        return doc


# --------------------------------------------------------------------------- #
# Language
# --------------------------------------------------------------------------- #


class Language:
    """El objeto ``nlp``: tokenizador + lista ordenada de componentes.

    Normalmente se crea con :func:`load`. Es invocable: ``nlp(texto) -> Doc``.

    Args:
        tokenizer: Tokenizador a usar. Por defecto :class:`~vortilo.tokenizer.Tokenizer`.
        stop_words: Lista de *stop-words*. Por defecto una copia de
            :data:`vortilo.stopwords.STOP_WORDS`.
    """

    lang = "eo"

    def __init__(self, tokenizer: Optional[Tokenizer] = None, stop_words: Optional[Iterable[str]] = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.stop_words = set(STOP_WORDS if stop_words is None else stop_words)
        self._components: List[Tuple[str, Component]] = []
        self._disabled: set = set()

    # -- fábricas por nombre (como nlp.add_pipe("tagger") en spaCy) ---------- #

    def _create(self, factory: str) -> Component:
        factories: Dict[str, Callable[[], Component]] = {
            "senter": SentenceSegmenter,
            "tagger": Tagger,
            "lemmatizer": Lemmatizer,
            "stopwords": lambda: StopWordMarker(self.stop_words),
        }
        if factory not in factories:
            raise ValueError(f"Componente desconocido {factory!r}. Opciones: {sorted(factories)}")
        return factories[factory]()

    # -- gestión del pipeline ------------------------------------------------ #

    @property
    def pipe_names(self) -> List[str]:
        """Nombres de los componentes activos, en orden."""
        return [n for n, _ in self._components if n not in self._disabled]

    @property
    def components(self) -> List[Tuple[str, Component]]:
        """Pares ``(nombre, componente)``, incluidos los desactivados."""
        return list(self._components)

    def has_pipe(self, name: str) -> bool:
        """¿Existe un componente con ese nombre?"""
        return any(n == name for n, _ in self._components)

    def get_pipe(self, name: str) -> Component:
        """Devuelve el componente ``name``."""
        for n, comp in self._components:
            if n == name:
                return comp
        raise KeyError(f"No hay componente {name!r}")

    def add_pipe(
        self,
        component: "str | Component",
        name: Optional[str] = None,
        *,
        before: Optional[str] = None,
        after: Optional[str] = None,
        first: bool = False,
        last: bool = False,
    ) -> Component:
        """Añade un componente al pipeline.

        Args:
            component: Nombre de una fábrica (``"senter"``, ``"tagger"``,
                ``"lemmatizer"``, ``"stopwords"``) o cualquier función
                ``Doc -> Doc``.
            name: Nombre con el que se registra. Por defecto el de la fábrica,
                el atributo ``name`` del componente o el de la función.
            before, after: Insertar antes/después de otro componente.
            first, last: Insertar al principio/al final (por defecto, al final).

        Returns:
            El componente añadido.
        """
        if isinstance(component, str):
            name = name or component
            component = self._create(component)
        name = name or getattr(component, "name", None) or getattr(component, "__name__", "component")
        if self.has_pipe(name):
            raise ValueError(f"Ya existe un componente llamado {name!r}")
        if sum(bool(x) for x in (before, after, first, last)) > 1:
            raise ValueError("Usa solo uno de before/after/first/last")

        names = [n for n, _ in self._components]
        if first:
            pos = 0
        elif before:
            pos = names.index(before)
        elif after:
            pos = names.index(after) + 1
        else:
            pos = len(names)
        self._components.insert(pos, (name, component))
        return component

    def remove_pipe(self, name: str) -> Tuple[str, Component]:
        """Quita un componente y lo devuelve."""
        for k, (n, comp) in enumerate(self._components):
            if n == name:
                self._disabled.discard(name)
                return self._components.pop(k)
        raise KeyError(f"No hay componente {name!r}")

    def disable_pipe(self, name: str) -> None:
        """Desactiva un componente sin quitarlo."""
        self.get_pipe(name)
        self._disabled.add(name)

    def enable_pipe(self, name: str) -> None:
        """Reactiva un componente."""
        self.get_pipe(name)
        self._disabled.discard(name)

    @contextmanager
    def select_pipes(self, disable: Sequence[str] = (), enable: Optional[Sequence[str]] = None):
        """Desactiva componentes temporalmente (``with nlp.select_pipes(disable=["senter"]):``)."""
        before = set(self._disabled)
        if enable is not None:
            self._disabled |= {n for n, _ in self._components if n not in enable}
        self._disabled |= set(disable)
        try:
            yield self
        finally:
            self._disabled = before

    # -- procesado ----------------------------------------------------------- #

    def make_doc(self, text: str) -> Doc:
        """Solo tokeniza (sin componentes)."""
        return self.tokenizer(text)

    def __call__(self, text: str, disable: Sequence[str] = ()) -> Doc:
        """Procesa ``text`` con todos los componentes activos."""
        doc = self.make_doc(text)
        for name, comp in self._components:
            if name in self._disabled or name in disable:
                continue
            doc = comp(doc)
        return doc

    def pipe(self, texts: Iterable[str], disable: Sequence[str] = (), batch_size: int = 1000) -> Iterator[Doc]:
        """Procesa muchos textos de forma perezosa (generador).

        ``batch_size`` se acepta por compatibilidad con spaCy pero no se usa.
        """
        for text in texts:
            yield self(text, disable=disable)

    def __repr__(self) -> str:
        return f"<vortilo.Language lang='eo' pipeline={self.pipe_names}>"


def blank() -> Language:
    """``nlp`` vacío: solo tokenizador, sin componentes."""
    return Language()


def load(name: str = "eo", *, disable: Sequence[str] = (), stop_words: Optional[Iterable[str]] = None) -> Language:
    """Crea el pipeline completo, como ``spacy.load``.

    Args:
        name: Se acepta por compatibilidad (``"eo"``); solo hay un modelo.
        disable: Componentes que no se añaden.
        stop_words: Lista de *stop-words* alternativa.

    Returns:
        Un :class:`Language` con ``senter``, ``tagger``, ``lemmatizer`` y ``stopwords``.

    Examples:
        >>> nlp = load()
        >>> nlp.pipe_names
        ['senter', 'tagger', 'lemmatizer', 'stopwords']
    """
    if name not in ("eo", "esperanto", "default"):
        raise ValueError(f"Modelo desconocido {name!r}; usa 'eo'")
    nlp = Language(stop_words=stop_words)
    for comp in ("senter", "tagger", "lemmatizer", "stopwords"):
        if comp not in disable:
            nlp.add_pipe(comp)
    return nlp
