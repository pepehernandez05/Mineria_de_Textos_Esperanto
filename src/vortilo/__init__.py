"""vortilo: librería de NLP basada en reglas para Esperanto, con API tipo spaCy.

Uso básico::

    import vortilo

    nlp = vortilo.load()
    doc = nlp("La hundoj kuris en la parko. Ĉu vi vidis ilin?")

    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.morph, token.is_stop)

    for sent in doc.sents:
        print(sent.text)

Módulos:

=====================  =========================================================
``vortilo.normalize``  Sistema X/H, Unicode, comillas tipográficas.
``vortilo.tokenizer``  Tokenizador y segmentador de frases.
``vortilo.lexicon``    Palabras de clase cerrada, correlativos, afijos.
``vortilo.morphology`` Análisis por reglas: POS, lema, rasgos y raíz.
``vortilo.stopwords``  Lista de *stop-words*.
``vortilo.tokens``     ``Doc``, ``Token``, ``Span``, ``MorphAnalysis``.
``vortilo.pipeline``   ``Language`` (el objeto ``nlp``) y sus componentes.
``vortilo.clean``      Limpieza de HTML/Wikipedia y división en párrafos.
``vortilo.features``   ``Analyzer`` para usar la librería con scikit-learn.
=====================  =========================================================
"""

from vortilo.clean import clean_text, split_paragraphs
from vortilo.features import Analyzer, make_analyzer
from vortilo.morphology import Analysis, analyze
from vortilo.normalize import normalize_text, x_to_unicode
from vortilo.pipeline import Language, blank, load
from vortilo.stopwords import STOP_WORDS, build_stop_words
from vortilo.tokenizer import Tokenizer, sentence_tokenizer, word_tokenizer
from vortilo.tokens import Doc, MorphAnalysis, Span, Token

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load",
    "blank",
    "Language",
    "Doc",
    "Token",
    "Span",
    "MorphAnalysis",
    "Tokenizer",
    "word_tokenizer",
    "sentence_tokenizer",
    "analyze",
    "Analysis",
    "STOP_WORDS",
    "build_stop_words",
    "normalize_text",
    "x_to_unicode",
    "clean_text",
    "split_paragraphs",
    "Analyzer",
    "make_analyzer",
]
