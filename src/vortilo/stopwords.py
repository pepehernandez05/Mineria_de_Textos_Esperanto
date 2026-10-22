"""Gestión de *stop-words*.

Las *stop-words* son palabras muy frecuentes y con poco contenido temático
(``la``, ``kaj``, ``de``…). Para clasificar por temática conviene quitarlas,
porque aparecen igual en todas las clases y solo añaden ruido.

La lista por defecto se construye a partir del léxico cerrado
(:mod:`vortilo.lexicon`) más unos pocos verbos y adverbios muy generales. Se
compara contra el **lema** y contra la forma normalizada, así que ``estas``,
``estis`` y ``esti`` cuentan todas como *stop-word* (lema ``esti``).

La lista es un ``set`` normal y se puede modificar en el ``nlp``::

    nlp = vortilo.load()
    nlp.stop_words.add("jaro")       # añadir
    nlp.stop_words.discard("ne")      # quitar
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Set

from vortilo import lexicon

__all__ = ["STOP_WORDS", "build_stop_words"]

#: Posesivos: mia, via, lia…
_POSSESSIVES = frozenset(p + "a" for p in lexicon.PERSONAL_PRONOUNS)

#: Verbos y otras palabras de contenido muy genérico.
_GENERIC = frozenset(
    {
        # verbos auxiliares / modales muy frecuentes (como lemas)
        "esti", "havi", "povi", "devi", "voli", "fari", "iĝi", "igi",
        # adverbios y adjetivos genéricos
        "ekzemple", "eble", "male", "nepre", "supre", "tute", "verŝajne", "precipe",
        "alia", "sama", "multa", "multe", "kelka", "pluraj", "ĉefe", "tiel", "nur",
        # palabras de cortesía / tratamiento
        "bonvolu", "dankon", "sinjoro", "sinjorino", "fraŭlino", "doktoro",
    }
)

#: Lista por defecto (inmutable). Úsala como base para la tuya.
STOP_WORDS: FrozenSet[str] = frozenset().union(
    lexicon.ARTICLE,
    lexicon.PERSONAL_PRONOUNS,
    _POSSESSIVES,
    lexicon.PREPOSITIONS,
    lexicon.COORD_CONJUNCTIONS,
    lexicon.SUBORD_CONJUNCTIONS,
    lexicon.PARTICLES,
    lexicon.PRIMITIVE_ADVERBS,
    lexicon.INTERJECTIONS,
    lexicon.NUMERALS,
    lexicon.CORRELATIVES,
    _GENERIC,
    {"l'"},
)


def build_stop_words(extra: Iterable[str] = (), remove: Iterable[str] = ()) -> Set[str]:
    """Crea una lista de *stop-words* a partir de la lista por defecto.

    Args:
        extra: Palabras (lemas) que añadir; por ejemplo, palabras genéricas de
            tu corpus como ``jaro`` o ``nomo``.
        remove: Palabras que quitar de la lista por defecto.

    Returns:
        Un ``set`` nuevo y modificable.

    Examples:
        >>> sw = build_stop_words(extra={"jaro"}, remove={"ne"})
        >>> "jaro" in sw, "ne" in sw, "kaj" in sw
        (True, False, True)
    """
    words = set(STOP_WORDS)
    words.update(w.lower() for w in extra)
    words.difference_update(w.lower() for w in remove)
    return words
