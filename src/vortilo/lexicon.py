"""Léxico cerrado del Esperanto.

El Esperanto tiene dos tipos de palabras:

* **Palabras de clase abierta** (sustantivos, adjetivos, verbos, adverbios
  derivados): su categoría se deduce de la terminación (``-o``, ``-a``, ``-i``…).
  De eso se encarga :mod:`vortilo.morphology`.
* **Palabras de clase cerrada** (artículo, pronombres, preposiciones,
  conjunciones, correlativos, numerales, partículas y adverbios primitivos): son
  una lista finita y *no* siguen las terminaciones. Por ejemplo, ``kaj`` no es
  un verbo aunque acabe en ``-j`` ni ``tre`` un adverbio derivado.

Este módulo contiene esas listas. Se consultan **antes** de aplicar las reglas
de terminaciones, lo que evita la mayoría de errores del etiquetador.

Las etiquetas siguen el estándar Universal Dependencies (UPOS), igual que spaCy.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = [
    "ARTICLE",
    "PERSONAL_PRONOUNS",
    "PREPOSITIONS",
    "COORD_CONJUNCTIONS",
    "SUBORD_CONJUNCTIONS",
    "PARTICLES",
    "PRIMITIVE_ADVERBS",
    "INTERJECTIONS",
    "NUMERALS",
    "CORRELATIVE_PREFIXES",
    "CORRELATIVE_SUFFIXES",
    "CORRELATIVES",
    "ABBREVIATIONS",
    "PREFIXES",
    "SUFFIXES",
    "PARTICIPLE_SUFFIXES",
    "ROOT_EXCEPTIONS",
    "closed_class",
]

# --------------------------------------------------------------------------- #
# Listas básicas
# --------------------------------------------------------------------------- #

#: Único artículo del Esperanto (definido). No hay artículo indefinido.
ARTICLE = frozenset({"la"})

#: Pronombres personales. Admiten acusativo (``min``) y, añadiendo ``-a``,
#: forman posesivos (``mia``, ``miajn``).
PERSONAL_PRONOUNS = frozenset(
    {"mi", "ci", "vi", "li", "ŝi", "ĝi", "ni", "ili", "oni", "si"}
)

#: Preposiciones (ADP).
PREPOSITIONS = frozenset(
    {
        "al", "anstataŭ", "antaŭ", "apud", "ĉe", "ĉirkaŭ", "da", "de", "dum",
        "ekster", "el", "en", "ĝis", "inter", "je", "kontraŭ", "krom", "kun",
        "laŭ", "malgraŭ", "per", "po", "por", "post", "preter", "pri", "pro",
        "sen", "sub", "super", "sur", "tra", "trans", "cis", "ekde", "fare",
        "pere", "far",
    }
)

#: Conjunciones coordinantes (CCONJ).
COORD_CONJUNCTIONS = frozenset({"kaj", "aŭ", "sed", "nek", "do", "tamen", "plus", "minus"})

#: Conjunciones subordinantes (SCONJ).
SUBORD_CONJUNCTIONS = frozenset({"ke", "se", "ĉar", "kvankam", "kvazaŭ", "ol", "dum", "ĝis", "apenaŭ"})

#: Partículas (PART): negación, interrogación, enfáticas…
PARTICLES = frozenset({"ne", "ĉu", "ja", "jen", "ajn", "ĉi", "jes", "mem", "des", "ju"})

#: Adverbios primitivos: adverbios que no terminan en ``-e`` o que no se
#: derivan de una raíz (``tre``, ``nun``, ``ankaŭ``…).
PRIMITIVE_ADVERBS = frozenset(
    {
        "almenaŭ", "ambaŭ", "ankaŭ", "ankoraŭ", "baldaŭ", "eĉ", "for", "hieraŭ",
        "hodiaŭ", "jam", "morgaŭ", "nun", "nur", "plej", "pli", "plu", "preskaŭ",
        "tre", "tro", "tuj", "ĵus", "ankaŭ", "kvazaŭ", "adiaŭ", "malpli",
        "malplej", "tamen",
    }
)

#: Interjecciones (INTJ).
INTERJECTIONS = frozenset({"ha", "he", "ho", "hu", "nu", "ve", "fi", "aĥ", "ve", "bis", "hura"})

#: Numerales básicos. Los compuestos (``dudek``, ``tricent``) se reconocen con
#: una expresión regular en :mod:`vortilo.morphology`.
NUMERALS = frozenset(
    {"nul", "unu", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ", "dek", "cent", "mil"}
)

# --------------------------------------------------------------------------- #
# Tabla de correlativos
# --------------------------------------------------------------------------- #

#: Inicio del correlativo -> significado. ``ki-`` interrogativo/relativo,
#: ``ti-`` demostrativo, ``i-`` indefinido, ``ĉi-`` universal, ``neni-`` negativo.
CORRELATIVE_PREFIXES = ("ki", "ti", "i", "ĉi", "neni")

#: Final del correlativo -> etiqueta UPOS y si admite ``-j`` (plural) y ``-n``.
#: ``(upos, admite_plural, admite_acusativo)``
CORRELATIVE_SUFFIXES: Dict[str, Tuple[str, bool, bool]] = {
    "u": ("DET", True, True),     # persona / individuo: kiu, tiu
    "o": ("PRON", False, True),   # cosa: kio, tio
    "a": ("DET", True, True),     # cualidad: kia, tia
    "es": ("DET", False, False),  # posesión: kies, ties
    "e": ("ADV", False, True),    # lugar: kie, tie  (kien = dirección)
    "el": ("ADV", False, False),  # modo: kiel, tiel
    "am": ("ADV", False, False),  # tiempo: kiam, tiam
    "om": ("ADV", False, False),  # cantidad: kiom, tiom
    "al": ("ADV", False, False),  # causa: kial, tial
}

#: Las 45 formas base de la tabla de correlativos y su etiqueta.
CORRELATIVES: Dict[str, str] = {
    prefix + suffix: info[0]
    for prefix in CORRELATIVE_PREFIXES
    for suffix, info in CORRELATIVE_SUFFIXES.items()
}

# --------------------------------------------------------------------------- #
# Abreviaturas frecuentes
# --------------------------------------------------------------------------- #

#: Abreviatura (en minúsculas, con punto) -> (lema expandido, UPOS).
#: El tokenizador las mantiene como un único token para no cortar la frase en
#: su punto.
ABBREVIATIONS: Dict[str, Tuple[str, str]] = {
    "ekz.": ("ekzemple", "ADV"),
    "k.t.p.": ("kaj tiel plu", "ADV"),
    "ktp.": ("kaj tiel plu", "ADV"),
    "ktp": ("kaj tiel plu", "ADV"),
    "t.e.": ("tio estas", "ADV"),
    "k.a.": ("kaj aliaj", "ADV"),
    "k.s.": ("kaj simile", "ADV"),
    "kp.": ("komparu", "VERB"),
    "vd.": ("vidu", "VERB"),
    "resp.": ("respektive", "ADV"),
    "p.k.": ("post Kristo", "ADV"),
    "a.k.": ("antaŭ Kristo", "ADV"),
    "n.": ("naskiĝis", "VERB"),
    "m.": ("mortis", "VERB"),
    "ĉ.": ("ĉirkaŭ", "ADP"),
    "proks.": ("proksimume", "ADV"),
    "s-ro": ("sinjoro", "NOUN"),
    "s-ino": ("sinjorino", "NOUN"),
    "d-ro": ("doktoro", "NOUN"),
    "d-rino": ("doktorino", "NOUN"),
    "prof.": ("profesoro", "NOUN"),
    "n-ro": ("numero", "NOUN"),
    "j.": ("jaro", "NOUN"),
    "jc.": ("jarcento", "NOUN"),
    "km": ("kilometro", "NOUN"),
}

# --------------------------------------------------------------------------- #
# Afijos (para la raíz, ``Token.root_``)
# --------------------------------------------------------------------------- #

# Decisión de diseño: el Esperanto tiene ~10 prefijos y ~30 sufijos, pero
# quitarlos todos a ciegas destroza muchos préstamos internacionales:
#   geografio -> ge+ografio, ekonomio -> ek+onomio, fiziko -> fi+ziko,
#   planedo/planeto -> plan+et, maŝino -> maŝ+in, proteino -> prote+in…
# Como el objetivo de la raíz es agrupar palabras de la misma familia para el
# clasificador, preferimos PRECISIÓN a COBERTURA: solo se quitan los afijos
# menos ambiguos y se mantiene una lista de excepciones conocidas.

#: Prefijos que se quitan. Solo ``mal-`` (contrario) es lo bastante seguro.
PREFIXES = ("mal",)

#: Sufijos que se quitan, de más largo a más corto (``-estr`` antes que ``-ist``…).
#: Se descartaron por ambiguos: -in, -et, -eg, -em, -er, -on, -id, -end, -um.
SUFFIXES = tuple(
    sorted(
        (
            "ad", "aĵ", "an", "ar", "ebl", "ec", "ej", "estr", "ig", "iĝ",
            "il", "ind", "ism", "ist", "uj", "ul",
        ),
        key=len,
        reverse=True,
    )
)

#: Raíces que *parecen* llevar afijo pero no lo llevan. No se descomponen.
ROOT_EXCEPTIONS = frozenset(
    {
        # -ad
        "parad", "olimpiad", "balad", "nomad", "fasad", "salad", "limonad", "brigad",
        # -an
        "ocean", "organ", "roman", "titan", "vulkan", "kapitan", "sultan", "membran",
        "metan", "etan", "propan", "butan", "uran", "german", "pelikan", "fazan",
        "tukan", "ŝaman", "satan", "urban", "kurtan", "turban", "orangutan",
        # -ar
        "dolar", "kalendar", "bazar", "cigar", "gitar", "jaguar", "polar", "solar",
        "lunar", "molar", "radar", "nektar", "sonar", "kaviar", "bulgar", "hungar",
        "tatar", "altar", "notar", "vikar", "seminar", "kancelar",
        # -ist
        "ekzist", "insist", "rezist", "asist", "konsist", "persist",
        # -ig
        "navig", "intrig",
        # -estr
        "orkestr", "semestr",
        # -il
        "april", "brazil", "civil", "krokodil", "simil", "facil", "humil", "fertil",
        "konsil", "goril", "profil", "argil", "subtil", "bacil", "pupil",
        # -ul
        "insul", "regul", "modul", "formul", "molekul", "partikul", "artikul",
        "kalkul", "cirkul", "angul", "simul", "stimul", "spekul", "vehikul",
        "kapsul", "konsul", "fabul", "ampul", "muskul", "nebul", "postul",
        "akumul", "manipul", "titul", "glandul", "tabul", "popul",
        # mal-
        "malari", "malt", "malv",
    }
)

#: Sufijos de participio: (sufijo, tiempo, voz).
PARTICIPLE_SUFFIXES = (
    ("ant", "Pres", "Act"),
    ("int", "Past", "Act"),
    ("ont", "Fut", "Act"),
    ("at", "Pres", "Pass"),
    ("it", "Past", "Pass"),
    ("ot", "Fut", "Pass"),
)


def closed_class(word: str) -> str | None:
    """Devuelve la etiqueta UPOS de ``word`` si es una palabra de clase cerrada.

    Solo mira formas base (sin ``-j``/``-n``); las formas flexionadas de
    pronombres y correlativos se resuelven en :mod:`vortilo.morphology`.

    Args:
        word: Palabra en minúsculas y normalizada.

    Returns:
        La etiqueta UPOS, o ``None`` si la palabra es de clase abierta.

    Examples:
        >>> closed_class("kaj")
        'CCONJ'
        >>> closed_class("tiu")
        'DET'
        >>> closed_class("hundo") is None
        True
    """
    if word in ARTICLE:
        return "DET"
    if word in PERSONAL_PRONOUNS:
        return "PRON"
    if word in CORRELATIVES:
        return CORRELATIVES[word]
    if word in PREPOSITIONS:
        return "ADP"
    if word in COORD_CONJUNCTIONS:
        return "CCONJ"
    if word in SUBORD_CONJUNCTIONS:
        return "SCONJ"
    if word in PARTICLES:
        return "PART"
    if word in PRIMITIVE_ADVERBS:
        return "ADV"
    if word in INTERJECTIONS:
        return "INTJ"
    if word in NUMERALS:
        return "NUM"
    return None
