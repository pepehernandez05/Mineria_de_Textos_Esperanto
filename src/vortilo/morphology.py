"""Análisis morfológico basado en reglas.

Es el corazón de la librería: dada una palabra, decide su categoría gramatical
(POS), su lema, sus rasgos morfológicos y su raíz. Lo usan tanto el etiquetador
como el lematizador.

El orden de las reglas importa y es el siguiente:

1. **Clase cerrada** (:mod:`vortilo.lexicon`): artículo, pronombres,
   preposiciones, correlativos… Se miran primero porque no siguen las
   terminaciones (``kaj`` no es un plural, ``tre`` no es un adverbio en ``-e``).
2. **Formas flexionadas de clase cerrada**: pronombres en acusativo (``min``),
   posesivos (``miajn``), correlativos flexionados (``kiujn``, ``kien``).
3. **Numerales**, simples y compuestos (``dudek``, ``kvincent``).
4. **Clase abierta por terminación**, tras quitar ``-n`` (acusativo) y ``-j``
   (plural):

   ======== ======== ==============================================
   Final    UPOS     Ejemplo
   ======== ======== ==============================================
   ``-o``   NOUN     ``hundojn`` → ``hundo``
   ``-a``   ADJ      ``belaj`` → ``bela``
   ``-e``   ADV      ``rapide`` → ``rapide``; ``hejmen`` → ``hejme``
   ``-as``  VERB     presente: ``kuras`` → ``kuri``
   ``-is``  VERB     pasado: ``kuris`` → ``kuri``
   ``-os``  VERB     futuro: ``kuros`` → ``kuri``
   ``-us``  VERB     condicional: ``kurus`` → ``kuri``
   ``-u``   VERB     imperativo/volitivo: ``kuru`` → ``kuri``
   ``-i``   VERB     infinitivo: ``kuri``
   ``-'``   NOUN     elisión poética: ``dom'`` → ``domo``
   ======== ======== ==============================================

5. Si nada encaja: ``X`` (palabra extranjera o desconocida).

Los nombres propios (``PROPN``) dependen del contexto (mayúscula en mitad de
frase), así que los decide el etiquetador en :mod:`vortilo.pipeline`, no aquí.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Tuple

from vortilo import lexicon

__all__ = ["Analysis", "analyze", "split_affixes"]

#: Numeral compuesto escrito junto: dudek, tricent, dekdu, dumil…
_UNITS = "unu|du|tri|kvar|kvin|ses|sep|ok|naŭ"
_COMPOUND_NUMERAL = re.compile(rf"^(?:(?:{_UNITS})?(?:dek|cent|mil))+(?:{_UNITS})?$")

#: Tipo de pronombre de cada inicio de correlativo (rasgo ``PronType`` de UD).
_PRONTYPE = {"ki": "Int,Rel", "ti": "Dem", "i": "Ind", "ĉi": "Tot", "neni": "Neg"}

#: Terminaciones verbales -> rasgos.
_VERB_ENDINGS = (
    ("as", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Pres"}),
    ("is", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Past"}),
    ("os", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Fut"}),
    ("us", {"VerbForm": "Fin", "Mood": "Cnd"}),
    ("u", {"VerbForm": "Fin", "Mood": "Jus"}),
    ("i", {"VerbForm": "Inf"}),
)

#: Verbos que funcionan como auxiliares (``estas farita``, ``havis``…).
_AUXILIARIES = frozenset({"esti"})

#: Longitud mínima de la raíz para aceptar una terminación de clase abierta.
#: Evita analizar ``bo`` o ``ha`` como si fueran palabras derivadas.
_MIN_STEM = 2

#: Longitud mínima que debe quedar al quitar un afijo.
_MIN_ROOT = 3


@dataclass(frozen=True)
class Analysis:
    """Resultado del análisis de una palabra.

    Attributes:
        pos: Etiqueta UPOS (``NOUN``, ``VERB``…).
        lemma: Forma de diccionario (``hundojn`` → ``hundo``).
        tag: Etiqueta fina: la terminación gramatical reconocida (``-ojn``),
            o la categoría léxica para clase cerrada.
        feats: Rasgos morfológicos ordenados, al estilo UD (``Case=Acc``).
        root: Raíz léxica sin terminación ni afijos (``lernejo`` → ``lern``).
        prefixes: Prefijos quitados para llegar a la raíz.
        suffixes: Sufijos quitados para llegar a la raíz (en orden de lectura).
    """

    pos: str
    lemma: str
    tag: str = ""
    feats: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    root: str = ""
    prefixes: Tuple[str, ...] = field(default_factory=tuple)
    suffixes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def feats_dict(self) -> Dict[str, str]:
        """Rasgos como diccionario."""
        return dict(self.feats)


def _feats(**kwargs: Optional[str]) -> Tuple[Tuple[str, str], ...]:
    """Ordena rasgos y descarta los vacíos (formato estable para comparar)."""
    return tuple(sorted((k, v) for k, v in kwargs.items() if v))


def _strip_inflection(word: str) -> Tuple[str, bool, bool]:
    """Quita ``-n`` (acusativo) y luego ``-j`` (plural) del final.

    Returns:
        ``(base, es_plural, es_acusativo)``.

    Examples:
        >>> _strip_inflection("hundojn")
        ('hundo', True, True)
        >>> _strip_inflection("hundo")
        ('hundo', False, False)
    """
    accusative = plural = False
    base = word
    if len(base) > 2 and base.endswith("n"):
        base, accusative = base[:-1], True
    if len(base) > 2 and base.endswith("j"):
        base, plural = base[:-1], True
    return base, plural, accusative


def split_affixes(stem: str) -> Tuple[Tuple[str, ...], str, Tuple[str, ...]]:
    """Separa prefijos y sufijos de una raíz (sin terminación gramatical).

    Es una heurística conservadora (ver la justificación en
    :data:`vortilo.lexicon.SUFFIXES`): solo quita afijos poco ambiguos, nunca
    deja una raíz de menos de 3 letras y respeta
    :data:`vortilo.lexicon.ROOT_EXCEPTIONS`.

    Args:
        stem: Lema sin la vocal final, en minúsculas (``malsanulej``).

    Returns:
        ``(prefijos, raíz, sufijos)``.

    Examples:
        >>> split_affixes("malsanulej")
        (('mal',), 'san', ('ul', 'ej'))
        >>> split_affixes("lernej")
        ((), 'lern', ('ej',))
        >>> split_affixes("insul")
        ((), 'insul', ())
    """
    prefixes = []
    suffixes = []
    root = stem

    # Prefijos primero: malsan -> san (si se quitara antes el sufijo, -an
    # se comería parte de la raíz: mals+an).
    changed = True
    while changed and root not in lexicon.ROOT_EXCEPTIONS:
        changed = False
        for prefix in lexicon.PREFIXES:
            if root.startswith(prefix) and len(root) - len(prefix) >= _MIN_ROOT:
                root = root[len(prefix):]
                prefixes.append(prefix)
                changed = True
                break

    # Sufijos: de derecha a izquierda, mientras se pueda.
    changed = True
    while changed and root not in lexicon.ROOT_EXCEPTIONS:
        changed = False
        for suffix in lexicon.SUFFIXES:
            if root.endswith(suffix) and len(root) - len(suffix) >= _MIN_ROOT:
                root = root[: -len(suffix)]
                suffixes.insert(0, suffix)
                changed = True
                break

    return tuple(prefixes), root, tuple(suffixes)


def _participle(stem: str) -> Optional[Tuple[str, str]]:
    """Si ``stem`` acaba en sufijo de participio, devuelve ``(tiempo, voz)``.

    Exige que quede una raíz de al menos 3 letras, para no confundir
    ``granda`` o ``plata`` con participios.
    """
    for suffix, tense, voice in lexicon.PARTICIPLE_SUFFIXES:
        if stem.endswith(suffix) and len(stem) - len(suffix) >= _MIN_ROOT:
            return tense, voice
    return None


def _open_class(stem: str, pos: str, lemma: str, tag: str, feats) -> Analysis:
    """Construye el análisis de una palabra de clase abierta, con su raíz."""
    prefixes, root, suffixes = split_affixes(stem)
    return Analysis(pos, lemma, tag, feats, root, prefixes, suffixes)


@lru_cache(maxsize=200_000)
def analyze(word: str) -> Analysis:
    """Analiza una palabra aislada (sin contexto).

    El resultado se cachea: en un corpus las palabras se repiten muchísimo, así
    que analizar cada forma una sola vez acelera mucho el procesado.

    Args:
        word: Palabra **normalizada y en minúsculas** (``Token.norm_``).

    Returns:
        Un :class:`Analysis`.

    Examples:
        >>> a = analyze("hundojn")
        >>> a.pos, a.lemma, dict(a.feats)
        ('NOUN', 'hundo', {'Case': 'Acc', 'Number': 'Plur'})
        >>> analyze("kuris").lemma
        'kuri'
        >>> analyze("kaj").pos
        'CCONJ'
    """
    if not word:
        return Analysis("X", word)

    # 0. Abreviaturas conocidas y artículo elidido.
    if word in lexicon.ABBREVIATIONS:
        lemma, pos = lexicon.ABBREVIATIONS[word]
        return Analysis(pos, lemma, "ABBR", root=lemma)
    if word == "l'":
        return Analysis("DET", "la", "ART", _feats(Definite="Def"), "la")

    # 1. Clase cerrada, forma base.
    closed = lexicon.closed_class(word)
    if closed is not None:
        feats = ()
        if word in lexicon.CORRELATIVES:
            prefix = next(p for p in sorted(lexicon.CORRELATIVE_PREFIXES, key=len, reverse=True)
                          if word.startswith(p))
            feats = _feats(PronType=_PRONTYPE[prefix])
        elif word in lexicon.PERSONAL_PRONOUNS:
            feats = _feats(PronType="Prs", Case="Nom")
        elif word in lexicon.ARTICLE:
            feats = _feats(Definite="Def")
        elif word in lexicon.NUMERALS:
            feats = _feats(NumType="Card")
        return Analysis(closed, word, closed, feats, word)

    base, plural, accusative = _strip_inflection(word)
    number = "Plur" if plural else None
    case = "Acc" if accusative else None

    # 2a. Pronombre personal en acusativo: min, lin, ilin.
    if accusative and not plural and base in lexicon.PERSONAL_PRONOUNS:
        return Analysis("PRON", base, "PRON", _feats(PronType="Prs", Case="Acc"), base)

    # 2b. Posesivo: mia, viajn, sia…  (pronombre + -a)
    if base.endswith("a") and base[:-1] in lexicon.PERSONAL_PRONOUNS:
        return Analysis(
            "DET", base, "POSS",
            _feats(Poss="Yes", Number=number or "Sing", Case=case or "Nom"), base[:-1],
        )

    # 2c. Correlativo flexionado: kiujn, tian, kien…
    if base in lexicon.CORRELATIVES:
        suffix = next(s for s in lexicon.CORRELATIVE_SUFFIXES if base.endswith(s)
                      and base[: -len(s)] in lexicon.CORRELATIVE_PREFIXES)
        upos, allows_plural, allows_acc = lexicon.CORRELATIVE_SUFFIXES[suffix]
        if (not plural or allows_plural) and (not accusative or allows_acc):
            prefix = base[: -len(suffix)]
            return Analysis(upos, base, upos, _feats(PronType=_PRONTYPE[prefix],
                                                     Number=number, Case=case), base)

    # 3. Numerales compuestos: dudek, tricent, dekunu.
    if _COMPOUND_NUMERAL.match(word):
        return Analysis("NUM", word, "NUM", _feats(NumType="Card"), word)

    # 4. Elisión poética del sustantivo: dom' = domo.
    if word.endswith("'") and len(word) > 2:
        stem = word[:-1]
        return _open_class(stem, "NOUN", stem + "o", "-'", _feats(Number="Sing"))

    # 5. Clase abierta por terminación (sobre la forma sin -j/-n).
    stem = base[:-1]
    if len(stem) >= _MIN_STEM:
        ending = base[-1]
        suffix_tag = "-" + base[-1] + ("j" if plural else "") + ("n" if accusative else "")

        if ending == "o":
            feats = _feats(Number=number or "Sing", Case=case or "Nom")
            return _open_class(stem, "NOUN", base, suffix_tag, feats)

        if ending == "a":
            feats = dict(Number=number or "Sing", Case=case or "Nom")
            if stem in lexicon.NUMERALS or _COMPOUND_NUMERAL.match(stem):
                feats["NumType"] = "Ord"           # dua, dudeka
            part = _participle(stem)
            if part:
                feats.update(VerbForm="Part", Tense=part[0], Voice=part[1])
            return _open_class(stem, "ADJ", base, suffix_tag, _feats(**feats))

        if ending == "e" and not plural:
            # -en indica dirección (hejmen = hacia casa); el lema es -e.
            feats = _feats(Case=case)
            part = _participle(stem)
            if part:
                feats = _feats(Case=case, VerbForm="Part", Tense=part[0], Voice=part[1])
            return _open_class(stem, "ADV", base, suffix_tag, feats)

    # 6. Verbos: solo si no se quitó -j/-n (los verbos no se flexionan así).
    if not plural and not accusative:
        for ending, feats in _VERB_ENDINGS:
            if word.endswith(ending) and len(word) - len(ending) >= _MIN_STEM:
                stem = word[: -len(ending)]
                lemma = stem + "i"
                pos = "AUX" if lemma in _AUXILIARIES else "VERB"
                return _open_class(stem, pos, lemma, "-" + ending, _feats(**feats))

    # 7. Nada encaja: palabra extranjera, sigla, nombre propio sin esperantizar…
    return Analysis("X", word, "X", (), word)
