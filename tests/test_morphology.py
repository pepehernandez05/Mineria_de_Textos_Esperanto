"""Tests del análisis morfológico (POS, lema, rasgos, raíz)."""

import pytest

from vortilo import analyze
from vortilo.morphology import split_affixes


@pytest.mark.parametrize(
    "word, pos, lemma",
    [
        # Sustantivos con plural y acusativo
        ("hundo", "NOUN", "hundo"),
        ("hundoj", "NOUN", "hundo"),
        ("hundon", "NOUN", "hundo"),
        ("hundojn", "NOUN", "hundo"),
        # Adjetivos
        ("bela", "ADJ", "bela"),
        ("belajn", "ADJ", "bela"),
        # Adverbios derivados y de dirección
        ("rapide", "ADV", "rapide"),
        ("hejmen", "ADV", "hejme"),
        # Verbos: todos los tiempos y modos
        ("kuras", "VERB", "kuri"),
        ("kuris", "VERB", "kuri"),
        ("kuros", "VERB", "kuri"),
        ("kurus", "VERB", "kuri"),
        ("kuru", "VERB", "kuri"),
        ("kuri", "VERB", "kuri"),
        ("estas", "AUX", "esti"),
        # Clase cerrada
        ("la", "DET", "la"),
        ("kaj", "CCONJ", "kaj"),
        ("ke", "SCONJ", "ke"),
        ("en", "ADP", "en"),
        ("ne", "PART", "ne"),
        ("tre", "ADV", "tre"),
        ("ankaŭ", "ADV", "ankaŭ"),
        # Pronombres y posesivos
        ("mi", "PRON", "mi"),
        ("min", "PRON", "mi"),
        ("ilin", "PRON", "ili"),
        ("mia", "DET", "mia"),
        ("miajn", "DET", "mia"),
        # Correlativos
        ("kiu", "DET", "kiu"),
        ("kiujn", "DET", "kiu"),
        ("tio", "PRON", "tio"),
        ("kion", "PRON", "kio"),
        ("kie", "ADV", "kie"),
        ("kien", "ADV", "kie"),
        ("kiam", "ADV", "kiam"),
        ("neniu", "DET", "neniu"),
        ("ĉiuj", "DET", "ĉiu"),
        # Numerales
        ("tri", "NUM", "tri"),
        ("dudek", "NUM", "dudek"),
        ("kvincent", "NUM", "kvincent"),
        ("dua", "ADJ", "dua"),
        # Elisión
        ("l'", "DET", "la"),
        ("dom'", "NOUN", "domo"),
        # Abreviaturas
        ("ekz.", "ADV", "ekzemple"),
    ],
)
def test_pos_and_lemma(word, pos, lemma):
    a = analyze(word)
    assert (a.pos, a.lemma) == (pos, lemma)


def test_closed_class_beats_endings():
    # "kaj" acaba en -j pero no es un plural; "tre" acaba en -e pero no es derivado.
    assert analyze("kaj").pos == "CCONJ"
    assert analyze("plej").pos == "ADV"
    assert analyze("tre").pos == "ADV"


def test_unknown_word_is_x():
    assert analyze("madrid").pos == "X"
    assert analyze("xyz").pos == "X"


def test_short_stem_is_not_open_class():
    # Raíz de una letra: no se acepta como palabra derivada.
    assert analyze("bo").pos == "X"


class TestFeatures:
    def test_noun_feats(self):
        assert analyze("hundojn").feats_dict == {"Case": "Acc", "Number": "Plur"}
        assert analyze("hundo").feats_dict == {"Case": "Nom", "Number": "Sing"}

    @pytest.mark.parametrize(
        "word, tense, mood",
        [("kuras", "Pres", "Ind"), ("kuris", "Past", "Ind"), ("kuros", "Fut", "Ind")],
    )
    def test_verb_tense(self, word, tense, mood):
        f = analyze(word).feats_dict
        assert f["Tense"] == tense and f["Mood"] == mood

    def test_conditional_and_infinitive(self):
        assert analyze("kurus").feats_dict["Mood"] == "Cnd"
        assert analyze("kuri").feats_dict["VerbForm"] == "Inf"

    def test_participle(self):
        f = analyze("legata").feats_dict
        assert f["VerbForm"] == "Part" and f["Tense"] == "Pres" and f["Voice"] == "Pass"
        f = analyze("skribintaj").feats_dict
        assert f["Tense"] == "Past" and f["Voice"] == "Act" and f["Number"] == "Plur"

    def test_not_a_participle(self):
        # "granda" acaba en -nda, no en sufijo de participio con raíz suficiente.
        assert "VerbForm" not in analyze("granda").feats_dict

    def test_ordinal(self):
        assert analyze("dua").feats_dict["NumType"] == "Ord"

    def test_correlative_prontype(self):
        assert analyze("kiu").feats_dict["PronType"] == "Int,Rel"
        assert analyze("tiu").feats_dict["PronType"] == "Dem"
        assert analyze("neniu").feats_dict["PronType"] == "Neg"
        assert analyze("ĉiu").feats_dict["PronType"] == "Tot"
        assert analyze("iu").feats_dict["PronType"] == "Ind"

    def test_tag_is_ending(self):
        assert analyze("hundojn").tag == "-ojn"
        assert analyze("kuris").tag == "-is"


class TestAffixes:
    @pytest.mark.parametrize(
        "stem, expected",
        [
            ("malsanulej", (("mal",), "san", ("ul", "ej"))),
            ("lernej", ((), "lern", ("ej",))),
            ("instruist", ((), "instru", ("ist",))),
            ("malgrand", (("mal",), "grand", ())),
            ("naskiĝ", ((), "nask", ("iĝ",))),
        ],
    )
    def test_split(self, stem, expected):
        assert split_affixes(stem) == expected

    @pytest.mark.parametrize("stem", ["insul", "ekzist", "vulkan", "simil", "facil", "muskul"])
    def test_exceptions_not_split(self, stem):
        assert split_affixes(stem) == ((), stem, ())

    def test_min_root_length(self):
        # "ejo" (lugar) no puede quedarse sin raíz.
        assert split_affixes("ej")[1] == "ej"

    def test_root_in_analysis(self):
        assert analyze("lernejoj").root == "lern"
        assert analyze("lernantoj").root == "lernant"  # -ant no se quita (participio)
