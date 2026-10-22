"""Tests del normalizador, tokenizador y segmentador de frases."""

import pytest

from vortilo import Tokenizer, normalize_text, sentence_tokenizer, word_tokenizer, x_to_unicode
from vortilo.normalize import h_to_unicode, normalize_word


class TestNormalize:
    def test_x_system_lower_and_upper(self):
        assert x_to_unicode("cxgxhxjxsxux") == "ĉĝĥĵŝŭ"
        assert x_to_unicode("CXambro Sxi") == "Ĉambro Ŝi"

    def test_x_system_leaves_normal_text(self):
        assert x_to_unicode("La hundo") == "La hundo"

    def test_h_system(self):
        assert h_to_unicode("chambro") == "ĉambro"

    def test_h_system_off_by_default(self):
        # flughaveno = flug-haveno: el sistema H lo estropearía.
        assert normalize_text("flughaveno") == "flughaveno"

    def test_typographic_quotes(self):
        assert normalize_text("“Saluton”") == '"Saluton"'
        assert normalize_text("l’urbo") == "l'urbo"

    def test_nfc(self):
        decomposed = "ĉ"  # c + acento circunflejo combinable
        assert normalize_text(decomposed) == "ĉ"

    def test_normalize_word(self):
        assert normalize_word("CXAMBRO") == "ĉambro"


class TestTokenizer:
    @pytest.fixture
    def tok(self):
        return Tokenizer()

    def test_simple(self, tok):
        assert [t.text for t in tok("La hundo kuras.")] == ["La", "hundo", "kuras", "."]

    def test_punctuation_split(self, tok):
        assert [t.text for t in tok("(Jes!)")] == ["(", "Jes", "!", ")"]

    def test_numbers(self, tok):
        texts = [t.text for t in tok("En 1887 kaj 3,14 kaj 1.000.000 kaj 50%")]
        assert "1887" in texts and "3,14" in texts and "1.000.000" in texts and "50%" in texts

    def test_ordinal(self, tok):
        doc = tok("La 20-a jarcento")
        assert doc[1].text == "20-a"
        assert doc[1].kind == "ORDINAL"

    def test_abbreviation_kept(self, tok):
        texts = [t.text for t in tok("Ekz. hundoj, katoj k.t.p. vivas.")]
        assert "Ekz." in texts and "k.t.p." in texts

    def test_single_letter_abbrev_case_sensitive(self, tok):
        # "M." es una inicial, no la abreviatura de "mortis".
        doc = tok("M. Boulton (m. 1809)")
        assert doc[0].kind == "WORD"
        assert any(t.text == "m." and t.kind == "ABBR" for t in doc)

    def test_url_and_email(self, tok):
        doc = tok("Vidu https://eo.wikipedia.org/wiki/Hundo aŭ skribu al ana@ekzemplo.org.")
        kinds = {t.text: t.kind for t in doc}
        assert kinds["https://eo.wikipedia.org/wiki/Hundo"] == "URL"
        assert kinds["ana@ekzemplo.org"] == "EMAIL"
        assert doc[-1].text == "."

    def test_elision(self, tok):
        assert [t.text for t in tok("l' urbo kaj dom'")] == ["l'", "urbo", "kaj", "dom'"]

    def test_hyphen_split(self, tok):
        assert [t.text for t in tok("ĉi-jare")] == ["ĉi", "-", "jare"]

    def test_x_system_applied(self, tok):
        doc = tok("Cxu?")
        assert doc[0].text == "Ĉu"
        assert doc.text == "Ĉu?"

    def test_ellipsis(self, tok):
        assert [t.text for t in tok("Nu...")] == ["Nu", "..."]

    def test_offsets_rebuild_text(self, tok):
        text = "La  hundo,\nkaj   la kato."
        doc = tok(text)
        assert "".join(t.text_with_ws for t in doc) == text
        for t in doc:
            assert doc.text[t.idx : t.idx + len(t.text)] == t.text

    def test_empty(self, tok):
        assert len(tok("")) == 0
        assert len(tok("   ")) == 0

    def test_word_tokenizer_shortcut(self):
        assert word_tokenizer("Saluton, mondo!") == ["Saluton", ",", "mondo", "!"]


class TestSentences:
    def test_basic(self):
        assert sentence_tokenizer("Mi lernas. Ĝi estas facila! Ĉu vere?") == [
            "Mi lernas.", "Ĝi estas facila!", "Ĉu vere?",
        ]

    def test_abbreviation_does_not_split(self):
        assert sentence_tokenizer("Bestoj, ekz. hundoj, vivas. Jes.") == [
            "Bestoj, ekz. hundoj, vivas.", "Jes.",
        ]

    def test_decimal_does_not_split(self):
        assert len(sentence_tokenizer("Pi estas 3.14 proksimume.")) == 1

    def test_initials_do_not_split(self):
        assert len(sentence_tokenizer("L. L. Zamenhof kreis Esperanton.")) == 1

    def test_lowercase_after_dot_does_not_split(self):
        assert len(sentence_tokenizer("Vidu la ĉap. tri de la libro.")) == 1

    def test_closing_quote_stays(self):
        assert sentence_tokenizer('Li diris "Jes." Poste li foriris.') == [
            'Li diris "Jes."', "Poste li foriris.",
        ]

    def test_paragraph_break_splits(self):
        assert len(sentence_tokenizer("Titolo\n\nla teksto komenciĝas")) == 2
