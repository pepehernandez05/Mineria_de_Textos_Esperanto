"""Tests de la API tipo spaCy: Language, Doc, Token, Span, stop-words y analyzer."""

import pickle

import pytest

import vortilo
from vortilo import Doc, Span, Token


@pytest.fixture(scope="module")
def nlp():
    return vortilo.load()


class TestLanguage:
    def test_default_pipeline(self, nlp):
        assert nlp.pipe_names == ["senter", "tagger", "lemmatizer", "stopwords"]

    def test_call_returns_doc(self, nlp):
        doc = nlp("La hundo kuras.")
        assert isinstance(doc, Doc)
        assert isinstance(doc[0], Token)

    def test_blank_only_tokenizes(self):
        doc = vortilo.blank()("La hundoj kuras.")
        assert len(doc) == 4
        assert doc[1].lemma_ == "" and doc[1].pos_ == ""

    def test_disable_at_call(self, nlp):
        doc = nlp("La hundoj kuras.", disable=["lemmatizer"])
        assert doc[1].lemma_ == ""
        assert doc[1].pos_ == "NOUN"

    def test_select_pipes_is_temporary(self, nlp):
        with nlp.select_pipes(disable=["tagger"]):
            assert "tagger" not in nlp.pipe_names
        assert "tagger" in nlp.pipe_names

    def test_add_custom_pipe(self):
        nlp = vortilo.load()

        def marker(doc):
            doc.user_data["seen"] = True
            return doc

        nlp.add_pipe(marker, name="marker", after="tagger")
        assert nlp.pipe_names == ["senter", "tagger", "marker", "lemmatizer", "stopwords"]
        assert nlp("Saluton").user_data["seen"] is True
        nlp.remove_pipe("marker")
        assert "marker" not in nlp.pipe_names

    def test_add_pipe_errors(self):
        nlp = vortilo.load()
        with pytest.raises(ValueError):
            nlp.add_pipe("tagger")  # duplicado
        with pytest.raises(ValueError):
            nlp.add_pipe("inexistente")
        with pytest.raises(KeyError):
            nlp.remove_pipe("inexistente")

    def test_load_without_components(self):
        nlp = vortilo.load(disable=["senter"])
        assert "senter" not in nlp.pipe_names

    def test_load_unknown_model(self):
        with pytest.raises(ValueError):
            vortilo.load("es_core_news_sm")

    def test_pipe_generator(self, nlp):
        docs = list(nlp.pipe(["Unu.", "Du."]))
        assert [d.text for d in docs] == ["Unu.", "Du."]


class TestDocTokenSpan:
    def test_token_attributes(self, nlp):
        doc = nlp("La hundojn.")
        t = doc[1]
        assert (t.text, t.lemma_, t.pos_, t.tag_) == ("hundojn", "hundo", "NOUN", "-ojn")
        assert str(t.morph) == "Case=Acc|Number=Plur"
        assert t.morph.get("Case") == ["Acc"]
        assert "Number=Plur" in t.morph
        assert t.i == 1 and t.idx == 3
        assert t.is_alpha and not t.is_punct and not t.is_stop

    def test_flags(self, nlp):
        doc = nlp("La 3 hundoj, vidu https://x.org!")
        assert doc[0].is_stop
        assert doc[1].like_num
        assert doc[3].is_punct
        assert any(t.like_url for t in doc)

    def test_shape(self, nlp):
        assert nlp("Esperanto")[0].shape_ == "Xxxxx"
        assert nlp("1887")[0].shape_ == "dddd"

    def test_nbor(self, nlp):
        doc = nlp("La hundo kuras.")
        assert doc[1].nbor().text == "kuras"
        assert doc[1].nbor(-1).text == "La"
        with pytest.raises(IndexError):
            doc[0].nbor(-1)

    def test_slicing_gives_span(self, nlp):
        doc = nlp("La granda hundo kuras.")
        span = doc[1:3]
        assert isinstance(span, Span)
        assert span.text == "granda hundo"
        assert len(span) == 2
        assert span.lemma_ == "granda hundo"
        assert span[0].text == "granda"

    def test_sents(self, nlp):
        doc = nlp("Mi lernas. Ĝi estas facila!")
        sents = list(doc.sents)
        assert [s.text for s in sents] == ["Mi lernas.", "Ĝi estas facila!"]
        assert doc[3].is_sent_start is True
        assert doc[3].sent.text == "Ĝi estas facila!"

    def test_sents_without_senter(self):
        doc = vortilo.load(disable=["senter"])("Unu. Du.")
        assert not doc.has_sents
        assert len(list(doc.sents)) == 1

    def test_count_by(self, nlp):
        counts = nlp("La hundo kaj la kato.").count_by("pos_")
        assert counts["DET"] == 2 and counts["NOUN"] == 2 and "PUNCT" not in counts

    def test_to_records_and_json(self, nlp):
        doc = nlp("La hundo.")
        rec = doc.to_records()
        assert rec[1]["lemma"] == "hundo"
        js = doc.to_json()
        assert js["text"] == "La hundo." and len(js["tokens"]) == 3

    def test_empty_doc(self, nlp):
        doc = nlp("")
        assert len(doc) == 0
        assert list(doc.sents) == []


class TestProperNouns:
    def test_mid_sentence_capital_is_propn(self, nlp):
        doc = nlp("Mi loĝas en Hispanio.")
        assert doc[3].pos_ == "PROPN"
        assert doc[3].lemma_ == "Hispanio"

    def test_propn_lemma_keeps_case(self, nlp):
        doc = nlp("Mi parolas Esperanton.")
        assert doc[2].pos_ == "PROPN"
        assert doc[2].lemma_ == "Esperanto"

    def test_foreign_name_at_start(self, nlp):
        assert nlp("Madrid estas granda.")[0].pos_ == "PROPN"

    def test_acronym(self, nlp):
        assert nlp("Mi estas membro de UEA.")[4].pos_ == "PROPN"

    def test_capitalised_closed_class_not_propn(self, nlp):
        doc = nlp("Mi legis La Ondo de Esperanto.")
        assert doc[2].pos_ == "DET"

    def test_sentence_initial_noun_stays_noun(self, nlp):
        assert nlp("Hundoj bojas.")[0].pos_ == "NOUN"


class TestStopWords:
    def test_default_contains_basics(self):
        for w in ("la", "kaj", "de", "en", "mi", "kiu", "esti"):
            assert w in vortilo.STOP_WORDS

    def test_inflected_forms_via_lemma(self, nlp):
        doc = nlp("Ili estis tie kun siaj amikoj.")
        stop = {t.text for t in doc if t.is_stop}
        assert {"Ili", "estis", "tie", "kun", "siaj"} <= stop
        assert "amikoj" not in stop

    def test_modify_at_runtime(self):
        nlp = vortilo.load()
        assert not nlp("jaro")[0].is_stop
        nlp.stop_words.add("jaro")
        assert nlp("jaroj")[0].is_stop  # por el lema
        nlp.stop_words.discard("la")
        assert not nlp("la")[0].is_stop

    def test_punct_never_stop(self, nlp):
        assert not nlp(".")[0].is_stop

    def test_build_stop_words(self):
        sw = vortilo.build_stop_words(extra=["Jaro"], remove=["ne"])
        assert "jaro" in sw and "ne" not in sw
        assert "ne" in vortilo.STOP_WORDS  # la lista por defecto no cambia


class TestAnalyzer:
    def test_default_lemmas_without_stopwords(self):
        an = vortilo.make_analyzer()
        assert an("La hundoj kuris en la parko en 2020.") == ["hundo", "kuri", "parko"]

    def test_root_attr(self):
        an = vortilo.make_analyzer(attr="root_")
        assert an("La lernejoj") == ["lern"]

    def test_keep_pos(self):
        an = vortilo.make_analyzer(keep_pos=["NOUN"])
        assert an("La granda hundo kuras.") == ["hundo"]

    def test_ngrams(self):
        an = vortilo.make_analyzer(ngram_range=(1, 2))
        assert an("granda hundo kuras") == ["granda", "hundo", "kuri", "granda hundo", "hundo kuri"]

    def test_add_pos(self):
        an = vortilo.make_analyzer(add_pos=True)
        assert an("hundo") == ["hundo/NOUN"]

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            vortilo.make_analyzer(attr="pos")
        with pytest.raises(ValueError):
            vortilo.make_analyzer(ngram_range=(2, 1))

    def test_picklable(self):
        an = vortilo.make_analyzer()
        an2 = pickle.loads(pickle.dumps(an))
        assert an2("La hundoj") == ["hundo"]

    def test_works_with_sklearn(self):
        sklearn = pytest.importorskip("sklearn.feature_extraction.text")
        vec = sklearn.TfidfVectorizer(analyzer=vortilo.make_analyzer())
        X = vec.fit_transform(["La hundoj kuras.", "La hundo dormas."])
        assert X.shape[0] == 2
        assert "hundo" in vec.get_feature_names_out()
