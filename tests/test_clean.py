"""Tests de limpieza de texto y división en párrafos."""

from vortilo import clean_text, split_paragraphs
from vortilo.clean import remove_references, strip_html


def test_strip_html_inline_and_block():
    assert strip_html("<p>La <b>hundo</b></p><p>La kato</p>") == "La hundo\n\nLa kato"


def test_strip_script_and_style():
    assert strip_html("<script>var x=1;</script><style>p{}</style>Teksto") == "Teksto"


def test_entities():
    assert strip_html("&laquo;Saluton&raquo; &amp; adiaŭ") == "«Saluton» & adiaŭ"


def test_references():
    assert remove_references("Fondita en 1950[1][noto 2].") == "Fondita en 1950."
    assert remove_references("Teksto[bezonata citaĵo].") == "Teksto."
    assert remove_references("Teksto {{ŝablono}} plu.") == "Teksto  plu."


def test_references_keep_real_brackets():
    # Corchetes con contenido largo que no es referencia se mantienen.
    assert "[la tuta frazo restas]" in remove_references("Vidu [la tuta frazo restas].")


def test_clean_text_contract():
    raw = "<p>Unu; du [3] https://x.org kaj ana@x.org ( )</p>"
    out = clean_text(raw)
    assert ";" not in out
    assert "http" not in out and "@" not in out and "[3]" not in out
    assert out == "Unu, du kaj"


def test_clean_text_keeps_newlines():
    assert clean_text("Unu\n\nDu") == "Unu\n\nDu"


def test_split_paragraphs_min_words():
    text = "Titolo\n\nUnua alineo kun kvin vortoj.\nDua alineo ankaŭ longa sufiĉe."
    assert split_paragraphs(text, min_words=3) == [
        "Unua alineo kun kvin vortoj.",
        "Dua alineo ankaŭ longa sufiĉe.",
    ]
