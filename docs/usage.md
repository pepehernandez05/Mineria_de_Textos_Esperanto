# Guía de uso

## El objeto `nlp`

```python
import vortilo

nlp = vortilo.load()          # pipeline completo
nlp.pipe_names                # ['senter', 'tagger', 'lemmatizer', 'stopwords']

doc = nlp("Mi parolas Esperanton. Ĉu vi?")
```

`vortilo.blank()` devuelve un `nlp` que solo tokeniza.

## Tokens

```python
t = doc[2]
t.text          # 'Esperanton'
t.lemma_        # 'Esperanto'
t.pos_          # 'PROPN'
t.tag_          # '-on'   (terminación reconocida)
t.morph         # MorphAnalysis('Case=Acc|Number=Sing')
t.morph.get("Case")   # ['Acc']
t.root_         # 'esperant'
t.is_stop, t.is_punct, t.like_num, t.like_url
t.i, t.idx, t.whitespace_, t.text_with_ws
t.nbor(1), t.sent
```

## Frases y fragmentos

```python
for sent in doc.sents:
    print(sent.text)

span = doc[0:2]     # Span
span.text, span.lemma_
```

## Gestionar el pipeline

```python
doc = nlp(texto, disable=["senter"])           # desactivar en una llamada

with nlp.select_pipes(disable=["tagger"]):     # desactivar temporalmente
    ...

def contar_sustantivos(doc):
    doc.user_data["n_nouns"] = sum(t.pos_ == "NOUN" for t in doc)
    return doc

nlp.add_pipe(contar_sustantivos, name="contador", last=True)
nlp.remove_pipe("contador")
```

## Stop-words

```python
nlp.stop_words.add("jaro")          # se aplica por lema: jaro, jaroj, jaron…
nlp.stop_words.discard("ne")
sw = vortilo.build_stop_words(extra={"jaro", "nomo"})
```

## Con scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import vortilo

analyzer = vortilo.make_analyzer(
    attr="lemma_",                 # o "root_" para agrupar familias de palabras
    remove_stop=True,
    keep_pos=("NOUN", "PROPN", "ADJ", "VERB"),
    ngram_range=(1, 2),
)
vec = TfidfVectorizer(analyzer=analyzer, min_df=2, sublinear_tf=True)
X = vec.fit_transform(textos)
```

## Limpieza de texto

```python
vortilo.clean_text("<p>Fondita en 1950[1]; vidu https://…</p>")
# 'Fondita en 1950, vidu'
vortilo.split_paragraphs(texto, min_words=10)
```
