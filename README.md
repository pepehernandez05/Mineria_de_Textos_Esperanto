# vortilo — NLP para Esperanto y clasificación temática

Práctica de **Minería de Textos** (CUNEF, 4º INF, 2026/2027).

* **Librería** `vortilo`: NLP basada en reglas para Esperanto con API tipo spaCy
  (tokenizador, lematizador, POS-tagging, stop-words, `Doc`/`Token`/`Span`).
* **Dataset**: párrafos de la Wikipedia en Esperanto etiquetados por temática.
* **Cuaderno**: modelo de clasificación y clasificación de páginas web reales.

## Instalación

```bash
pip install git+https://github.com/USUARIO/vortilo.git
```

```python
import vortilo
nlp = vortilo.load()
doc = nlp("La hundoj kuris en la parko.")
[(t.text, t.lemma_, t.pos_) for t in doc]
```

Documentación completa en `docs/` (MkDocs / Read the Docs).

## Estructura

```
src/vortilo/           librería
  normalize.py         sistema X, Unicode
  tokenizer.py         tokens y frases
  lexicon.py           clase cerrada, correlativos, afijos
  morphology.py        análisis por reglas (POS, lema, rasgos, raíz)
  stopwords.py         stop-words
  tokens.py            Doc, Token, Span, MorphAnalysis
  pipeline.py          Language (nlp) y componentes
  clean.py             limpieza de HTML/Wikipedia
  features.py          analyzer para scikit-learn
tests/                 pytest (+ doctests de los módulos)
scripts/
  build_dataset.py     genera data/dataset.csv desde Wikipedia
  categories.json      clases y categorías de Wikipedia
  check_dataset.py     valida el contrato de datos
data/                  dataset.csv (+ _meta.csv, _report.json)
notebooks/             cuaderno de entrenamiento y clasificación
docs/                  documentación (mkdocs)
```

## Dataset

```bash
pip install -e ".[dataset]"
# 1) comprobar que las categorías tienen artículos suficientes
python scripts/build_dataset.py --explore "Kategorio:Mamuloj" "Kategorio:Birdoj" --depth 2
# 2) generar el dataset
python scripts/build_dataset.py
# 3) validar el contrato
python scripts/check_dataset.py data/dataset.csv
```

Contrato de datos: CSV UTF-8, separador `;`, cabecera `text;class`, un párrafo
por fila, sin `;`, HTML, enlaces ni referencias.

## Tests

```bash
pip install -e ".[test]" scikit-learn
pytest
```

## Licencia

MIT
