# vortilo

**vortilo** (*vort-* "palabra" + *-il-* "herramienta") es una librería de
Procesamiento de Lenguaje Natural para **Esperanto**, basada en reglas y con
una API inspirada en [spaCy](https://spacy.io).

```python
import vortilo

nlp = vortilo.load()
doc = nlp("La hundoj kuris en la parko.")

for t in doc:
    print(t.text, t.lemma_, t.pos_, t.morph, t.is_stop)
```

```
La       la      DET    Definite=Def          True
hundoj   hundo   NOUN   Case=Nom|Number=Plur  False
kuris    kuri    VERB   Mood=Ind|Tense=Past|VerbForm=Fin  False
en       en      ADP                          True
la       la      DET    Definite=Def          True
parko    parko   NOUN   Case=Nom|Number=Sing  False
.        .       PUNCT                        False
```

## Instalación

```bash
pip install git+https://github.com/USUARIO/vortilo.git
```

No tiene dependencias: es Python puro (≥ 3.9).

## Qué incluye

| Componente | Qué hace |
| --- | --- |
| Normalizador | Sistema X (`cx` → `ĉ`), Unicode NFC, comillas tipográficas |
| Tokenizador | Palabras, números, ordinales (`20-a`), URLs, abreviaturas (`ekz.`), elisión (`l'`) |
| Segmentador | Frases, sin cortar en abreviaturas, decimales ni iniciales |
| Etiquetador | Categoría UPOS, etiqueta fina y rasgos morfológicos UD |
| Lematizador | Lema y raíz (sin afijos) |
| Stop-words | Lista por defecto ampliable en caliente |
| `Doc` / `Token` / `Span` | Estructuras de resultado, como en spaCy |
| `make_analyzer` | Conecta la librería con `TfidfVectorizer` de scikit-learn |
