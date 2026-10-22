# Diseño lingüístico

Esta página explica **por qué** la librería funciona como funciona. Es la base
para defender las decisiones de diseño.

## Por qué basada en reglas

La gramática del Esperanto es regular y sin excepciones: la categoría de una
palabra de clase abierta se lee en su terminación. Eso permite un etiquetador
y un lematizador deterministas, sin datos de entrenamiento, rápidos y
explicables (cada decisión se puede rastrear hasta una regla).

## Orden de las reglas del análisis morfológico

1. **Clase cerrada primero.** Artículo, pronombres, preposiciones, conjunciones,
   partículas, adverbios primitivos y correlativos son listas finitas que *no*
   siguen las terminaciones: `kaj` no es un plural en `-j`, `tre` no es un
   adverbio derivado en `-e`, `kiu` no es un imperativo en `-u`. Si se aplicaran
   antes las terminaciones, estas palabras —las más frecuentes del idioma— se
   etiquetarían mal.
2. **Formas flexionadas de clase cerrada**: `min`, `miajn`, `kiujn`, `kien`.
   Para los correlativos se comprueba qué flexiones admite cada serie (la serie
   `-u` admite `-j` y `-n`; la serie `-am` no admite ninguna).
3. **Numerales** simples y compuestos (`dudek`, `kvincent`) con una expresión
   regular.
4. **Terminaciones**: se quitan `-n` y luego `-j` y se mira la vocal final.
5. **Verbos** solo si no se ha quitado `-j`/`-n`, porque el verbo no se
   flexiona en número ni caso.
6. **`X`** si nada encaja.

Se exige una raíz de al menos 2 letras para aceptar una terminación de clase
abierta, para no analizar interjecciones o siglas cortas como palabras
derivadas.

## Nombres propios

Una palabra en mayúscula que no es de clase cerrada se marca `PROPN` si:

* está en mitad de frase (`Mi loĝas en Hispanio`), o
* está toda en mayúsculas (sigla: `UEA`), o
* no tiene terminación esperanta (`Madrid`), aunque esté al principio.

**Limitación conocida**: un nombre propio esperantizado al principio de frase
(`Esperanto estas…`) se etiqueta `NOUN`, porque la mayúscula de inicio de frase
no aporta información. Resolverlo requeriría un diccionario de nombres propios.

## Lema

* Sustantivo → `-o` singular nominativo; adjetivo → `-a`; adverbio → `-e`
  (`hejmen` → `hejme`); verbo → infinitivo `-i`.
* Pronombres y correlativos → forma base (`ilin` → `ili`, `kiujn` → `kiu`).
* Participios → se lematizan como adjetivo (`legata`), y la información verbal
  queda en los rasgos (`VerbForm=Part|Tense=Pres|Voice=Pass`).
* Nombres propios → conservan las mayúsculas (`Esperanton` → `Esperanto`).

## Raíz (`root_`) y afijos

El Esperanto es aglutinante: `mal-san-ul-ej-o` (hospital). Quitar los afijos
agrupa familias de palabras (`lernejo`, `lernanto`, `lerni` → `lern`), lo que
puede ayudar al clasificador.

El problema es que los préstamos internacionales *parecen* llevar afijos:
`geografio` no es `ge-` + `ografio`, `fiziko` no es `fi-` + `ziko`, `maŝino`
no es `maŝ` + `-in-`. Como la raíz es para agrupar palabras de un mismo tema,
un error así **mezcla temas distintos** (`planeto` → `plan`, como `plano`).

Decisión: **precisión antes que cobertura**.

* Solo se quita el prefijo `mal-`.
* Solo se quitan los sufijos menos ambiguos (`-ej-`, `-ist-`, `-ism-`, `-ig-`,
  `-iĝ-`, `-ul-`, `-il-`, `-ad-`, …). Se descartaron `-in-`, `-et-`, `-eg-`,
  `-em-`, `-er-`, `-on-`, `-id-`.
* Nunca se deja una raíz de menos de 3 letras.
* Una lista de excepciones (`insul`, `ekzist`, `vulkan`, `muskul`…) evita los
  falsos positivos más frecuentes.
* Los prefijos se quitan **antes** que los sufijos: `malsan` → `san`. Al revés,
  `-an` se comería parte de la raíz (`mals` + `an`).

## Segmentación en frases

Se corta tras `. ! ? …` solo si el siguiente token empieza por mayúscula, cifra
o signo de apertura. Las abreviaturas (`ekz.`, `k.t.p.`) y los decimales
(`3.14`) son un único token desde el tokenizador, así que no cortan. Tampoco
cortan las iniciales (`L. L. Zamenhof`).

## Stop-words

Lista por defecto = todo el léxico cerrado + posesivos + unos pocos verbos y
adverbios muy genéricos (`esti`, `havi`, `povi`, `fari`…). Se compara contra
el lema, así que una sola entrada (`esti`) cubre `estas`, `estis`, `estos`…
