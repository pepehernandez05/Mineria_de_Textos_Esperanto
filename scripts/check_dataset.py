#!/usr/bin/env python3
"""Valida que un CSV cumple el contrato de datos de la práctica.

Contrato: CSV UTF-8, separador ``;``, cabecera ``text;class``, dos columnas,
un párrafo por fila, sin ``;`` en el texto, sin HTML ni enlaces.

Uso::

    python scripts/check_dataset.py data/dataset.csv

Devuelve código de salida 1 si hay errores (lo usa GitHub Actions).
Si el fichero aún no existe, avisa y termina sin error.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

HTML = re.compile(r"<[a-zA-Z/][^>]*>")
URL = re.compile(r"https?://|www\.")
REF = re.compile(r"\[\d+\]")
RECOMMENDED_PER_CLASS = 300


def check(path: Path) -> int:
    if not path.exists():
        print(f"AVISO: {path} no existe todavía; nada que validar.")
        return 0

    errors: list = []
    try:
        raw = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        print(f"ERROR: el fichero no es UTF-8 ({exc})")
        return 1

    lines = raw.splitlines()
    if not lines or lines[0].lstrip("﻿") != "text;class":
        errors.append(f"Cabecera incorrecta: {lines[0] if lines else '(vacío)'!r}; debe ser 'text;class'")

    reader = csv.reader(raw.splitlines()[1:], delimiter=";")
    counts: Counter = Counter()
    seen = set()
    dups = 0
    for n, row in enumerate(reader, start=2):
        if len(row) != 2:
            errors.append(f"Línea {n}: {len(row)} columnas (¿hay un ';' en el texto?)")
            continue
        text, label = row
        if not text.strip() or not label.strip():
            errors.append(f"Línea {n}: texto o clase vacíos")
        if HTML.search(text):
            errors.append(f"Línea {n}: contiene HTML")
        if URL.search(text):
            errors.append(f"Línea {n}: contiene un enlace")
        if REF.search(text):
            errors.append(f"Línea {n}: contiene una referencia [n]")
        if text in seen:
            dups += 1
        seen.add(text)
        counts[label] += 1

    print(f"{sum(counts.values())} filas, {len(counts)} clases")
    for label, c in counts.most_common():
        flag = "" if c >= RECOMMENDED_PER_CLASS else f"  (aviso: < {RECOMMENDED_PER_CLASS} recomendados)"
        print(f"  {label}: {c}{flag}")
    if dups:
        print(f"Aviso: {dups} párrafos duplicados")
    if len(counts) < 3:
        errors.append("Se necesitan al menos 3 clases")

    for e in errors[:30]:
        print("ERROR:", e)
    if len(errors) > 30:
        print(f"… y {len(errors) - 30} errores más")
    return 1 if errors else 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/dataset.csv")
    sys.exit(check(target))
