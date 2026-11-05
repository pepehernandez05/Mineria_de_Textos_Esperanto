#!/usr/bin/env python3
"""Construye el dataset etiquetado a partir de la Wikipedia en Esperanto.

Uso::

    pip install -e ".[dataset]"
    python scripts/build_dataset.py                       # usa scripts/categories.json
    python scripts/build_dataset.py --config otra.json --out data/dataset.csv
    python scripts/build_dataset.py --explore "Kategorio:Mamuloj"   # solo contar
    python scripts/build_dataset.py --no-cache            # ignora la caché en disco

Proceso
-------
1. Para cada clase, recorre las categorías de Wikipedia indicadas en el JSON
   (y sus subcategorías hasta ``depth``) y recoge los títulos de artículos.
2. **Descarta los artículos que aparecen en más de una clase**: su etiqueta es
   ambigua y meterían ruido seguro.
3. Descarga el texto plano de cada artículo (API ``extracts``) y corta en la
   primera sección de cola (``Referencoj``, ``Eksteraj ligiloj``…).
4. Divide en párrafos, los limpia con :func:`vortilo.clean_text` (sin HTML,
   enlaces, referencias ni ``;``) y descarta los de menos de ``min_words``.
5. Elimina párrafos duplicados (en Wikipedia hay plantillas que se repiten).
6. Limita cada clase a ``max_per_class`` párrafos (muestreo con semilla fija).
7. Escribe:

   * ``dataset.csv``: el contrato de datos (``text;class``, UTF-8, ``;``).
   * ``dataset_meta.csv``: mismo orden + artículo de origen y revisión. Sirve
     para hacer la partición train/test **por artículo** en el cuaderno y para
     reproducir exactamente qué versión de cada artículo se usó.
   * ``dataset_report.json``: estadísticas y medida del ruido.

Peticiones a la API
-------------------
El cliente sigue las recomendaciones de Wikimedia: User-Agent con contacto,
peticiones en serie con una pausa mínima, parámetro ``maxlag`` y respeto de la
cabecera ``Retry-After`` ante un 429/503. Además, cada respuesta correcta se
guarda en una caché en disco, de modo que si el script se interrumpe se puede
relanzar y continúa donde lo dejó sin repetir peticiones.

Ruido de etiquetado
-------------------
Cada párrafo hereda la etiqueta de su artículo, pero hay párrafos genéricos
(``Ĝi estis fondita en 1950``) que no dicen nada del tema. El informe estima
este ruido como el porcentaje de párrafos con pocas palabras de contenido
(lemas que no son *stop-words*). En el cuaderno se mide de forma más fina con
las probabilidades de validación cruzada.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

import vortilo

API = "https://eo.wikipedia.org/w/api.php"
# Wikimedia pide un User-Agent descriptivo con una forma de contacto (URL o correo).
USER_AGENT = (
    "vortilo-dataset/0.1 (https://github.com/pepehernandez05/vortilo; "
    "CUNEF Mineria de Textos; practica academica)"
)
#: Retraso de replicación máximo (s) que toleramos; si el servidor va peor,
#: la API responde con error ``maxlag`` y esperamos.
MAXLAG = 5

#: Secciones a partir de las cuales el artículo ya no tiene texto útil.
TAIL_SECTIONS = re.compile(
    r"^=+\s*(Referencoj|Notoj|Eksteraj ligiloj|Vidu ankaŭ|Bibliografio|Literaturo|"
    r"Fontoj|Piednotoj|Rilataj artikoloj|Galerio|Ligiloj)\s*=+\s*$",
    re.IGNORECASE | re.MULTILINE,
)
HEADING = re.compile(r"^=+.*=+\s*$")


# --------------------------------------------------------------------------- #
# Cliente de la API (con límite de ritmo, reintentos y caché)
# --------------------------------------------------------------------------- #


class WikiClient:
    def __init__(
        self,
        delay: float = 0.5,
        retries: int = 6,
        timeout: float = 30.0,
        max_wait: float = 120.0,
        cache_dir: Optional[Path] = None,
    ):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.delay = delay            # pausa mínima entre peticiones (s)
        self.retries = retries
        self.timeout = timeout
        self.max_wait = max_wait      # tope de espera ante 429/maxlag (s)
        self.cache_dir = cache_dir
        self._last_request = 0.0
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- utilidades internas ------------------------------------------------

    def _throttle(self) -> None:
        """Garantiza una pausa mínima entre dos peticiones consecutivas."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def _retry_after(self, resp: requests.Response, default: float) -> float:
        """Segundos a esperar según ``Retry-After`` (o ``default`` si no viene)."""
        try:
            return min(float(resp.headers["Retry-After"]), self.max_wait)
        except (KeyError, ValueError):
            return min(default, self.max_wait)

    def _cache_file(self, params: dict) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    # -- petición con reintentos --------------------------------------------

    def get(self, **params) -> dict:
        params = {"format": "json", "formatversion": "2", **params}

        cache_file = self._cache_file(params)
        if cache_file is not None and cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except ValueError:
                pass  # caché corrupta: se vuelve a pedir

        request_params = {**params, "maxlag": MAXLAG}
        for attempt in range(self.retries):
            self._throttle()
            try:
                resp = self.session.get(API, params=request_params, timeout=self.timeout)
                self._last_request = time.monotonic()

                # Límite de peticiones o servidor saturado: respetar Retry-After
                if resp.status_code in (429, 503):
                    wait = self._retry_after(resp, default=5 * 2 ** attempt)
                    print(f"  ! HTTP {resp.status_code}; espero {wait:.0f}s "
                          f"(intento {attempt + 1}/{self.retries})", file=sys.stderr)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Retraso de replicación en el servidor (maxlag)
                if data.get("error", {}).get("code") == "maxlag":
                    wait = self._retry_after(resp, default=5.0)
                    print(f"  ! maxlag; espero {wait:.0f}s "
                          f"(intento {attempt + 1}/{self.retries})", file=sys.stderr)
                    time.sleep(wait)
                    continue

                if cache_file is not None and "error" not in data:
                    cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                return data
            except (requests.RequestException, ValueError) as exc:
                self._last_request = time.monotonic()
                wait = min(2 ** attempt, self.max_wait)
                print(f"  ! error ({exc}); reintento en {wait}s", file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError(f"La API falló {self.retries} veces con {params}")

    def category_members(self, category: str, depth: int) -> Set[str]:
        """Títulos de artículos (ns=0) de una categoría y sus subcategorías."""
        if not category.startswith("Kategorio:"):
            category = "Kategorio:" + category
        seen_cats: Set[str] = set()
        articles: Set[str] = set()
        frontier = [(category, 0)]
        while frontier:
            cat, level = frontier.pop()
            if cat in seen_cats:
                continue
            seen_cats.add(cat)
            cont: dict = {}
            while True:
                data = self.get(action="query", list="categorymembers", cmtitle=cat,
                                cmtype="page|subcat", cmlimit="500", **cont)
                for m in data.get("query", {}).get("categorymembers", []):
                    if m["ns"] == 0:
                        articles.add(m["title"])
                    elif m["ns"] == 14 and level < depth:
                        frontier.append((m["title"], level + 1))
                if "continue" not in data:
                    break
                cont = data["continue"]
        return articles

    def extract(self, title: str) -> Tuple[str, Optional[int]]:
        """Texto plano completo de un artículo y su id de revisión."""
        data = self.get(action="query", prop="extracts|revisions", titles=title,
                        explaintext="1", exsectionformat="wiki", rvprop="ids", redirects="1")
        pages = data.get("query", {}).get("pages", [])
        if not pages or "missing" in pages[0]:
            return "", None
        page = pages[0]
        revid = page.get("revisions", [{}])[0].get("revid")
        return page.get("extract", "") or "", revid


# --------------------------------------------------------------------------- #
# Texto -> párrafos
# --------------------------------------------------------------------------- #


def article_paragraphs(text: str, min_words: int) -> List[str]:
    """Párrafos limpios de un artículo, sin la cola de referencias."""
    tail = TAIL_SECTIONS.search(text)
    if tail:
        text = text[: tail.start()]
    paragraphs = []
    for line in text.split("\n"):
        if HEADING.match(line.strip()):
            continue
        clean = vortilo.clean_text(line)
        clean = " ".join(clean.split())  # sin saltos de línea dentro del párrafo
        if len(clean.split()) >= min_words:
            paragraphs.append(clean)
    return paragraphs


def content_words(analyzer: vortilo.Analyzer, text: str) -> int:
    return len(analyzer(text))


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #


def explore(client: WikiClient, categories: Iterable[str], depth: int) -> None:
    for cat in categories:
        titles = client.category_members(cat, depth)
        print(f"{cat} (profundidad {depth}): {len(titles)} artículos")


def build(config: dict, out: Path, client: WikiClient) -> None:
    seed = config.get("seed", 42)
    min_words = config.get("min_words", 12)
    max_per_class = config.get("max_per_class", 1500)
    max_articles = config.get("max_articles_per_class", 400)
    classes: Dict[str, dict] = config["classes"]
    rng = random.Random(seed)

    # 1. Títulos por clase
    titles_by_class: Dict[str, Set[str]] = {}
    for label, spec in classes.items():
        titles: Set[str] = set(spec.get("articles", []))
        for cat in spec.get("categories", []):
            found = client.category_members(cat, spec.get("depth", 1))
            print(f"[{label}] {cat}: {len(found)} artículos")
            titles |= found
        titles_by_class[label] = titles

    # 2. Artículos ambiguos (en más de una clase)
    owner = Counter(t for ts in titles_by_class.values() for t in ts)
    ambiguous = {t for t, n in owner.items() if n > 1}
    for label in titles_by_class:
        titles_by_class[label] -= ambiguous
    print(f"Artículos en más de una clase (descartados): {len(ambiguous)}")

    # 3-5. Descargar y trocear
    rows: List[dict] = []
    seen_text: Set[str] = set()
    dropped = Counter()
    for label, titles in titles_by_class.items():
        ordered = sorted(titles)
        rng.shuffle(ordered)
        ordered = ordered[:max_articles]
        print(f"[{label}] descargando {len(ordered)} artículos…")
        for k, title in enumerate(ordered, 1):
            try:
                text, revid = client.extract(title)
            except RuntimeError as exc:
                print(f"  ! {title}: {exc}", file=sys.stderr)
                dropped["error_descarga"] += 1
                continue
            for p_idx, para in enumerate(article_paragraphs(text, min_words)):
                if para in seen_text:
                    dropped["duplicado"] += 1
                    continue
                seen_text.add(para)
                rows.append({"text": para, "class": label, "article": title,
                             "revid": revid, "paragraph": p_idx})
            if k % 50 == 0:
                print(f"  {k}/{len(ordered)}")

    # 6. Límite por clase
    by_class: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)
    final: List[dict] = []
    for label in classes:
        items = by_class.get(label, [])
        if len(items) > max_per_class:
            dropped["recorte_por_clase"] += len(items) - max_per_class
            items = rng.sample(items, max_per_class)
        final.extend(items)
    rng.shuffle(final)

    # Comprobación del contrato antes de escribir
    for r in final:
        assert ";" not in r["text"] and "\n" not in r["text"], r["text"][:80]

    # 7. Escribir
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["text", "class"])
        for r in final:
            w.writerow([r["text"], r["class"]])
    meta_path = out.with_name(out.stem + "_meta.csv")
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["row", "class", "article", "revid", "paragraph"])
        for i, r in enumerate(final):
            w.writerow([i, r["class"], r["article"].replace(";", ","), r["revid"], r["paragraph"]])

    # Informe de ruido
    analyzer = vortilo.make_analyzer()
    low_content_threshold = config.get("low_content_words", 5)
    report = {"seed": seed, "min_words": min_words, "max_per_class": max_per_class,
              "ambiguous_articles": len(ambiguous), "dropped": dict(dropped), "classes": {}}
    for label in classes:
        items = [r for r in final if r["class"] == label]
        if not items:
            continue
        cw = [content_words(analyzer, r["text"]) for r in items]
        report["classes"][label] = {
            "paragraphs": len(items),
            "articles": len({r["article"] for r in items}),
            "mean_words": round(sum(len(r["text"].split()) for r in items) / len(items), 1),
            "low_content_pct": round(100 * sum(c < low_content_threshold for c in cw) / len(items), 1),
        }
    report_path = out.with_name(out.stem + "_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nEscrito {out} ({len(final)} párrafos)")
    print(json.dumps(report["classes"], ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=here / "categories.json")
    parser.add_argument("--out", type=Path, default=here.parent / "data" / "dataset.csv")
    parser.add_argument("--explore", nargs="*", help="Solo contar artículos de estas categorías")
    parser.add_argument("--depth", type=int, default=1, help="Profundidad para --explore")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Pausa mínima entre peticiones (s)")
    parser.add_argument("--cache-dir", type=Path, default=here.parent / "data" / ".cache",
                        help="Carpeta de caché de respuestas de la API")
    parser.add_argument("--no-cache", action="store_true",
                        help="No usar caché en disco")
    args = parser.parse_args(argv)

    cache_dir = None if args.no_cache else args.cache_dir
    client = WikiClient(delay=args.delay, cache_dir=cache_dir)
    if args.explore:
        explore(client, args.explore, args.depth)
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    build(config, args.out, client)


if __name__ == "__main__":
    main()