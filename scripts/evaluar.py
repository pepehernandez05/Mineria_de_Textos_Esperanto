"""Entrena el clasificador, muestra sus métricas y guarda los gráficos en reports/."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

import vortilo

OUTPUT_DIR = Path("reports")
TOP_WORDS = 10

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
METRIC_COLORS = {"Precisión": BLUE, "Recall": "#eb6834", "F1": "#1baf7a"}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "font.size": 11,
    }
)


def percent(value, decimals=1):
    """Formatea una proporción como porcentaje con coma decimal."""
    return f"{value * 100:.{decimals}f} %".replace(".", ",")


def clean_axes(ax, grid_axis):
    """Deja solo la línea base y una rejilla discreta."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom" if grid_axis == "y" else "left"].set_visible(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_confusion_matrix(y_true, y_pred, classes, accuracy):
    """Matriz de confusión: el color es el % de cada clase real, así el desequilibrio no la distorsiona."""
    counts = confusion_matrix(y_true, y_pred, labels=classes)
    shares = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.pcolormesh(shares, cmap=BLUES, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for row, col in np.ndindex(counts.shape):
        ax.text(
            col + 0.5,
            row + 0.5,
            f"{counts[row, col]}\n{percent(shares[row, col])}",
            ha="center",
            va="center",
            color="#ffffff" if shares[row, col] >= 0.45 else INK,
        )

    ticks = np.arange(len(classes)) + 0.5
    ax.set_xticks(ticks, classes)
    ax.set_yticks(ticks, classes)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión (accuracy {percent(accuracy)})", pad=12)
    fig.text(
        0.5, 0.01, "Párrafos y % sobre el total de cada clase real",
        ha="center", color=INK_SECONDARY, fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_class_metrics(y_true, y_pred, classes):
    """Precisión, recall y F1 por clase; solo se rotula el valor más bajo de cada una."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    metrics = dict(zip(METRIC_COLORS, (precision, recall, f1)))
    positions = np.arange(len(classes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, (name, values) in enumerate(metrics.items()):
        ax.bar(positions + (k - 1) * width, values, width - 0.015, color=METRIC_COLORS[name], label=name)

    table = np.array(list(metrics.values()))
    for c in range(len(classes)):
        lowest = table[:, c].argmin()
        ax.text(
            positions[c] + (lowest - 1) * width,
            table[lowest, c] + 0.015,
            percent(table[lowest, c], 0),
            ha="center",
            fontsize=10,
        )

    clean_axes(ax, "y")
    ax.set_xticks(positions, [f"{name}\n{n} párrafos" for name, n in zip(classes, support)])
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 5), [percent(v, 0) for v in np.linspace(0, 1, 5)])
    ax.set_title("Resultado por clase en test (rotulado: el valor más bajo)", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, handlelength=1, borderaxespad=0)
    fig.tight_layout()
    return fig


def plot_top_words(clf, words, top_n):
    """Palabras que más empujan al modelo hacia cada clase."""
    classes = clf.classes_
    fig, axes = plt.subplots(1, len(classes), figsize=(11, 5), sharex=True)
    for ax, name, weights in zip(axes, classes, clf.coef_):
        top = np.argsort(weights)[-top_n:]
        ax.barh(words[top], weights[top], height=0.55, color=BLUE)
        clean_axes(ax, "x")
        ax.set_title(name)
        ax.set_xlabel("Peso en el modelo")
    fig.suptitle(f"Las {top_n} palabras con más peso para cada clase", fontweight="semibold")
    fig.tight_layout()
    return fig


def main():
    df = pd.read_csv("data/dataset.csv", sep=";")
    meta = pd.read_csv("data/dataset_meta.csv", sep=";")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(df, df["class"], groups=meta["article"]))

    vec = TfidfVectorizer(analyzer=vortilo.make_analyzer())
    Xtr = vec.fit_transform(df["text"].iloc[tr])
    Xte = vec.transform(df["text"].iloc[te])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, df["class"].iloc[tr])

    y_true = df["class"].iloc[te]
    pred = clf.predict(Xte)
    classes = list(clf.classes_)
    accuracy = accuracy_score(y_true, pred)

    print("Accuracy:", accuracy)
    print("Referencia (predecir siempre la clase mayoritaria):", y_true.value_counts(normalize=True).max())
    print(classification_report(y_true, pred))
    print("Matriz de confusión (filas: clase real, columnas: clase predicha):")
    print(pd.DataFrame(confusion_matrix(y_true, pred, labels=classes), index=classes, columns=classes))

    figures = {
        "matriz_confusion.png": plot_confusion_matrix(y_true, pred, classes, accuracy),
        "metricas_por_clase.png": plot_class_metrics(y_true, pred, classes),
        "palabras_por_clase.png": plot_top_words(clf, vec.get_feature_names_out(), TOP_WORDS),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    for filename, fig in figures.items():
        fig.savefig(OUTPUT_DIR / filename, dpi=150)
    print(f"\nGráficos guardados en {OUTPUT_DIR}/")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
