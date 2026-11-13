"""Эмбеддеры: превращают фото и текст в векторы одного пространства."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps


class UnsupportedQuery(Exception):
    """Запрос нельзя превратить в вектор выбранным эмбеддером."""


class Embedder(Protocol):
    name: str
    description: str

    def embed_images(self, paths: list[Path]) -> list[list[float] | None]: ...
    def embed_text(self, text: str) -> list[float]: ...


def _load_rgb(path: Path | str, max_side: int | None = None) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_side:
        img.thumbnail((max_side, max_side))
    return img


# ---------------------------------------------------------------- CLIP -----
class ClipEmbedder:
    """Настоящий семантический поиск: CLIP ViT-B/32 через fastembed (ONNX, без torch).

    Модели (~350 МБ) скачиваются с HuggingFace при первом запуске.
    """

    name = "clip"
    description = "CLIP ViT-B/32 — семантический поиск по смыслу («собака на пляже»)"

    def __init__(self) -> None:
        from fastembed import ImageEmbedding, TextEmbedding

        self._text = TextEmbedding("Qdrant/clip-ViT-B-32-text")
        self._image = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")

    def embed_images(self, paths: list[Path]) -> list[list[float] | None]:
        try:
            return [v.tolist() for v in self._image.embed([str(p) for p in paths])]
        except Exception:  # один битый файл не должен ронять всю пачку
            out: list[list[float] | None] = []
            for p in paths:
                try:
                    out.append(next(iter(self._image.embed([str(p)]))).tolist())
                except Exception:
                    out.append(None)
            return out

    def embed_text(self, text: str) -> list[float]:
        return next(iter(self._text.embed([text]))).tolist()


# --------------------------------------------------------------- Цвета -----
_BINS = 4  # 4x4x4 = 64 корзины RGB
_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 30, 30), "красн": (220, 30, 30),
    "green": (40, 170, 60), "зелен": (40, 170, 60), "зелён": (40, 170, 60),
    "blue": (30, 60, 220), "син": (30, 60, 220),
    "голуб": (110, 180, 240), "sky": (110, 180, 240), "cyan": (60, 200, 220),
    "yellow": (240, 220, 40), "желт": (240, 220, 40), "жёлт": (240, 220, 40),
    "orange": (245, 140, 20), "оранж": (245, 140, 20),
    "purple": (140, 50, 190), "violet": (140, 50, 190), "фиолет": (140, 50, 190),
    "pink": (245, 130, 190), "розов": (245, 130, 190),
    "white": (245, 245, 245), "бел": (245, 245, 245),
    "black": (15, 15, 15), "чёрн": (15, 15, 15), "черн": (15, 15, 15),
    "gray": (128, 128, 128), "grey": (128, 128, 128), "сер": (128, 128, 128),
    "brown": (120, 70, 30), "корич": (120, 70, 30),
}


def _bin_centers() -> list[tuple[float, float, float]]:
    step = 256 / _BINS
    return [
        ((r + .5) * step, (g + .5) * step, (b + .5) * step)
        for r in range(_BINS) for g in range(_BINS) for b in range(_BINS)
    ]


def _normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in vec))
    return [v / n for v in vec] if n else vec


class ColorEmbedder:
    """Офлайн-запасной вариант без скачивания моделей: цветовая гистограмма.

    Ищет по названию цвета («красный закат») и по похожим картинкам,
    но не понимает смысл сцены. Нужен для запуска без интернета и для тестов.
    """

    name = "color"
    description = "Цветовая гистограмма (офлайн-режим): ищет по цветам и по похожим фото"

    def embed_images(self, paths: list[Path]) -> list[list[float] | None]:
        out: list[list[float] | None] = []
        for p in paths:
            try:
                img = _load_rgb(p, 64)
            except Exception:
                out.append(None)
                continue
            hist = [0.0] * (_BINS ** 3)
            q = 256 // _BINS
            for r, g, b in img.getdata():
                hist[(r // q) * _BINS * _BINS + (g // q) * _BINS + (b // q)] += 1
            total = sum(hist)
            out.append(_normalize([math.sqrt(h / total) for h in hist]))
        return out

    def embed_text(self, text: str) -> list[float]:
        words = re.findall(r"[\wё]+", text.lower())
        targets = [rgb for w in words for key, rgb in _COLORS.items() if w.startswith(key)]
        if not targets:
            raise UnsupportedQuery(
                "Офлайн-режим понимает только названия цветов (красный, синий, зелёный…). "
                "Для поиска по смыслу нужен режим CLIP (нужен интернет для загрузки модели)."
            )
        vec = [0.0] * (_BINS ** 3)
        for i, (cr, cg, cb) in enumerate(_bin_centers()):
            for tr, tg, tb in targets:
                d2 = (cr - tr) ** 2 + (cg - tg) ** 2 + (cb - tb) ** 2
                vec[i] += math.exp(-d2 / (2 * 55.0 ** 2))
        return _normalize([math.sqrt(v) for v in vec])


def make_embedder(kind: str = "auto") -> tuple[Embedder, str | None]:
    """Возвращает (эмбеддер, предупреждение). kind: auto | clip | color."""
    if kind == "color":
        return ColorEmbedder(), None
    try:
        return ClipEmbedder(), None
    except Exception as exc:
        if kind == "clip":
            raise
        msg = (f"Не удалось загрузить CLIP ({type(exc).__name__}); включён офлайн-режим по цветам. "
               "Проверьте интернет и перезапустите сервер для семантического поиска.")
        return ColorEmbedder(), msg
