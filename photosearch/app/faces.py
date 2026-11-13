"""Поиск и распознавание лиц: каждое лицо на фото превращается в 128-мерный вектор.

Расстояние между векторами одного человека обычно < 0.6, разных людей — больше.
Движок по умолчанию — dlib (ResNet, ~99.4 % на LFW); модели ставятся пакетом
face_recognition_models с PyPI, без скачивания с других сайтов.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

DEFAULT_TOLERANCE = 0.55  # чем меньше, тем строже совпадение
MAX_SIDE = 1600           # большие фото уменьшаем: быстрее и почти без потери качества


@dataclass
class Face:
    box: tuple[int, int, int, int]  # left, top, right, bottom — в пикселях исходного фото
    vec: list[float]

    @property
    def area(self) -> int:
        l, t, r, b = self.box
        return max(0, r - l) * max(0, b - t)


class FaceEngine(Protocol):
    name: str
    description: str

    def detect(self, image: Image.Image) -> list[Face]: ...


def load_image(path: Path | str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def detect_file(engine: FaceEngine, path: Path | str) -> list[Face] | None:
    """Лица на фото; None — если файл не читается."""
    try:
        img = load_image(path)
    except Exception:
        return None
    return engine.detect(img)


def crop_jpeg(img: Image.Image, box: tuple[int, int, int, int], size: int = 112) -> bytes:
    """Миниатюра лица (с полями вокруг) в JPEG."""
    l, t, r, b = box
    pad = int(max(r - l, b - t) * 0.35)
    c = img.crop((max(0, l - pad), max(0, t - pad), min(img.width, r + pad), min(img.height, b + pad)))
    c.thumbnail((size, size))
    buf = io.BytesIO()
    c.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def crop_data_url(img: Image.Image, box: tuple[int, int, int, int], size: int = 112) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(crop_jpeg(img, box, size)).decode()


class DlibFaceEngine:
    name = "dlib"
    description = "dlib ResNet: 128-мерные векторы лиц"

    def __init__(self, jitters: int = 1) -> None:
        import dlib
        import face_recognition_models as frm

        self._np = __import__("numpy")
        self._detector = dlib.get_frontal_face_detector()
        self._shape = dlib.shape_predictor(frm.pose_predictor_five_point_model_location())
        self._rec = dlib.face_recognition_model_v1(frm.face_recognition_model_location())
        self.jitters = jitters

    def detect(self, image: Image.Image) -> list[Face]:
        w, h = image.size
        scale = 1.0
        if max(image.size) > MAX_SIDE:
            scale = MAX_SIDE / max(image.size)
            image = image.resize((round(image.width * scale), round(image.height * scale)))
        arr = self._np.asarray(image)
        # маленькие фото увеличиваем при поиске, чтобы находить мелкие лица
        upsample = 2 if max(image.size) < 800 else 1
        faces = []
        for rect in self._detector(arr, upsample):
            vec = self._rec.compute_face_descriptor(arr, self._shape(arr, rect), self.jitters)
            l, t, r, btm = (int(round(v / scale)) for v in (rect.left(), rect.top(), rect.right(), rect.bottom()))
            box = (max(0, l), max(0, t), min(w, r), min(h, btm))  # лицо у края может «вылезать» за кадр
            faces.append(Face(box, list(vec)))
        return faces


def make_face_engine(kind: str = "auto") -> tuple[FaceEngine | None, str | None]:
    """(движок, предупреждение). kind: auto | dlib | off."""
    if kind == "off":
        return None, None
    try:
        return DlibFaceEngine(), None
    except Exception as exc:
        if kind == "dlib":
            raise
        return None, (f"Поиск по лицам выключен: не удалось загрузить dlib ({type(exc).__name__}). "
                      "Установите: pip install dlib-bin face_recognition_models")
