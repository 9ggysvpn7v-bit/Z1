"""Создаёт тестовые картинки в ./photo (если там пусто), чтобы проверить поиск."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

dest = Path(sys.argv[1] if len(sys.argv) > 1 else "../photo")
dest.mkdir(parents=True, exist_ok=True)
if any(dest.iterdir()):
    sys.exit(f"{dest} не пуста — ничего не создаю")

def scene(name, sky, ground, sun=None):
    im = Image.new("RGB", (640, 420), sky)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 250, 640, 420], fill=ground)
    if sun:
        d.ellipse([480, 40, 560, 120], fill=sun)
    im.save(dest / f"{name}.jpg")

scene("закат_красный", (200, 40, 40), (60, 20, 20), (250, 200, 60))
scene("море_синее", (60, 120, 230), (20, 50, 160), (250, 250, 250))
scene("лес_зелёный", (150, 210, 240), (30, 140, 50))
scene("поле_жёлтое", (130, 190, 240), (235, 210, 50), (255, 255, 255))
scene("ночь_чёрная", (10, 10, 25), (5, 5, 5), (230, 230, 230))
(dest / "поездки").mkdir()
scene_dir = dest
Image.new("RGB", (640, 420), (150, 90, 200)).save(dest / "поездки" / "сиреневый_вечер.png")
print("создано в", dest.resolve())
