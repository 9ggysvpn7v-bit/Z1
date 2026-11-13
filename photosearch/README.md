# Поиск по фото

FastAPI + ChromaDB + чистый HTML/JS. Фото берутся из `../photo` (рядом с `photosearch/`).

```bash
pip install -r requirements.txt
python run.py            # http://127.0.0.1:8000
python -m pytest         # тесты
```

Переменные: `PHOTO_DIR`, `DATA_DIR`, `PORT`, `HOST`, `EMBEDDER=auto|clip|color`.

- **clip** — семантический поиск («собака на пляже»), CLIP ViT-B/32 через fastembed; модель (~350 МБ) скачивается при первом запуске.
- **color** — офлайн-режим без моделей: ищет по названию цвета и по похожим фото.
- `auto` пробует CLIP и при неудаче переключается на color (в интерфейсе появится предупреждение).

Кнопка «Переиндексировать» работает инкрементально (новые/изменённые/удалённые файлы), прогресс виден в UI.
Тестовые картинки: `python scripts/make_samples.py ../photo` (только если папка пуста).
