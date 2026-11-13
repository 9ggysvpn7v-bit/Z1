from __future__ import annotations

import io
import os
import tempfile
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image, ImageOps

from .embedders import Embedder, UnsupportedQuery, make_embedder
from .faces import (DEFAULT_TOLERANCE, FaceEngine, crop_data_url, crop_jpeg, load_image,
                    make_face_engine)
from .indexer import PhotoIndex

STATIC = Path(__file__).parent / "static"
_FROM_ENV = object()  # faces не передан явно — берём из переменной FACES
ROOT = Path(__file__).resolve().parent.parent


def create_app(photo_dir: Path | None = None, data_dir: Path | None = None,
               embedder: Embedder | None = None, warning: str | None = None,
               auto_index: bool = True, faces: FaceEngine | None | object = _FROM_ENV) -> FastAPI:
    photo_dir = Path(photo_dir or os.getenv("PHOTO_DIR", ROOT.parent / "photo"))
    data_dir = Path(data_dir or os.getenv("DATA_DIR", ROOT / "data"))
    if embedder is None:
        embedder, warning = make_embedder(os.getenv("EMBEDDER", "auto"))
    faces_warning = None
    if faces is _FROM_ENV:
        faces, faces_warning = make_face_engine(os.getenv("FACES", "auto"))
    index = PhotoIndex(photo_dir, data_dir, embedder, warning, faces, faces_warning)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if auto_index:
            index.start_reindex()
        yield

    app = FastAPI(title="Photo Search", lifespan=lifespan)
    app.state.index = index

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict:
        s = index.state
        return {
            "embedder": embedder.name, "embedder_description": embedder.description,
            "warning": index.warning, "photo_dir": str(index.photo_dir.resolve()),
            "photo_dir_exists": index.photo_dir.is_dir(),
            "indexed": index.collection.count(),
            "running": s.running, "done": s.done, "total": s.total,
            "added": s.added, "updated": s.updated, "removed": s.removed,
            "skipped": s.skipped, "failed": s.failed, "error": s.error,
            "last_finished": s.last_finished,
            "faces_enabled": index.faces is not None, "faces_indexed": index.faces_count(),
            "faces_warning": index.faces_warning,
        }

    @app.post("/api/reindex")
    def reindex(full: bool = False) -> dict:
        return {"started": index.start_reindex(full=full)}

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=1), limit: int = Query(24, ge=1, le=200)) -> dict:
        try:
            return {"results": index.search_text(q, limit), "message": None}
        except UnsupportedQuery as exc:
            return {"results": [], "message": str(exc)}

    @app.post("/api/search/image")
    async def search_by_image(file: UploadFile = File(...), limit: int = 24) -> dict:
        data = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            try:
                Image.open(io.BytesIO(data)).convert("RGB").save(tmp.name, "PNG")
            except Exception:
                raise HTTPException(400, "Это не изображение")
            return {"results": index.search_image(Path(tmp.name), limit), "message": None}

    def _check_faces() -> None:
        if index.faces is None:
            raise HTTPException(503, index.faces_warning or "Поиск по лицам выключен")

    @app.post("/api/search/face")
    async def search_by_face(files: list[UploadFile] = File(...),
                             face: int | None = Query(None, ge=0),
                             tolerance: float = Query(DEFAULT_TOLERANCE, ge=0.3, le=0.8),
                             limit: int = Query(500, ge=1, le=5000)) -> dict:
        """Эталонные фото человека -> все фото из папки, где он есть.

        На каждом эталоне берётся самое крупное лицо; если файл один и на нём
        несколько лиц, нужное можно выбрать параметром face (номер из query_faces).
        """
        _check_faces()
        vecs, shown = [], []
        for n, up in enumerate(files):
            data = await up.read()
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                try:
                    Image.open(io.BytesIO(data)).save(tmp.name, "PNG")
                except Exception:
                    raise HTTPException(400, f"{up.filename}: это не изображение")
                found, img = index.detect_query_faces(Path(tmp.name))
            if not found:
                continue
            pick = face if (len(files) == 1 and face is not None) else 0
            if pick >= len(found):
                raise HTTPException(400, f"На фото только {len(found)} лиц(а)")
            vecs.append(found[pick].vec)
            shown += [{"file": n, "index": i, "chosen": i == pick, "crop": crop_data_url(img, f.box)}
                      for i, f in enumerate(found) if len(files) == 1 or i == pick]
        if not vecs:
            return {"results": [], "query_faces": [],
                    "message": "На загруженном фото не найдено лицо. Возьмите фото, где лицо видно анфас и крупно."}
        return {"results": index.search_faces(vecs, tolerance, limit), "query_faces": shown, "message": None}

    @app.get("/api/search/face_id")
    def search_by_face_id(id: list[str] = Query(...),
                          tolerance: float = Query(DEFAULT_TOLERANCE, ge=0.3, le=0.8),
                          limit: int = Query(500, ge=1, le=5000)) -> dict:
        """Поиск по лицу, которое уже есть в индексе (кнопка «ещё с этим лицом»)."""
        _check_faces()
        vecs = index.face_vectors(id)
        if not vecs:
            raise HTTPException(404, "Лицо не найдено в индексе")
        return {"results": index.search_faces(vecs, tolerance, limit), "query_faces": [], "message": None}

    @lru_cache(maxsize=1024)
    def _face_crop(path: str, mtime: float, box: str) -> bytes:
        l, t, r, b = (int(v) for v in box.split(","))
        return crop_jpeg(load_image(path), (l, t, r, b), 160)

    @app.get("/facecrop/{rel:path}")
    def facecrop(rel: str, box: str = Query(..., pattern=r"^\d+,\d+,\d+,\d+$")):
        p = index.resolve(rel)
        if not p:
            raise HTTPException(404)
        return Response(_face_crop(str(p), p.stat().st_mtime, box), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    @app.get("/photo/{rel:path}")
    def photo(rel: str):
        p = index.resolve(rel)
        if not p:
            raise HTTPException(404)
        return FileResponse(p)

    @lru_cache(maxsize=512)
    def _thumb(path: str, mtime: float) -> bytes:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img.thumbnail((480, 480))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=82)
        return buf.getvalue()

    @app.get("/thumb/{rel:path}")
    def thumb(rel: str):
        p = index.resolve(rel)
        if not p:
            raise HTTPException(404)
        return Response(_thumb(str(p), p.stat().st_mtime), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    return app


def app_factory() -> FastAPI:  # uvicorn app.main:app_factory --factory
    return create_app()
