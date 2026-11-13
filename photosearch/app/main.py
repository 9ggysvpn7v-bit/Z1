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
from .indexer import PhotoIndex

STATIC = Path(__file__).parent / "static"
ROOT = Path(__file__).resolve().parent.parent


def create_app(photo_dir: Path | None = None, data_dir: Path | None = None,
               embedder: Embedder | None = None, warning: str | None = None,
               auto_index: bool = True) -> FastAPI:
    photo_dir = Path(photo_dir or os.getenv("PHOTO_DIR", ROOT.parent / "photo"))
    data_dir = Path(data_dir or os.getenv("DATA_DIR", ROOT / "data"))
    if embedder is None:
        embedder, warning = make_embedder(os.getenv("EMBEDDER", "auto"))
    index = PhotoIndex(photo_dir, data_dir, embedder, warning)

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
