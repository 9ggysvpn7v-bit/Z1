"""Сканирование папки с фото и синхронизация с ChromaDB (инкрементально)."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from .embedders import Embedder
from .faces import DEFAULT_TOLERANCE, Face, FaceEngine, detect_file, load_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
BATCH = 16


def scan_photos(root: Path) -> dict[str, Path]:
    """{относительный posix-путь: абсолютный путь} для всех картинок."""
    root = root.resolve()
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
            found[p.relative_to(root).as_posix()] = p
    return found


@dataclass
class IndexState:
    running: bool = False
    done: int = 0
    total: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    error: str | None = None
    last_finished: float | None = None


class PhotoIndex:
    def __init__(self, photo_dir: Path, data_dir: Path, embedder: Embedder, warning: str | None = None,
                 faces: FaceEngine | None = None, faces_warning: str | None = None):
        self.photo_dir = Path(photo_dir)
        self.embedder = embedder
        self.warning = warning
        self.faces = faces
        self.faces_warning = faces_warning
        import chromadb  # импорт здесь: find_face.py работает и без chromadb

        self.client = chromadb.PersistentClient(path=str(Path(data_dir) / "chroma"))
        # отдельная коллекция на эмбеддер: размерности векторов не смешиваются
        self.collection = self.client.get_or_create_collection(
            f"photos_{embedder.name}", metadata={"hnsw:space": "cosine"}
        )
        # одна запись на каждое найденное лицо: id = "путь#номер", metadata.path = путь к фото
        self.face_collection = (
            self.client.get_or_create_collection(f"faces_{faces.name}", metadata={"hnsw:space": "l2"})
            if faces else None
        )
        self.state = IndexState()
        self._lock = threading.Lock()

    # -- индексация --
    def start_reindex(self, full: bool = False) -> bool:
        """Запускает фоновую переиндексацию. False, если она уже идёт."""
        with self._lock:
            if self.state.running:
                return False
            self.state = IndexState(running=True)
        threading.Thread(target=self._run, args=(full,), daemon=True).start()
        return True

    def reindex_sync(self, full: bool = False) -> IndexState:
        if self.start_reindex(full):
            while self.state.running:
                threading.Event().wait(0.02)
        return self.state

    def _run(self, full: bool) -> None:
        import time

        st = self.state
        try:
            files = scan_photos(self.photo_dir)
            existing = self.collection.get(include=["metadatas"])
            known = {i: m for i, m in zip(existing["ids"], existing["metadatas"])}

            stale = [i for i in known if i not in files]
            if stale:
                self.collection.delete(ids=stale)
                self._delete_faces(stale)
                st.removed = len(stale)
            face_tag = self.faces.name if self.faces else None

            todo: list[tuple[str, Path, int, int]] = []
            for rel, p in files.items():
                s = p.stat()
                sig = (s.st_mtime_ns, s.st_size)
                m = known.get(rel)
                if (not full and m and (m.get("mtime_ns"), m.get("size")) == sig
                        and (face_tag is None or m.get("faces") == face_tag)):
                    st.skipped += 1
                else:
                    todo.append((rel, p, *sig))
            st.total = len(todo)

            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i + BATCH]
                vecs = self.embedder.embed_images([c[1] for c in chunk])
                ids, embs, metas = [], [], []
                for (rel, p, mt, size), v in zip(chunk, vecs):
                    if v is None:
                        st.failed.append(rel)
                        continue
                    meta = {"mtime_ns": mt, "size": size}
                    if self.faces:
                        self._index_faces(rel, p)
                        meta["faces"] = face_tag
                    ids.append(rel)
                    embs.append(v)
                    metas.append(meta)
                    if rel in known:
                        st.updated += 1
                    else:
                        st.added += 1
                if ids:
                    self.collection.upsert(ids=ids, embeddings=embs, metadatas=metas)
                st.done += len(chunk)
        except Exception as exc:  # показываем причину в UI
            st.error = f"{type(exc).__name__}: {exc}"
        finally:
            st.last_finished = time.time()
            st.running = False

    # -- лица --
    def _delete_faces(self, rels: list[str]) -> None:
        if self.face_collection is not None and rels:
            self.face_collection.delete(where={"path": {"$in": rels}})

    def _index_faces(self, rel: str, path: Path) -> None:
        self._delete_faces([rel])
        found = detect_file(self.faces, path) or []
        if found:
            self.face_collection.upsert(
                ids=[f"{rel}#{n}" for n in range(len(found))],
                embeddings=[f.vec for f in found],
                metadatas=[{"path": rel, "box": ",".join(map(str, f.box))} for f in found],
            )

    def faces_count(self) -> int:
        return self.face_collection.count() if self.face_collection is not None else 0

    def search_faces(self, vecs: list[list[float]], tolerance: float = DEFAULT_TOLERANCE,
                     limit: int = 500) -> list[dict]:
        """Все фото, где есть лицо ближе tolerance хотя бы к одному из эталонов."""
        if self.face_collection is None:
            raise RuntimeError(self.faces_warning or "Поиск по лицам выключен")
        n = self.face_collection.count()
        if n == 0 or not vecs:
            return []
        res = self.face_collection.query(query_embeddings=vecs, n_results=min(n, 20000),
                                         include=["metadatas", "distances"])
        best: dict[str, dict] = {}
        for ids, metas, dists in zip(res["ids"], res["metadatas"], res["distances"]):
            for fid, meta, d2 in zip(ids, metas, dists):
                d = max(d2, 0.0) ** 0.5  # chroma l2 = квадрат расстояния
                if d > tolerance:
                    break  # результаты отсортированы по расстоянию
                path = meta["path"]
                if path not in best or d < best[path]["distance"]:
                    best[path] = {"path": path, "face_id": fid, "box": meta["box"],
                                  "distance": round(d, 4), "score": round(max(0.0, 1 - d), 4)}
        return sorted(best.values(), key=lambda r: r["distance"])[:limit]

    def face_vectors(self, face_ids: list[str]) -> list[list[float]]:
        if self.face_collection is None:
            return []
        got = self.face_collection.get(ids=face_ids, include=["embeddings"])
        return [list(v) for v in got["embeddings"]]

    def detect_query_faces(self, path: Path) -> tuple[list[Face], "object"]:
        """Лица на загруженном эталонном фото (от крупного к мелкому) и само изображение."""
        if self.faces is None:
            raise RuntimeError(self.faces_warning or "Поиск по лицам выключен")
        img = load_image(path)
        return sorted(self.faces.detect(img), key=lambda f: -f.area), img

    # -- поиск --
    def _query(self, vec: list[float], limit: int) -> list[dict]:
        n = self.collection.count()
        if n == 0:
            return []
        res = self.collection.query(query_embeddings=[vec], n_results=min(limit, n))
        return [
            {"path": i, "score": round(1 - d, 4)}
            for i, d in zip(res["ids"][0], res["distances"][0])
        ]

    def search_text(self, text: str, limit: int = 24) -> list[dict]:
        return self._query(self.embedder.embed_text(text), limit)

    def search_image(self, path: Path, limit: int = 24) -> list[dict]:
        vec = self.embedder.embed_images([path])[0]
        if vec is None:
            raise ValueError("Не удалось прочитать изображение")
        return self._query(vec, limit)

    def resolve(self, rel: str) -> Path | None:
        """Безопасно превращает относительный путь в файл внутри папки с фото."""
        root = self.photo_dir.resolve()
        p = (root / rel).resolve()
        if root in p.parents and p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return p
        return None
