import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.embedders import ColorEmbedder, UnsupportedQuery
from app.indexer import PhotoIndex, scan_photos
from app.main import create_app


def solid(path, color, size=(80, 60)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


@pytest.fixture
def photos(tmp_path):
    d = tmp_path / "photo"
    solid(d / "red.jpg", (220, 30, 30))
    solid(d / "blue.png", (30, 60, 220))
    solid(d / "sub" / "green.jpg", (40, 170, 60))
    (d / "notes.txt").write_text("not an image")
    return d


@pytest.fixture
def index(photos, tmp_path):
    return PhotoIndex(photos, tmp_path / "data", ColorEmbedder())


@pytest.fixture
def client(photos, tmp_path):
    app = create_app(photos, tmp_path / "data", ColorEmbedder(), auto_index=False, faces=None)
    with TestClient(app) as c:
        c.app_index = app.state.index
        yield c


def wait_idle(client):
    for _ in range(200):
        if not client.get("/api/status").json()["running"]:
            return client.get("/api/status").json()
        time.sleep(0.02)
    raise AssertionError("indexing did not finish")


def test_scan_finds_only_images_recursively(photos):
    assert set(scan_photos(photos)) == {"red.jpg", "blue.png", "sub/green.jpg"}


def test_scan_missing_dir(tmp_path):
    assert scan_photos(tmp_path / "nope") == {}


def test_index_adds_and_search_by_color(index):
    st = index.reindex_sync()
    assert (st.added, st.error) == (3, None)
    assert index.collection.count() == 3
    assert index.search_text("красный", 1)[0]["path"] == "red.jpg"
    assert index.search_text("blue sky", 3)[0]["path"] == "blue.png"
    assert index.search_text("зелёный", 1)[0]["path"] == "sub/green.jpg"


def test_reindex_is_incremental(index, photos):
    index.reindex_sync()
    st = index.reindex_sync()
    assert (st.added, st.updated, st.skipped) == (0, 0, 3)

    solid(photos / "new.jpg", (240, 220, 40))
    solid(photos / "red.jpg", (10, 10, 10), (81, 60))  # изменён
    (photos / "blue.png").unlink()                     # удалён
    st = index.reindex_sync()
    assert (st.added, st.updated, st.removed) == (1, 1, 1)
    assert index.search_text("жёлтый", 1)[0]["path"] == "new.jpg"
    assert index.search_text("чёрный", 1)[0]["path"] == "red.jpg"
    assert "blue.png" not in [r["path"] for r in index.search_text("синий", 10)]


def test_full_reindex_reprocesses_everything(index):
    index.reindex_sync()
    assert index.reindex_sync(full=True).updated == 3


def test_corrupt_image_reported_not_fatal(index, photos):
    (photos / "broken.jpg").write_bytes(b"garbage")
    st = index.reindex_sync()
    assert st.failed == ["broken.jpg"] and st.added == 3 and st.error is None


def test_unsupported_text_query():
    with pytest.raises(UnsupportedQuery):
        ColorEmbedder().embed_text("собака на пляже")


def test_resolve_blocks_path_traversal(index, tmp_path):
    (tmp_path / "secret.png").write_bytes(b"x")
    assert index.resolve("../secret.png") is None
    assert index.resolve("red.jpg") is not None
    assert index.resolve("notes.txt") is None


# ---- HTTP API ----
def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Переиндексировать" in r.text


def test_api_reindex_and_search(client):
    assert client.get("/api/status").json()["indexed"] == 0
    assert client.post("/api/reindex").json() == {"started": True}
    s = wait_idle(client)
    assert s["indexed"] == 3 and s["added"] == 3 and s["embedder"] == "color"
    res = client.get("/api/search", params={"q": "красный", "limit": 2}).json()
    assert res["results"][0]["path"] == "red.jpg" and len(res["results"]) == 2


def test_api_reindex_rejects_concurrent_run(client):
    client.app_index.state.running = True
    assert client.post("/api/reindex").json() == {"started": False}
    client.app_index.state.running = False


def test_api_search_unsupported_query_gives_message(client):
    client.post("/api/reindex"); wait_idle(client)
    d = client.get("/api/search", params={"q": "кот"}).json()
    assert d["results"] == [] and "цвет" in d["message"]


def test_api_search_validation(client):
    assert client.get("/api/search").status_code == 422
    assert client.get("/api/search", params={"q": "red", "limit": 0}).status_code == 422


def test_api_search_empty_index(client):
    assert client.get("/api/search", params={"q": "red"}).json()["results"] == []


def test_api_search_by_image(client, tmp_path):
    client.post("/api/reindex"); wait_idle(client)
    q = tmp_path / "q.png"; solid(q, (225, 35, 35))
    with q.open("rb") as f:
        r = client.post("/api/search/image", files={"file": ("q.png", f, "image/png")})
    assert r.json()["results"][0]["path"] == "red.jpg"
    bad = client.post("/api/search/image", files={"file": ("x.png", b"nope", "image/png")})
    assert bad.status_code == 400


def test_photo_and_thumb_endpoints(client):
    assert client.get("/photo/red.jpg").status_code == 200
    assert client.get("/photo/sub/green.jpg").status_code == 200
    t = client.get("/thumb/red.jpg")
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg"
    assert client.get("/photo/notes.txt").status_code == 404
    assert client.get("/photo/%2e%2e/%2e%2e/etc/passwd").status_code == 404
