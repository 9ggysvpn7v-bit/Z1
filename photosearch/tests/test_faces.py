"""Поиск по лицам. FakeFaceEngine заменяет нейросеть: «лицо» — цветной квадрат,
«личность» — его цвет. Так проверяется вся логика без моделей и реальных людей."""
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.embedders import ColorEmbedder
from app.faces import Face
from app.indexer import PhotoIndex
from app.main import create_app

BG = (128, 128, 128)
ANNA, BORIS, VERA = (200, 40, 40), (40, 40, 200), (40, 200, 40)


class FakeFaceEngine:
    name = "fake"
    description = "тестовый"

    def detect(self, image):
        faces, seen = [], {}
        px = image.load()
        for y in range(0, image.height, 4):
            for x in range(0, image.width, 4):
                c = px[x, y]
                if c == BG:
                    continue
                if c not in seen:
                    seen[c] = [x, y, x, y]
                b = seen[c]
                b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x + 1), max(b[3], y + 1)
        for c, box in seen.items():
            # вектор «личности»: 0.3 расстояния между разными людьми не хватит
            faces.append(Face(tuple(box), [c[0] / 100, c[1] / 100, c[2] / 100] + [0.0] * 125))
        return faces


def photo(path, *people, size=(160, 100)):
    """Фото с «лицами» людей; people — (цвет, сторона квадрата)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(im)
    x = 8
    for color, side in people:
        d.rectangle([x, 8, x + side - 1, 8 + side - 1], fill=color)
        x += side + 12
    im.save(path)
    return path


@pytest.fixture
def photos(tmp_path):
    d = tmp_path / "photo"
    photo(d / "anna_alone.png", (ANNA, 40))
    photo(d / "anna_boris.png", (ANNA, 24), (BORIS, 40))
    photo(d / "trip" / "boris.png", (BORIS, 40))
    photo(d / "vera.png", (VERA, 40))
    photo(d / "landscape.png")
    return d


@pytest.fixture
def index(photos, tmp_path):
    return PhotoIndex(photos, tmp_path / "data", ColorEmbedder(), faces=FakeFaceEngine())


@pytest.fixture
def client(photos, tmp_path):
    app = create_app(photos, tmp_path / "data", ColorEmbedder(), auto_index=False, faces=FakeFaceEngine())
    with TestClient(app) as c:
        c.app_index = app.state.index
        c.post("/api/reindex")
        for _ in range(200):
            if not c.get("/api/status").json()["running"]:
                break
            time.sleep(0.02)
        yield c


def paths(results):
    return sorted(r["path"] for r in results)


def vec(color):
    return FakeFaceEngine().detect(Image.new("RGB", (8, 8), color))[0].vec


def test_index_stores_every_face(index):
    index.reindex_sync()
    assert index.faces_count() == 5
    got = index.face_collection.get(where={"path": "anna_boris.png"})
    assert sorted(got["ids"]) == ["anna_boris.png#0", "anna_boris.png#1"]


def test_search_finds_all_photos_of_person_and_only_them(index):
    index.reindex_sync()
    assert paths(index.search_faces([vec(ANNA)])) == ["anna_alone.png", "anna_boris.png"]
    assert paths(index.search_faces([vec(BORIS)])) == ["anna_boris.png", "trip/boris.png"]
    for r in index.search_faces([vec(BORIS)]):
        assert r["distance"] == 0 and r["score"] == 1
        assert r["face_id"] == {"anna_boris.png": "anna_boris.png#1", "trip/boris.png": "trip/boris.png#0"}[r["path"]]


def test_several_references_are_combined(index):
    index.reindex_sync()
    assert paths(index.search_faces([vec(ANNA), vec(VERA)])) == ["anna_alone.png", "anna_boris.png", "vera.png"]


def test_unknown_person_finds_nothing(index):
    index.reindex_sync()
    assert index.search_faces([vec((240, 240, 40))]) == []


def test_faces_follow_file_changes(index, photos):
    index.reindex_sync()
    (photos / "anna_alone.png").unlink()
    photo(photos / "vera.png", (ANNA, 30))  # на фото теперь Анна
    index.reindex_sync()
    assert paths(index.search_faces([vec(ANNA)])) == ["anna_boris.png", "vera.png"]
    assert index.search_faces([vec(VERA)]) == []
    assert index.faces_count() == 4


def test_existing_index_gets_faces_once_engine_enabled(photos, tmp_path):
    PhotoIndex(photos, tmp_path / "data", ColorEmbedder()).reindex_sync()
    idx = PhotoIndex(photos, tmp_path / "data", ColorEmbedder(), faces=FakeFaceEngine())
    assert idx.reindex_sync().updated == 5 and idx.faces_count() == 5
    assert idx.reindex_sync().skipped == 5


# ---- HTTP API ----
def upload(client, *files, **params):
    return client.post("/api/search/face", params=params,
                       files=[("files", (f.name, f.read_bytes(), "image/png")) for f in files])


def test_api_status_reports_faces(client):
    s = client.get("/api/status").json()
    assert s["faces_enabled"] and s["faces_indexed"] == 5


def test_api_search_by_face(client, tmp_path):
    d = upload(client, photo(tmp_path / "q.png", (BORIS, 50))).json()
    assert paths(d["results"]) == ["anna_boris.png", "trip/boris.png"]
    assert len(d["query_faces"]) == 1 and d["query_faces"][0]["crop"].startswith("data:image/jpeg")


def test_api_group_photo_uses_largest_face_or_chosen_one(client, tmp_path):
    q = photo(tmp_path / "group.png", (VERA, 20), (ANNA, 50))
    d = upload(client, q).json()
    assert paths(d["results"]) == ["anna_alone.png", "anna_boris.png"]
    assert [f["chosen"] for f in d["query_faces"]] == [True, False]
    assert paths(upload(client, q, face=1).json()["results"]) == ["vera.png"]
    assert upload(client, q, face=5).status_code == 400


def test_api_multiple_reference_files(client, tmp_path):
    d = upload(client, photo(tmp_path / "a.png", (ANNA, 40)), photo(tmp_path / "v.png", (VERA, 40))).json()
    assert paths(d["results"]) == ["anna_alone.png", "anna_boris.png", "vera.png"]


def test_api_no_face_and_bad_file(client, tmp_path):
    d = upload(client, photo(tmp_path / "empty.png")).json()
    assert d["results"] == [] and "лицо" in d["message"]
    bad = client.post("/api/search/face", files=[("files", ("x.png", b"nope", "image/png"))])
    assert bad.status_code == 400


def test_api_tolerance_validation(client, tmp_path):
    assert upload(client, photo(tmp_path / "q.png", (ANNA, 40)), tolerance=0.1).status_code == 422


def test_api_search_by_indexed_face_and_crop(client):
    r = client.get("/api/search/face_id", params={"id": "trip/boris.png#0"}).json()
    assert paths(r["results"]) == ["anna_boris.png", "trip/boris.png"]
    hit = next(x for x in r["results"] if x["path"] == "anna_boris.png")
    c = client.get("/facecrop/anna_boris.png", params={"box": hit["box"]})
    assert c.status_code == 200 and c.headers["content-type"] == "image/jpeg"
    assert client.get("/api/search/face_id", params={"id": "nope#0"}).status_code == 404
    assert client.get("/facecrop/anna_boris.png", params={"box": "1,2"}).status_code == 422
    assert client.get("/facecrop/../x.png", params={"box": "1,2,3,4"}).status_code == 404


def test_api_faces_disabled(tmp_path, photos):
    app = create_app(photos, tmp_path / "d", ColorEmbedder(), auto_index=False, faces=None)
    with TestClient(app) as c:
        assert c.get("/api/status").json()["faces_enabled"] is False
        r = c.post("/api/search/face", files=[("files", ("q.png", b"x", "image/png"))])
        assert r.status_code == 503
