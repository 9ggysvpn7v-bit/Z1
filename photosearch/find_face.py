"""Найти все фото с нужным человеком — без сервера, одной командой.

    python find_face.py эталон.jpg [ещё_эталон.jpg ...] --photos ПАПКА [--out НАЙДЕНО] [--tolerance 0.55]

Эталон — фото, где лицо человека видно крупно. Если на эталоне несколько лиц,
берётся самое крупное (или укажите --face N). Несколько эталонов (разные ракурсы,
возраст, очки) повышают число находок. Результат печатается и сохраняется в CSV;
с --out найденные фото копируются в отдельную папку (оригиналы не трогаются).
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

from app.faces import DEFAULT_TOLERANCE, detect_file, make_face_engine
from app.indexer import scan_photos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Поиск всех фото с нужным лицом в папке")
    ap.add_argument("reference", nargs="+", type=Path, help="фото человека, которого ищем")
    ap.add_argument("--photos", type=Path, required=True, help="папка с фото (ищет во вложенных тоже)")
    ap.add_argument("--out", type=Path, help="куда скопировать найденные фото")
    ap.add_argument("--csv", type=Path, default=Path("found_faces.csv"), help="файл с результатами")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help=f"строгость: меньше — строже (0.45 строго, 0.6 мягко), по умолчанию {DEFAULT_TOLERANCE}")
    ap.add_argument("--face", type=int, help="номер лица на эталоне, если их несколько (0 — самое крупное)")
    args = ap.parse_args(argv)

    engine, _ = make_face_engine("dlib")
    np = __import__("numpy")

    refs = []
    for ref in args.reference:
        faces = detect_file(engine, ref)
        if faces is None:
            sys.exit(f"Не удалось открыть {ref}")
        if not faces:
            sys.exit(f"На {ref} не найдено лицо — возьмите фото, где лицо видно анфас и крупно")
        faces.sort(key=lambda f: -f.area)
        pick = args.face or 0
        if pick >= len(faces):
            sys.exit(f"На {ref} только {len(faces)} лиц(а)")
        if len(faces) > 1:
            print(f"{ref}: лиц на фото — {len(faces)}, беру №{pick} (самое крупное = 0)")
        refs.append(np.array(faces[pick].vec))
    refs = np.stack(refs)

    files = scan_photos(args.photos)
    if not files:
        sys.exit(f"В {args.photos} нет фото")
    print(f"Проверяю {len(files)} фото…")

    found, failed, t0 = [], [], time.time()
    for n, (rel, path) in enumerate(files.items(), 1):
        faces = detect_file(engine, path)
        if faces is None:
            failed.append(rel)
        best = None
        for f in faces or []:
            d = float(np.linalg.norm(refs - np.array(f.vec), axis=1).min())
            if d <= args.tolerance and (best is None or d < best[0]):
                best = (d, f.box)
        if best:
            found.append((rel, path, *best))
        if n % 25 == 0 or n == len(files):
            print(f"\r  {n}/{len(files)}  найдено: {len(found)}  ({time.time() - t0:.0f} с)", end="", flush=True)
    print()

    found.sort(key=lambda r: r[2])
    with args.csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "similarity_%", "distance", "face_box"])
        for rel, _p, d, box in found:
            w.writerow([rel, round((1 - d) * 100), round(d, 4), ",".join(map(str, box))])

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for rel, p, d, _box in found:
            # в имени — процент сходства и исходный путь, чтобы не было конфликтов имён
            shutil.copy2(p, args.out / f"{round((1 - d) * 100):02d}%_{rel.replace('/', '__')}")

    for rel, _p, d, _box in found:
        print(f"{round((1 - d) * 100):3d}%  {rel}")
    print(f"\nНайдено фото: {len(found)} из {len(files)}. Список: {args.csv}"
          + (f", копии: {args.out}" if args.out else ""))
    if failed:
        print(f"Не удалось прочитать: {len(failed)} файл(ов), например {failed[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
