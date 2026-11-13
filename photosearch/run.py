"""Запуск: python run.py  (PORT, HOST, PHOTO_DIR, EMBEDDER=auto|clip|color)"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app_factory", factory=True,
                host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
