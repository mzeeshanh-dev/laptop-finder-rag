"""FastAPI entry point.  Run with:  python app/main.py   ->  http://localhost:8000"""
from __future__ import annotations

import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# Allow `python app/main.py` (script mode) as well as `uvicorn app.main:app`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import config  # noqa: E402
from app.answer.builder import card  # noqa: E402
from app.engine import kb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("laptop-rag")


def _build_in_background() -> None:
    try:
        kb.build()
    except Exception as exc:  # noqa: BLE001
        log.exception("Index build failed")
        kb.status.update(stage="error", message=f"Index build failed: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_build_in_background, daemon=True, name="kb-build").start()
    yield


app = FastAPI(title="Laptop RAG (no-LLM)", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field("default", max_length=64)


class ResetRequest(BaseModel):
    session_id: str = Field("default", max_length=64)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    return {**kb.status, "stats": kb.stats}


@app.post("/api/chat")
def chat(req: ChatRequest):
    return kb.ask(req.message, req.session_id)


@app.post("/api/reset")
def reset(req: ResetRequest):
    kb.reset_session(req.session_id)
    return {"ok": True}


@app.get("/api/laptops/{laptop_id}")
def laptop(laptop_id: int):
    if kb.df is None or laptop_id not in kb.df.index:
        raise HTTPException(404, "Laptop not found")
    return card(kb.df.loc[laptop_id])


if __name__ == "__main__":
    print(f"\n  Laptop RAG running at http://localhost:{config.PORT}\n")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
