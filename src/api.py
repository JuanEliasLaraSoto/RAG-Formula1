"""API REST del RAG de Formula 1 con FastAPI.

Uso:
    pip install fastapi uvicorn
    uvicorn src.api:app --reload

Endpoints:
    GET  /health           -> estado y numero de documentos indexados
    POST /ask              -> {"question": "...", "top_k": 5} -> respuesta + fuentes
    GET  /docs             -> documentacion interactiva (Swagger UI)

La base de conocimiento se elige con la variable de entorno RAG_DATA
(por defecto data/f1_data.jsonl si existe, si no data/sample_data.jsonl).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from rag import F1Rag  # noqa: E402

DATA_DEFAULT = Path("data/f1_data.jsonl")
DATA_FALLBACK = Path("data/sample_data.jsonl")

app = FastAPI(
    title="F1 RAG API",
    description="Preguntas en lenguaje natural sobre Formula 1, con recuperacion BM25 o embeddings.",
    version="1.0.0",
)

_rag: F1Rag | None = None


def get_rag() -> F1Rag:
    """Carga perezosa: el indice se construye en la primera peticion."""
    global _rag
    if _rag is None:
        import os
        data = Path(os.environ.get("RAG_DATA", "") or (DATA_DEFAULT if DATA_DEFAULT.exists() else DATA_FALLBACK))
        if not data.exists():
            raise HTTPException(status_code=500, detail=f"No existe la base de conocimiento: {data}")
        _rag = F1Rag(data)
    return _rag


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["quien gano el mundial de 2024?"])
    top_k: int = Field(5, ge=1, le=20)


class Source(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    rag = get_rag()
    return {"status": "ok", "documents": len(rag.documents)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    return get_rag().ask(req.question, top_k=req.top_k)
