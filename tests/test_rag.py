"""Pruebas basicas del pipeline RAG. Ejecutar: python tests/test_rag.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag import F1Rag, TfidfIndex, Document, tokenize  # noqa: E402

DATA = Path(__file__).parent.parent / "data" / "sample_data.jsonl"


def test_tokenize():
    tokens = tokenize("¿Quién ganó el Mundial de 2024?")
    assert "gano" in tokens and "2024" in tokens
    assert "el" not in tokens and "de" not in tokens


def test_retrieval_champion():
    rag = F1Rag(DATA)
    results = rag.index.search("quien gano el campeonato de pilotos 2024", top_k=1)
    assert results, "sin resultados"
    assert "Verstappen" in results[0][0].text


def test_retrieval_rules():
    rag = F1Rag(DATA)
    results = rag.index.search("cuantos puntos da ganar una carrera", top_k=1)
    assert results and "25" in results[0][0].text


def test_ask_structure():
    rag = F1Rag(DATA)
    out = rag.ask("que es el DRS")
    assert "answer" in out and "sources" in out
    assert any("DRS" in s["text"] for s in out["sources"])


def test_empty_query():
    index = TfidfIndex([Document(id="x", text="prueba")])
    assert index.search("el la de") == []


def test_embeddings_index_sin_red():
    # Funcion de embedding falsa e inyectable: bolsa de palabras normalizada,
    # suficiente para comprobar el pipeline de vectores sin llamar a OpenAI.
    import math
    from rag import EmbeddingIndex

    def fake_embed(texts):
        vectores = []
        for t in texts:
            toks = tokenize(t)
            v = [0.0] * 8
            for tok in toks:
                v[hash(tok) % 8] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vectores.append([x / n for x in v])
        return vectores

    docs = [Document(id="a", text="Verstappen gano el campeonato 2023"),
            Document(id="b", text="El DRS permite adelantar abriendo el aleron")]
    index = EmbeddingIndex(docs, embed_fn=fake_embed)
    results = index.search("quien gano el campeonato 2023", top_k=1)
    assert results and results[0][0].id == "a"
    assert index.search("", top_k=1) == []


def test_retriever_desconocido():
    try:
        F1Rag(DATA, retriever="magia")
    except ValueError:
        return
    raise AssertionError("debio lanzar ValueError")


def test_datos_reales_multitemporada():
    # Solo si existe la base descargada (2021-2026).
    real = DATA.parent / "f1_data.jsonl"
    if not real.exists():
        return
    rag = F1Rag(real)
    assert len(rag.documents) > 400
    r24 = rag.index.search("quien gano el campeonato de pilotos 2024", top_k=1)
    assert r24 and "Verstappen" in r24[0][0].text
    r26 = rag.index.search("quien gano el campeonato de pilotos 2026", top_k=1)
    assert r26, "sin resultados para 2026"
    assert "Antonelli" in r26[0][0].text


def test_api_si_fastapi_instalado():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        return
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import api
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    resp = client.post("/ask", json={"question": "que es el DRS", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body and body["sources"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK {t.__name__}")
    print(f"\n{len(tests)} pruebas superadas")
