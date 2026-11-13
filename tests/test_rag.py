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
                v[__import__("zlib").crc32(tok.encode()) % 8] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vectores.append([x / n for x in v])
        return vectores

    docs = [Document(id="a", text="Verstappen gano el campeonato 2023"),
            Document(id="b", text="El DRS permite adelantar abriendo el aleron")]
    index = EmbeddingIndex(docs, embed_fn=fake_embed)
    results = index.search("quien gano el campeonato 2023", top_k=1)
    assert results and results[0][0].id == "a"
    assert index.search("", top_k=1) == []


def _fake_embed_contador():
    """Embedding falso e inyectable que cuenta llamadas, para probar el cache."""
    import math

    def fake_embed(texts):
        fake_embed.llamadas += len(texts)
        vectores = []
        for t in texts:
            v = [0.0] * 8
            for tok in tokenize(t):
                v[__import__("zlib").crc32(tok.encode()) % 8] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vectores.append([x / n for x in v])
        return vectores

    fake_embed.llamadas = 0
    return fake_embed


def test_embeddings_cache_reusa_vectores():
    import tempfile
    from rag import EmbeddingIndex

    docs = [Document(id="a", text="Verstappen gano el campeonato 2023"),
            Document(id="b", text="El DRS permite adelantar abriendo el aleron")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "emb.json"
        fake = _fake_embed_contador()
        EmbeddingIndex(docs, embed_fn=fake, cache_path=cache)
        assert fake.llamadas == 2
        assert cache.exists()
        # Segunda carga con el mismo cache: no recalcula nada.
        fake2 = _fake_embed_contador()
        index2 = EmbeddingIndex(docs, embed_fn=fake2, cache_path=cache)
        assert fake2.llamadas == 0
        assert index2.search("quien gano el campeonato 2023", top_k=1)[0][0].id == "a"


def test_embeddings_cache_incremental_y_purga():
    import json
    import tempfile
    from rag import EmbeddingIndex

    docs = [Document(id="a", text="Verstappen gano el campeonato 2023"),
            Document(id="b", text="El DRS permite adelantar abriendo el aleron")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "emb.json"
        fake = _fake_embed_contador()
        EmbeddingIndex(docs, embed_fn=fake, cache_path=cache)
        # Un documento nuevo y uno borrado: solo calcula el nuevo y purga el viejo.
        docs2 = [docs[0], Document(id="c", text="Antonelli se proclamo campeon en Mexico")]
        fake2 = _fake_embed_contador()
        EmbeddingIndex(docs2, embed_fn=fake2, cache_path=cache)
        assert fake2.llamadas == 1
        guardado = json.loads(cache.read_text(encoding="utf-8"))["vectors"]
        assert len(guardado) == 2  # el vector del doc borrado se purgo
        # Un documento editado cambia de texto y se recalcula.
        docs3 = [docs[0], Document(id="c", text="Antonelli, campeon anticipado 2026")]
        fake3 = _fake_embed_contador()
        EmbeddingIndex(docs3, embed_fn=fake3, cache_path=cache)
        assert fake3.llamadas == 1


def test_embedding_cache_path_por_defecto_y_off():
    import os
    assert str(F1Rag._embedding_cache_path("data/f1_data.jsonl")).endswith("f1_data.embeddings.json")
    os.environ["RAG_EMBEDDING_CACHE"] = "off"
    try:
        assert F1Rag._embedding_cache_path("data/f1_data.jsonl") is None
    finally:
        del os.environ["RAG_EMBEDDING_CACHE"]


def _fake_embed_bow():
    """Embedding falso e inyectable (bolsa de palabras), para probar sin red."""
    import math

    def fake_embed(texts):
        vectores = []
        for t in texts:
            v = [0.0] * 8
            for tok in tokenize(t):
                v[__import__("zlib").crc32(tok.encode()) % 8] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vectores.append([x / n for x in v])
        return vectores

    return fake_embed


def test_hybrid_rrf_fusiona_los_dos_motores():
    from rag import HybridIndex

    docs = [Document(id="bm25", text="campeonato de pilotos 2023 verstappen"),
            Document(id="emb", text="ganador maximo del mundo")]
    # Embeddings controlados: "emb" es el mas parecido a la consulta para los
    # vectores, pero BM25 solo encuentra "bm25" (comparte terminos exactos).
    tabla = {"quien gano el campeonato de pilotos": [1.0, 0.0],
             "campeonato de pilotos 2023 verstappen": [0.7, 0.3],
             "ganador maximo del mundo": [1.0, 0.0]}
    index = HybridIndex(docs, embed_fn=lambda texts: [tabla[t] for t in texts])
    query = "quien gano el campeonato de pilotos"
    # Cada motor por separado solo acierta con su documento.
    assert [d.id for d, _ in index.bm25.search(query)] == ["bm25"]
    assert [d.id for d, _ in index.embeddings.search(query)][0] == "emb"
    # La fusion RRF junta ambos y da prioridad al que esta en las dos listas.
    ids = [d.id for d, _ in index.search(query)]
    assert ids == ["bm25", "emb"]
    # Sin terminos utiles, no hay resultados.
    assert index.search("el la de") == []


def test_hybrid_en_f1rag_sin_red():
    import os
    os.environ["RAG_EMBEDDING_CACHE"] = "off"
    try:
        rag = F1Rag(DATA, retriever="hybrid", embed_fn=_fake_embed_bow())
        out = rag.ask("que es el DRS")
        assert any("DRS" in s["text"] for s in out["sources"])
    finally:
        del os.environ["RAG_EMBEDDING_CACHE"]


def test_prompt_y_respuesta_con_fuentes():
    from rag import SYSTEM_PROMPT, build_prompt, extractive_answer

    doc = Document(id="driver-verstappen", text="Verstappen gano en 2023")
    prompt = build_prompt("quien gano", [(doc, 1.0)])
    assert "fuente: driver-verstappen" in prompt
    assert "[Fragmento" in SYSTEM_PROMPT
    assert "driver-verstappen" in extractive_answer([(doc, 1.0)])


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




def test_sinonimos_mundial_campeonato():
    # "mundial" debe encontrar documentos que solo dicen "campeonato".
    from rag import BM25Index

    docs = [Document(id="champ", text="Max Verstappen gano el campeonato de pilotos de 2023.",
                     metadata={"season": 2023}),
            Document(id="otro", text="El reglamento de 2023 limita el numero de motores.",
                     metadata={"season": 2023})]
    index = BM25Index(docs)
    results = index.search("quien gano el mundial 2023", top_k=1)
    assert results and results[0][0].id == "champ"


def test_sinonimos_equipo_escuderia_y_gp():
    from rag import BM25Index

    docs = [Document(id="equipo", text="Ferrari es un equipo italiano de Formula 1."),
            Document(id="carrera", text="El Monaco Grand Prix se corre en Montecarlo.")]
    index = BM25Index(docs)
    assert index.search("que escuderia es Ferrari", top_k=1)[0][0].id == "equipo"
    assert index.search("donde es el GP de Monaco", top_k=1)[0][0].id == "carrera"


def test_boost_por_temporada():
    from rag import BM25Index

    docs = [Document(id="y2023", text="El campeon de pilotos celebra su titulo.",
                     metadata={"season": 2023}),
            Document(id="y2024", text="El campeon de pilotos celebra su titulo.",
                     metadata={"season": 2024})]
    index = BM25Index(docs)
    assert index.search("campeon de pilotos 2024", top_k=1)[0][0].id == "y2024"
    assert index.search("campeon de pilotos 2023", top_k=1)[0][0].id == "y2023"
    # Sin ano en la consulta el orden no cambia (empate: gana el primero).
    assert index.search("campeon de pilotos", top_k=1)[0][0].id == "y2023"
    # Los documentos sin metadata de temporada no se penalizan.
    docs2 = [Document(id="reglas", text="El campeon suma puntos en cada gran premio."),
             Document(id="y2023b", text="El campeon de 2023 fue Verstappen.",
                      metadata={"season": 2023})]
    index2 = BM25Index(docs2)
    ids = [d.id for d, _ in index2.search("campeon 2022")]
    assert "reglas" in ids


def test_extract_years():
    from rag import extract_years

    assert extract_years("quien gano en 2024 y 2023?") == {"2023", "2024"}
    assert extract_years("que es el DRS") == set()


def test_mundial_en_datos_reales():
    # Con el corpus real, "mundial <ano>" debe dar un documento de campeon
    # de esa temporada (no una carrera ni una clasificacion).
    real = DATA.parent / "f1_data.jsonl"
    if not real.exists():
        return
    rag = F1Rag(real)
    r = rag.index.search("quien gano el mundial de pilotos 2024", top_k=1)
    assert r and r[0][0].id in {"champion-2024-drivers", "season-2024-champion"}, r[0][0].id
    assert "Verstappen" in r[0][0].text


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK {t.__name__}")
    print(f"\n{len(tests)} pruebas superadas")
