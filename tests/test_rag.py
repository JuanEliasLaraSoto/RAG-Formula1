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
            Document(id="emb", text="el mas grande de la historia")]
    # Embeddings controlados: "emb" es el mas parecido a la consulta para los
    # vectores, pero BM25 solo encuentra "bm25" (comparte terminos exactos).
    tabla = {"quien gano el campeonato de pilotos": [1.0, 0.0],
             "campeonato de pilotos 2023 verstappen": [0.7, 0.3],
             "el mas grande de la historia": [1.0, 0.0]}
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




def test_stopwords_interrogativas_y_auxiliares():
    # Las palabras de pregunta y los auxiliares no deben actuar como terminos.
    tokens = tokenize("quien ha ganado el mundial de 2023")
    for w in ("quien", "ha", "el", "de"):
        assert w not in tokens
    assert "ganado" in tokens and "mundial" in tokens and "2023" in tokens


def test_cluster_victoria_en_sinonimos():
    # "gano" en la consulta encuentra documentos que hablan de "victorias".
    from rag import BM25Index

    docs = [Document(id="record", text="Tiene 105 victorias y 104 poles en su carrera."),
            Document(id="otro", text="El reglamento tecnico limita el aleron.")]
    index = BM25Index(docs)
    results = index.search("cuantas carreras gano", top_k=1)
    assert results and results[0][0].id == "record"




def test_diversify_groups_limita_plantillas_repetidas():
    # Una consulta amplia ya no llena el top-k con la misma plantilla:
    # maximo 2 documentos del mismo (tipo, temporada).
    from rag import BM25Index

    docs = [Document(id=f"standing-{i}", text=f"mundial pilotos {i}",
                     metadata={"type": "standing", "season": 2026}) for i in range(5)]
    docs.append(Document(id="reglas",
                         text="el mundial de pilotos reparte puntos segun la posicion final de cada carrera"))
    docs.append(Document(id="historia",
                         text="el campeonato del mundo arranco en el ano 1950 en silverstone"))
    index = BM25Index(docs)
    ids = [d.id for d, _ in index.search("mundial de pilotos", top_k=4)]
    assert ids[0] == "standing-0"  # el mejor resultado nunca se mueve
    assert ids == ["standing-0", "standing-1", "reglas", "historia"]
    # Desactivado, vuelve el ranking puro: todo la misma plantilla.
    ids_raw = [d.id for d, _ in index.search("mundial de pilotos", top_k=4, diversify=False)]
    assert ids_raw == ["standing-0", "standing-1", "standing-2", "standing-3"]


def test_diversify_groups_rellena_si_no_hay_variedad():
    from rag import diversify_groups, Document

    docs = [(Document(id=f"s-{i}", text="x", metadata={"type": "standing", "season": 2026}),
             1.0 - i * 0.01) for i in range(5)]
    out = diversify_groups(docs, top_k=4, max_per_group=2)
    assert len(out) == 4  # rellena con los descartados, en orden
    assert [d.id for d, _ in out][:2] == ["s-0", "s-1"]
    # Los documentos sin tipo+temporada nunca se limitan.
    libres = [(Document(id=f"libre-{i}", text="x"), 1.0 - i * 0.01) for i in range(4)]
    assert len(diversify_groups(libres, top_k=4)) == 4


def test_diversify_desactivable_por_entorno():
    import os
    from rag import diversify_enabled

    assert diversify_enabled(None) is True
    os.environ["RAG_DIVERSIFY"] = "off"
    try:
        assert diversify_enabled(None) is False
    finally:
        del os.environ["RAG_DIVERSIFY"]
    assert diversify_enabled(False) is False
    assert diversify_enabled(True) is True





def test_apodos_de_pilotos_expanden_al_nombre():
    # "Checo" no aparece en los documentos, pero refiere a Sergio Perez:
    # el documento sobre Perez gana al historico de Schumacher.
    from rag import BM25Index

    docs = [Document(id="perez", text="Sergio Perez gano el Monaco Grand Prix de la temporada 2022.",
                     metadata={"season": 2022}),
            Document(id="schumi", text="Michael Schumacher gano siete titulos en su carrera.")]
    index = BM25Index(docs)
    results = index.search("cuantas carreras gano Checo", top_k=1)
    assert results and results[0][0].id == "perez"


def test_apodos_magic_y_seb():
    from rag import BM25Index

    docs = [Document(id="alonso", text="Fernando Alonso es un piloto de Formula 1 (Spanish)."),
            Document(id="vettel", text="Sebastian Vettel es un piloto de Formula 1 (German).")]
    index = BM25Index(docs)
    assert index.search("que tal Magic", top_k=1)[0][0].id == "alonso"
    assert index.search("que tal Seb", top_k=1)[0][0].id == "vettel"


def test_kimi_no_es_alias_por_ser_ambiguo():
    # "Kimi" vale para Raikkonen y para Andrea Kimi Antonelli: no puede ser
    # alias de uno solo. Su nombre ya esta en las fichas, asi que la
    # temporada de la consulta desempata.
    from rag import ALIASES

    assert "kimi" not in ALIASES


def test_apodos_no_roban_consultas_sin_piloto():
    # Sin apodo en la consulta, el ranking no cambia.
    from rag import BM25Index

    docs = [Document(id="reglas", text="Los diez primeros clasificados puntuan en cada carrera."),
            Document(id="perez", text="Sergio Perez es un piloto de Formula 1 (Mexican).")]
    index = BM25Index(docs)
    results = index.search("como se puntua en las carreras", top_k=1)
    assert results and results[0][0].id == "reglas"


def test_apodos_en_datos_reales():
    # Con el corpus real, los apodos recuperan documentos de SU piloto.
    real = DATA.parent / "f1_data.jsonl"
    if not real.exists():
        return
    rag = F1Rag(real)
    r = rag.index.search("cuantas carreras gano Checo", top_k=5)
    assert r and "Pérez" in r[0][0].text
    r = rag.index.search("como le fue a Kimi en 2021", top_k=1)
    assert r and "Räikkönen" in r[0][0].text
    # Con 2026 el mismo "Kimi" es Antonelli: sin forzar piloto sin contexto.
    r = rag.index.search("como le fue a Kimi en 2026", top_k=5)
    ids = [d.id for d, _ in r]
    assert any("antonelli" in i for i in ids), ids
    assert not any("raikkonen" in i for i in ids), ids




def test_actualidad_sin_ano_usa_la_temporada_mas_reciente():
    # "mas reciente" sin ano se resuelve con la ultima temporada del corpus.
    from rag import BM25Index

    docs = [Document(id="c2023", text="Max Verstappen se proclamo campeon del mundo de 2023.",
                     metadata={"type": "champion", "season": 2023}),
            Document(id="c2024", text="Max Verstappen se proclamo campeon del mundo de 2024.",
                     metadata={"type": "champion", "season": 2024})]
    index = BM25Index(docs)
    results = index.search("quien es el campeon mas reciente", top_k=1)
    assert results and results[0][0].id == "c2024"


def test_ultima_carrera_es_la_de_fecha_mas_reciente():
    # "la ultima carrera" da la carrera de fecha mas reciente, no cualquiera.
    from rag import BM25Index

    docs = [Document(id="r1", text="El Bahrain Grand Prix de 2026 se disputo el 2026-03-08. Gano Oscar Piastri.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="r2", text="El Mexico City Grand Prix de 2026 se disputo el 2026-11-01. Gano Max Verstappen.",
                     metadata={"type": "race", "season": 2026})]
    index = BM25Index(docs)
    results = index.search("quien gano la ultima carrera", top_k=1)
    assert results and results[0][0].id == "r2"


def test_ultima_carrera_con_ano_explicito_no_salta_de_temporada():
    # Con ano explicito, "la ultima carrera" se queda dentro de ese ano.
    from rag import BM25Index

    docs = [Document(id="r2025", text="El Abu Dhabi Grand Prix de 2025 se disputo el 2025-12-07. Gano Lando Norris.",
                     metadata={"type": "race", "season": 2025}),
            Document(id="r2026", text="El Mexico City Grand Prix de 2026 se disputo el 2026-11-01. Gano Max Verstappen.",
                     metadata={"type": "race", "season": 2026})]
    index = BM25Index(docs)
    results = index.search("quien gano la ultima carrera de 2025", top_k=1)
    assert results and results[0][0].id == "r2025"


def test_actualidad_en_datos_reales():
    real = DATA.parent / "f1_data.jsonl"
    if not real.exists():
        return
    rag = F1Rag(real)
    r = rag.index.search("quien es el campeon mas reciente", top_k=1)
    assert r and r[0][0].id == "champion-2026-drivers", r[0][0].id
    r = rag.index.search("quien gano la ultima carrera", top_k=1)
    import json as _json
    import re as _re
    esperada = max(
        (_json.loads(l) for l in open(real, encoding="utf-8") if l.strip()
         if _json.loads(l)["metadata"].get("type") == "race"),
        key=lambda d: _re.search(r"(\d{4}-\d{2}-\d{2})", d["text"]).group(0))
    assert r and r[0][0].id == esperada["id"], (r[0][0].id, esperada["id"])
    r = rag.index.search("quien gano la ultima carrera de 2025", top_k=1)
    assert r and r[0][0].id == "race-2025-r24", r[0][0].id




def test_nota_de_cobertura_con_actualidad():
    # Las respuestas de actualidad declaran hasta donde llegan los datos.
    from rag import BM25Index, corpus_coverage

    docs = [Document(id="r1", text="El Bahrain Grand Prix de la temporada 2026 se disputo el 2026-03-08. Gano Oscar Piastri.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="r2", text="El Mexico City Grand Prix de la temporada 2026 se disputo el 2026-11-01. Gano Max Verstappen.",
                     metadata={"type": "race", "season": 2026})]
    import rag as rag_mod
    r = F1Rag.__new__(F1Rag)
    r.documents = docs
    r.index = BM25Index(docs)
    r.coverage = corpus_coverage(docs)
    a = r.ask("quien gano la ultima carrera")
    assert "Datos indexados hasta el Mexico City Grand Prix, 2026-11-01." in a["answer"]


def test_nota_de_cobertura_solo_con_actualidad():
    # Sin terminos de actualidad no se anade la nota.
    from rag import BM25Index, corpus_coverage

    docs = [Document(id="c1", text="Max Verstappen se proclamo campeon del mundo de 2024.",
                     metadata={"type": "champion", "season": 2024}),
            Document(id="r1", text="El Bahrain Grand Prix de la temporada 2026 se disputo el 2026-03-08. Gano Oscar Piastri.",
                     metadata={"type": "race", "season": 2026})]
    r = F1Rag.__new__(F1Rag)
    r.documents = docs
    r.index = BM25Index(docs)
    r.coverage = corpus_coverage(docs)
    a = r.ask("quien es el campeon de 2024")
    assert "Datos indexados" not in a["answer"]


def test_cobertura_datos_reales_no_desactualizada():
    # Regresion de foto antigua: la cobertura declarada tiene que ser la
    # ultima carrera QUE HAY en el corpus, calculada leyendo los datos.
    # Si alguien refresca datos y rompe el calculo (o lo fija a mano),
    # este test salta.
    from rag import corpus_coverage
    import json as _json
    import re as _re

    real = DATA.parent / "f1_data.jsonl"
    if not real.exists():
        return
    fechas = []
    for l in open(real, encoding="utf-8"):
        if not l.strip():
            continue
        d = _json.loads(l)
        if d["metadata"].get("type") == "race":
            fechas.append(_re.search(r"(\d{4}-\d{2}-\d{2})", d["text"]).group(0))
    rag = F1Rag(real)
    assert rag.coverage["fecha"] == max(fechas), (rag.coverage, max(fechas))
    a = rag.ask("quien gano la ultima carrera")
    assert rag.coverage["fecha"] in a["answer"]
    assert rag.coverage["ultima_carrera"] in a["answer"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK {t.__name__}")
    print(f"\n{len(tests)} pruebas superadas")
