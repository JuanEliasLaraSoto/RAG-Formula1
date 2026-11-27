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




def test_recuento_de_victorias_piloto():
    # "cuantas carreras gano X en <ano>" responde con el numero, calculado.
    from rag import count_wins

    docs = [Document(id="r1", text="El Bahrain Grand Prix de la temporada 2026 se disputo el 2026-03-08. Gano Max Verstappen con Red Bull en un tiempo de 1:30:00.000.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="r2", text="El Mexico City Grand Prix de la temporada 2026 se disputo el 2026-11-01. Gano Max Verstappen con Red Bull en un tiempo de 1:40:00.000.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="r3", text="El Monaco Grand Prix de la temporada 2026 se disputo el 2026-05-24. Gano Lando Norris con McLaren en un tiempo de 1:35:00.000.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="d1", text="Max Verstappen es un piloto de Formula 1 (Dutch).",
                     metadata={"type": "driver", "season": 2026, "name": "Max Verstappen"}),
            Document(id="d2", text="Lando Norris es un piloto de Formula 1 (British).",
                     metadata={"type": "driver", "season": 2026, "name": "Lando Norris"})]
    r = count_wins("cuantas carreras gano Verstappen en 2026", docs)
    assert r and "Max Verstappen gano 2 carreras en 2026" in r, r
    r = count_wins("cuantas victorias tiene Norris en 2026", docs)
    assert r and "Lando Norris gano 1 carrera en 2026" in r, r
    r = count_wins("cuantas carreras gano Alonso en 2026", docs)
    assert r is None  # Alonso no esta en este corpus de prueba


def test_recuento_de_victorias_equipo():
    from rag import count_wins

    docs = [Document(id="r1", text="El Bahrain Grand Prix de la temporada 2025 se disputo el 2025-03-16. Gano Lando Norris con McLaren en un tiempo de 1:30:00.000.",
                     metadata={"type": "race", "season": 2025}),
            Document(id="r2", text="El Monaco Grand Prix de la temporada 2025 se disputo el 2025-05-25. Gano Oscar Piastri con McLaren en un tiempo de 1:35:00.000.",
                     metadata={"type": "race", "season": 2025}),
            Document(id="t1", text="McLaren es un equipo (constructor) de Formula 1.",
                     metadata={"type": "constructor", "name": "McLaren"})]
    r = count_wins("cuantas carreras gano McLaren en 2025", docs)
    assert r and "McLaren gano 2 carreras en 2025" in r, r


def test_recuento_sin_ano_abarca_el_corpus_y_sin_forzar_apodos_ambiguos():
    from rag import count_wins

    docs = [Document(id="r1", text="El Bahrain Grand Prix de la temporada 2025 se disputo el 2025-03-16. Gano Kimi Raikkonen con Alfa Romeo en un tiempo de 1:30:00.000.",
                     metadata={"type": "race", "season": 2025}),
            Document(id="r2", text="El Monza Grand Prix de la temporada 2026 se disputo el 2026-09-06. Gano Andrea Kimi Antonelli con Mercedes en un tiempo de 1:20:00.000.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="d1", text="Kimi Raikkonen es un piloto.", metadata={"type": "driver", "name": "Kimi Raikkonen"}),
            Document(id="d2", text="Andrea Kimi Antonelli es un piloto.", metadata={"type": "driver", "name": "Andrea Kimi Antonelli"})]
    # "Kimi" solo es ambiguo: sin apellido ni contexto no se fuerza.
    assert count_wins("cuantas carreras gano Kimi", docs) is None
    # Con el apellido si.
    r = count_wins("cuantas carreras gano Antonelli", docs)
    assert r and "Andrea Kimi Antonelli gano 1 carrera" in r, r
    # Sin ano, el alcance dice las temporadas del corpus.
    r = count_wins("cuantas carreras gano Raikkonen", docs)
    assert r and "2025-2026" in r, r


def _victorias_en_corpus(docs, quien, temporada, equipo=False):
    # Cuenta las victorias leyendo el corpus, igual que count_wins: sirve
    # para que los tests sobre datos reales calculen lo esperado de los
    # datos y no lo fijen a mano (si se refresca 2026, el test se mueve
    # con los datos en vez de quedarse en una foto antigua).
    import re as _re

    n = 0
    for d in docs:
        if d.metadata.get("type") != "race" or str(d.metadata.get("season")) != str(temporada):
            continue
        m = _re.search(r"Gano (.+?) con (.+?) en un tiempo", d.text)
        if not m:
            continue
        if equipo:
            qt, et = set(quien.lower().split()), set(m.group(2).lower().split())
            hit = qt <= et or et <= qt
        else:
            hit = m.group(1) == quien
        n += 1 if hit else 0
    return n


def test_recuento_en_datos_reales():
    real = DATA.parent / "f1_data.jsonl"
    if not real.exists():
        return
    from rag import load_documents

    docs = load_documents(real)
    rag = F1Rag(real)
    esp = _victorias_en_corpus(docs, "Max Verstappen", 2026)
    a = rag.ask("cuantas carreras gano Verstappen en 2026")
    assert f"Max Verstappen gano {esp} carrera" in a["answer"], a["answer"][:300]
    # 2026 es la ultima temporada del corpus: el recuento declara cobertura.
    assert "Datos indexados hasta el" in a["answer"], a["answer"][:300]
    esp = _victorias_en_corpus(docs, "McLaren", 2025, equipo=True)
    a = rag.ask("cuantas carreras gano McLaren en 2025")
    assert f"McLaren gano {esp} carreras en 2025" in a["answer"], a["answer"][:300]
    # Temporada sin carreras indexadas: ausencia de datos, no un cero.
    a = rag.ask("cuantas carreras gano Verstappen en 2027")
    assert "No tengo carreras indexadas de 2027" in a["answer"], a["answer"][:300]
    # Una pregunta que no es de recuento no lleva frase calculada.
    a = rag.ask("quien gano el mundial de pilotos 2024")
    assert "Segun los datos indexados" not in a["answer"]


def test_recuento_temporada_sin_datos_no_es_cero():
    # Pedir una temporada sin carreras indexadas no es un cero medido:
    # es ausencia de datos, y la respuesta lo dice junto con hasta donde
    # llega la base.
    from rag import count_wins

    docs = [Document(id="r1", text="El Bahrain Grand Prix de la temporada 2026 se disputo el 2026-03-08. Gano Max Verstappen con Red Bull en un tiempo de 1:30:00.000.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="d1", text="Max Verstappen es un piloto de Formula 1 (Dutch).",
                     metadata={"type": "driver", "season": 2026, "name": "Max Verstappen"})]
    r = count_wins("cuantas carreras gano Verstappen en 2027", docs)
    assert r is not None
    assert "No tengo carreras indexadas de 2027" in r, r
    assert "no gano ninguna" not in r, r
    assert "2026" in r, r


def test_recuento_ano_explicito_declara_cobertura():
    # Un recuento con ano explicito sobre la ultima temporada del corpus
    # (que puede ir incompleta) declara hasta que carrera llegan los
    # datos, aunque la pregunta no use palabras de actualidad. Una
    # temporada cerrada no lleva la nota.
    from rag import count_wins

    docs = [Document(id="r1", text="El Monaco Grand Prix de la temporada 2025 se disputo el 2025-05-25. Gano Lando Norris con McLaren en un tiempo de 1:35:00.000.",
                     metadata={"type": "race", "season": 2025}),
            Document(id="r2", text="El Mexico City Grand Prix de la temporada 2026 se disputo el 2026-11-01. Gano Max Verstappen con Red Bull en un tiempo de 1:40:00.000.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="d1", text="Max Verstappen es un piloto.",
                     metadata={"type": "driver", "name": "Max Verstappen"}),
            Document(id="d2", text="Lando Norris es un piloto.",
                     metadata={"type": "driver", "name": "Lando Norris"})]
    r = count_wins("cuantas carreras gano Verstappen en 2026", docs)
    assert "Max Verstappen gano 1 carrera en 2026" in r, r
    assert "Datos indexados hasta el Mexico City Grand Prix, 2026-11-01." in r, r
    r = count_wins("cuantas carreras gano Norris en 2025", docs)
    assert "Lando Norris gano 1 carrera en 2025" in r, r
    assert "Datos indexados" not in r, r


def test_recuento_cero_real_en_temporada_cubierta():
    # Un piloto sin victorias en una temporada SI cubierta es un cero
    # medido, no ausencia de datos: la frase lo afirma sin ambiguedad.
    from rag import count_wins

    docs = [Document(id="r1", text="El Bahrain Grand Prix de la temporada 2026 se disputo el 2026-03-08. Gano Max Verstappen con Red Bull en un tiempo de 1:30:00.000.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="d1", text="Max Verstappen es un piloto.",
                     metadata={"type": "driver", "name": "Max Verstappen"}),
            Document(id="d2", text="Fernando Alonso es un piloto.",
                     metadata={"type": "driver", "name": "Fernando Alonso"})]
    r = count_wins("cuantas carreras gano Alonso en 2026", docs)
    assert r and "Fernando Alonso no gano ninguna carrera en 2026" in r, r
    assert "No tengo carreras indexadas" not in r, r


def test_recuento_con_actualidad_no_duplica_la_nota():
    # Recuento + palabras de actualidad: la nota de cobertura sale una
    # sola vez (la pone el recuento y ask() no la repite).
    from rag import BM25Index, corpus_coverage

    docs = [Document(id="r1", text="El Mexico City Grand Prix de la temporada 2026 se disputo el 2026-11-01. Gano Max Verstappen con Red Bull en un tiempo de 1:40:00.000.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="d1", text="Max Verstappen es un piloto.",
                     metadata={"type": "driver", "name": "Max Verstappen"})]
    r = F1Rag.__new__(F1Rag)
    r.documents = docs
    r.index = BM25Index(docs)
    r.coverage = corpus_coverage(docs)
    a = r.ask("cuantas carreras gano Verstappen en la ultima temporada")
    assert a["answer"].count("Datos indexados hasta el") == 1, a["answer"]


def test_puntos_y_posicion_piloto():
    # "cuantos puntos tiene X" / "en que posicion va X" responde con la
    # clasificacion calculada de los documentos standing.
    from rag import standing_lookup

    docs = [Document(id="s1", text="En el campeonato de pilotos 2026, Lando Norris ocupa la posicion 5 con 264 puntos y 3 victorias tras la carrera 21 de 23 (Las Vegas Grand Prix, 2026-11-22), a fecha de 2026-11-25. La temporada aun esta en curso.",
                     metadata={"type": "standing", "season": 2026}),
            Document(id="r1", text="El Las Vegas Grand Prix de la temporada 2026 se disputo el 2026-11-22. Gano Max Verstappen con Red Bull en un tiempo de 1:27:39.624.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="d1", text="Lando Norris es un piloto.",
                     metadata={"type": "driver", "name": "Lando Norris"}),
            Document(id="d2", text="Max Verstappen es un piloto.",
                     metadata={"type": "driver", "name": "Max Verstappen"})]
    r = standing_lookup("cuantos puntos tiene Norris en 2026", docs)
    assert r and "Lando Norris ocupa la posicion 5" in r and "264 puntos" in r, r
    # Sin ano se usa la temporada mas reciente de los datos, con cobertura.
    r = standing_lookup("en que posicion va Norris", docs)
    assert r and "posicion 5" in r and "2026" in r, r
    assert "Datos indexados hasta el Las Vegas Grand Prix, 2026-11-22." in r, r
    # Piloto sin fila en la clasificacion: ausencia en los datos, no cero.
    r = standing_lookup("cuantos puntos tiene Verstappen en 2026", docs)
    assert r and "no aparece en la clasificacion" in r, r


def test_puntos_temporada_sin_datos_y_temporada_cerrada():
    # Una temporada sin clasificacion indexada es ausencia de datos; una
    # temporada cerrada responde en pasado y sin nota de cobertura.
    from rag import standing_lookup

    docs = [Document(id="s1", text="En el campeonato de pilotos 2025, Lando Norris termino en posicion 1 con 423 puntos y 7 victorias.",
                     metadata={"type": "standing", "season": 2025}),
            Document(id="s2", text="En el campeonato de pilotos 2026, Max Verstappen ocupa la posicion 1 con 400 puntos y 9 victorias tras la carrera 21 de 23 (Las Vegas Grand Prix, 2026-11-22), a fecha de 2026-11-25. La temporada aun esta en curso.",
                     metadata={"type": "standing", "season": 2026}),
            Document(id="r1", text="El Abu Dhabi Grand Prix de la temporada 2025 se disputo el 2025-12-07. Gano Lando Norris con McLaren en un tiempo de 1:25:00.000.",
                     metadata={"type": "race", "season": 2025}),
            Document(id="r2", text="El Las Vegas Grand Prix de la temporada 2026 se disputo el 2026-11-22. Gano Max Verstappen con Red Bull en un tiempo de 1:27:39.624.",
                     metadata={"type": "race", "season": 2026}),
            Document(id="d1", text="Lando Norris es un piloto.",
                     metadata={"type": "driver", "name": "Lando Norris"}),
            Document(id="d2", text="Max Verstappen es un piloto.",
                     metadata={"type": "driver", "name": "Max Verstappen"})]
    r = standing_lookup("cuantos puntos tiene Norris en 2027", docs)
    assert r and "No tengo clasificacion indexada de 2027" in r, r
    assert "2026" in r, r  # dice hasta donde llega la base
    r = standing_lookup("en que posicion quedo Norris en 2025", docs)
    assert r and "termino en la posicion 1" in r, r
    assert "Datos indexados" not in r, r  # temporada cerrada: sin nota


def test_puntos_apodos_y_ambiguos():
    # Los apodos valen ("Checo") y los ambiguos no se fuerzan ("Kimi").
    from rag import standing_lookup

    docs = [Document(id="s1", text="En el campeonato de pilotos 2023, Sergio Perez termino en posicion 2 con 285 puntos y 2 victorias.",
                     metadata={"type": "standing", "season": 2023}),
            Document(id="r1", text="El Abu Dhabi Grand Prix de la temporada 2023 se disputo el 2023-11-26. Gano Max Verstappen con Red Bull en un tiempo de 1:27:00.000.",
                     metadata={"type": "race", "season": 2023}),
            Document(id="d1", text="Sergio Perez es un piloto.", metadata={"type": "driver", "name": "Sergio Perez"}),
            Document(id="d2", text="Kimi Raikkonen es un piloto.", metadata={"type": "driver", "name": "Kimi Raikkonen"}),
            Document(id="d3", text="Andrea Kimi Antonelli es un piloto.", metadata={"type": "driver", "name": "Andrea Kimi Antonelli"})]
    r = standing_lookup("cuantos puntos tiene Checo en 2023", docs)
    assert r and "Sergio Perez termino en la posicion 2" in r and "285 puntos" in r, r
    assert standing_lookup("cuantos puntos tiene Kimi en 2023", docs) is None
    # Una consulta sin piloto claro no se calcula.
    assert standing_lookup("cuantos puntos tiene el equipo", docs) is None


def test_puntos_en_datos_reales():
    real = DATA.parent / "f1_data.jsonl"
    if not real.exists():
        return
    import re as _re

    from rag import load_documents
    docs = load_documents(real)
    # Esperado calculado del corpus: la fila de Norris de la ultima temporada.
    temporadas = sorted(str(d.metadata.get("season")) for d in docs
                        if d.metadata.get("type") == "standing")
    ultima = temporadas[-1]
    esperado = None
    for d in docs:
        if d.metadata.get("type") == "standing" and str(d.metadata.get("season")) == ultima:
            m = _re.search(r", Lando Norris (ocupa la posicion|termino en posicion) (\d+) con ([\d.]+) puntos", d.text)
            if m:
                esperado = m
                break
    assert esperado is not None, "falta la fila de Norris en el corpus"
    rag = F1Rag(real)
    a = rag.ask("cuantos puntos tiene Norris")
    assert f"posicion {esperado.group(2)}" in a["answer"], a["answer"][:300]
    assert f"{esperado.group(3)} puntos" in a["answer"], a["answer"][:300]
    # La ultima temporada declara cobertura una sola vez.
    assert a["answer"].count("Datos indexados hasta el") == 1, a["answer"][:300]
    # Y el recuento sigue funcionando igual (no hay confusion de intentos).
    a = rag.ask("cuantas carreras gano Norris en 2025")
    assert "puntos" not in a["answer"].split("\n")[0], a["answer"][:300]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK {t.__name__}")
    print(f"\n{len(tests)} pruebas superadas")
