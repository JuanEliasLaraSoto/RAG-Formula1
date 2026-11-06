# F1 RAG

Un RAG (Retrieval-Augmented Generation) de Formula 1 desde cero, en Python.
Hazle preguntas en lenguaje natural ("quien gano el mundial de 2026?",
"como funciona el DRS?") y responde basandose en su base de conocimiento.

## Como funciona

```
pregunta ──> recuperacion (BM25 o embeddings) ──> fragmentos relevantes ──> respuesta
                                                                              │
                                                    con OPENAI_API_KEY: GPT redacta la respuesta
                                                    sin clave: respuesta extractiva (sin LLM)
```

1. **Base de conocimiento** (`data/*.jsonl`): un documento por linea,
   con `id`, `text` y `metadata`. `data/f1_data.jsonl` trae los datos reales
   de las temporadas 2021-2026 (pilotos, equipos, ganadores de cada carrera
   y clasificaciones) mas records historicos y reglas; `data/sample_data.jsonl`
   es una muestra ligera.
2. **Recuperacion** (`src/rag.py`): dos motores intercambiables.
   - **BM25** (por defecto): en Python puro, sin dependencias. Normaliza
     tildes y maneja espanol e ingles.
   - **Embeddings** (`RAG_RETRIEVER=embeddings`): vectores de OpenAI con
     similitud de coseno; captura sinonimos y parafrasis que BM25 no ve.
3. **Generacion** (`src/rag.py`): si defines `OPENAI_API_KEY`, un LLM redacta
   la respuesta usando SOLO el contexto recuperado (con instruccion de no
   inventar). Sin clave, devuelve los fragmentos mas relevantes.
4. **Datos reales** (`src/data_ingest.py`): descarga pilotos, equipos,
   carreras y clasificaciones de cualquier temporada desde la
   [API Jolpica](https://api.jolpi.ca/ergast/) (sucesora de Ergast, gratuita).
5. **API REST** (`src/api.py`): sirve el RAG con FastAPI (`POST /ask`,
   `GET /health`, Swagger en `/docs`).

## Inicio rapido

```bash
git clone https://github.com/JuanEliasLaraSoto/RAG-Formula1.git
cd RAG-Formula1

# Funciona sin instalar nada (Python 3.10+):
python src/ask.py "quien es el piloto con mas victorias?"
python src/ask.py "quien gano el mundial de 2026?"
python src/ask.py          # modo interactivo
```

## Con datos reales de otra temporada

```bash
pip install requests
python src/data_ingest.py --season 2010 --output data/f1_2010.jsonl
python src/ask.py --data data/f1_2010.jsonl "quien gano el campeonato?"
```

## Con generacion LLM y embeddings (opcional)

```bash
pip install openai
cp .env.example .env   # edita y pega tu OPENAI_API_KEY
python src/ask.py "resume el palmares de Fernando Alonso"
python src/ask.py --retriever embeddings "que piloto ha ganado mas titulos?"
```

## API REST (opcional)

```bash
pip install fastapi uvicorn
uvicorn src.api:app --reload
# POST http://127.0.0.1:8000/ask  {"question": "quien gano en Monaco 2024?"}
# Documentacion interactiva en http://127.0.0.1:8000/docs
```

## Pruebas

```bash
python tests/test_rag.py
```

## Estructura

```
RAG-Formula1/
├── data/
│   ├── f1_data.jsonl        # temporadas 2021-2026 + historia y reglas
│   └── sample_data.jsonl    # muestra ligera
├── src/
│   ├── rag.py               # indices BM25 y embeddings + pipeline RAG
│   ├── ask.py               # CLI
│   ├── api.py               # API REST con FastAPI
│   └── data_ingest.py       # descarga de la API Jolpica
└── tests/
    └── test_rag.py
```

## Siguientes pasos (ideas para extenderlo)

- Indexar noticias y cronicas de carreras, no solo resultados.
- Persistir los embeddings en una base vectorial (FAISS/Chroma) para no recalcularlos.
- Anadir citas con la fuente de cada dato en la respuesta redactada.
