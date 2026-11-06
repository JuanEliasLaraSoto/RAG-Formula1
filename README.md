# F1 RAG

Un RAG (Retrieval-Augmented Generation) de Formula 1 desde cero, en Python.
Hazle preguntas en lenguaje natural ("quien gano el mundial de 2024?",
"como funciona el DRS?") y responde basandose en su base de conocimiento.

## Como funciona

```
pregunta ──> recuperacion BM25 ──> fragmentos relevantes ──> respuesta
                                                              │
                                    con OPENAI_API_KEY: GPT redacta la respuesta
                                    sin clave: respuesta extractiva (sin LLM)
```

1. **Base de conocimiento** (`data/*.jsonl`): un documento por linea,
   con `id`, `text` y `metadata`. Incluye datos de ejemplo con records
   historicos, la temporada 2024, reglas y equipos.
2. **Recuperacion** (`src/rag.py`): indice BM25 con similitud de coseno,
   en Python puro, sin dependencias. Normaliza tildes y maneja espanol e ingles.
3. **Generacion** (`src/rag.py`): si defines `OPENAI_API_KEY`, un LLM redacta
   la respuesta usando SOLO el contexto recuperado (con instruccion de no
   inventar). Sin clave, devuelve los fragmentos mas relevantes.
4. **Datos reales** (`src/data_ingest.py`): descarga pilotos, equipos,
   carreras y clasificaciones de cualquier temporada desde la
   [API Jolpica](https://api.jolpi.ca/ergast/) (sucesora de Ergast, gratuita).

## Inicio rapido

```bash
git clone <este-repo>
cd f1-rag

# Funciona sin instalar nada (Python 3.10+):
python src/ask.py "quien es el piloto con mas victorias?"
python src/ask.py          # modo interactivo
```

## Con datos reales de una temporada

```bash
pip install requests
python src/data_ingest.py --season 2024 --include-sample --output data/f1_data.jsonl
python src/ask.py --data data/f1_data.jsonl "quien gano en Monaco?"
```

## Con generacion LLM (opcional)

```bash
pip install openai
cp .env.example .env   # edita y pega tu OPENAI_API_KEY
python src/ask.py "resume el palmarés de Fernando Alonso"
```

## Pruebas

```bash
python tests/test_rag.py
```

## Estructura

```
f1-rag/
├── data/
│   └── sample_data.jsonl    # base de conocimiento de ejemplo
├── src/
│   ├── rag.py               # indice BM25 + pipeline RAG
│   ├── ask.py               # CLI
│   └── data_ingest.py       # descarga de la API Jolpica
└── tests/
    └── test_rag.py
```

## Siguientes pasos (ideas para extenderlo)

- Sustituir BM25 por embeddings (`sentence-transformers`) + FAISS/Chroma.
- Indexar mas temporadas y noticias de carreras.
- Servirlo como API con FastAPI.
- Anadir citas con la fuente de cada dato en la respuesta.
