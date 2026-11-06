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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK {t.__name__}")
    print(f"\n{len(tests)} pruebas superadas")
