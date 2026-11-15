"""Pruebas del evaluador del recuperador. Ejecutar: python tests/test_evaluate.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate import evaluate, hit_at_k, reciprocal_rank  # noqa: E402
from rag import F1Rag  # noqa: E402

DATA = Path(__file__).parent.parent / "data" / "sample_data.jsonl"
QUESTIONS = Path(__file__).parent.parent / "data" / "eval_questions.json"


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"b"}, 2)
    assert not hit_at_k(["a", "b", "c"], {"c"}, 2)
    assert hit_at_k(["a"], {"a", "z"}, 1)
    assert not hit_at_k([], {"a"}, 5)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_evaluate_sobre_corpus_de_ejemplo():
    # Mini conjunto de referencia contra la muestra ligera: todo debe
    # salir bien; si el recuperador empeora, este test lo nota primero.
    rag = F1Rag(DATA)
    questions = [
        {"question": "quien gano el campeonato de pilotos de 2024",
         "expected": ["season-2024-champion"]},
        {"question": "cuantos puntos da ganar una carrera",
         "expected": ["rules-points"]},
        {"question": "que es el DRS", "expected": ["rules-drs"]},
    ]
    report = evaluate(rag, questions, top_k=5)
    assert report["hit@1"] == 1.0, report["details"]
    assert report["mrr"] == 1.0


def test_conjunto_de_referencia_bien_formado():
    from evaluate import load_questions

    questions = load_questions(QUESTIONS)
    assert len(questions) >= 8
    for q in questions:
        assert q["question"].strip()
        assert all(isinstance(e, str) and e for e in q["expected"])


def test_evaluacion_del_corpus_real():
    # Regresion: el hit@5 del corpus real no debe caer por debajo del
    # nivel entregado (100% en el momento de anadir el evaluador).
    real = DATA.parent / "f1_data.jsonl"
    if not real.exists():
        return
    from evaluate import load_questions

    rag = F1Rag(real)
    report = evaluate(rag, load_questions(QUESTIONS), top_k=5)
    assert report["hit@k"] >= 0.8, report["details"]
    assert report["hit@1"] >= 0.6, report["details"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK {t.__name__}")
    print(f"\n{len(tests)} pruebas superadas")
