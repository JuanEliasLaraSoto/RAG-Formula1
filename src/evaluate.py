"""Evaluacion del recuperador: mide si las preguntas de referencia
devuelven el documento correcto.

Uso:
    python src/evaluate.py                      # corpus real, BM25
    python src/evaluate.py --retriever hybrid   # requiere OPENAI_API_KEY
    python src/evaluate.py --data data/sample_data.jsonl \
        --questions data/eval_questions_sample.json

Para cada pregunta del fichero de referencia (JSON con "question" y
"expected", la lista de ids de documentos que responden correctamente),
informa de si el documento esperado aparece en el top-1 y en el top-k, y
resume con hit@1, hit@k y MRR (mean reciprocal rank). Sirve para detectar
regresiones al tocar el recuperador: si una mejora baja estas metricas,
algo se ha roto.

Con --fail-under F el proceso sale con codigo 1 si el hit@1 queda por
debajo de F (util en CI).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag import F1Rag  # noqa: E402


def hit_at_k(ranked_ids: list[str], expected: set[str], k: int) -> bool:
    """True si alguno de los documentos esperados esta en los k primeros."""
    return bool(set(ranked_ids[:k]) & expected)


def reciprocal_rank(ranked_ids: list[str], expected: set[str]) -> float:
    """1/(posicion del primer acierto), o 0.0 si no hay ninguno."""
    for i, doc_id in enumerate(ranked_ids, 1):
        if doc_id in expected:
            return 1.0 / i
    return 0.0


def load_questions(path: str | Path) -> list[dict]:
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    for q in questions:
        if "question" not in q or not q.get("expected"):
            raise ValueError(f"Pregunta de evaluacion mal formada: {q}")
    return questions


def evaluate(rag: F1Rag, questions: list[dict], top_k: int = 5) -> dict:
    """Ejecuta las preguntas y devuelve las metricas y el detalle."""
    details = []
    for q in questions:
        expected = set(q["expected"])
        ranked = [doc.id for doc, _ in rag.index.search(q["question"], top_k=top_k)]
        details.append({
            "question": q["question"],
            "expected": sorted(expected),
            "ranked": ranked,
            "hit@1": hit_at_k(ranked, expected, 1),
            "hit@k": hit_at_k(ranked, expected, top_k),
            "rr": reciprocal_rank(ranked, expected),
        })
    n = len(details) or 1
    return {
        "n": len(details),
        "top_k": top_k,
        "hit@1": sum(d["hit@1"] for d in details) / n,
        "hit@k": sum(d["hit@k"] for d in details) / n,
        "mrr": sum(d["rr"] for d in details) / n,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua el recuperador del RAG.")
    parser.add_argument("--data", default="data/f1_data.jsonl")
    parser.add_argument("--questions", default="data/eval_questions.json")
    parser.add_argument("--retriever", default=None,
                        help="bm25 (defecto), embeddings o hybrid.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fail-under", type=float, default=None,
                        help="Sale con error si el hit@1 queda por debajo.")
    args = parser.parse_args()

    rag = F1Rag(args.data, retriever=args.retriever)
    questions = load_questions(args.questions)
    report = evaluate(rag, questions, top_k=args.top_k)

    for d in report["details"]:
        mark = "OK  " if d["hit@1"] else ("top5" if d["hit@k"] else "MISS")
        print(f"{mark}  {d['question']}  ->  {d['ranked'][:3]}")
    print(f"\n{report['n']} preguntas | hit@1: {report['hit@1']:.0%} | "
          f"hit@{report['top_k']}: {report['hit@k']:.0%} | MRR: {report['mrr']:.3f}")

    if args.fail_under is not None and report["hit@1"] < args.fail_under:
        print(f"hit@1 por debajo del umbral ({args.fail_under:.0%})")
        sys.exit(1)


if __name__ == "__main__":
    main()
