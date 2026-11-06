"""CLI para preguntar al RAG de Formula 1.

Uso:
    python src/ask.py "Quien gano el mundial de 2024?"
    python src/ask.py            # modo interactivo
    python src/ask.py --data data/f1_data.jsonl --top-k 3 "pregunta"
    python src/ask.py --retriever embeddings "pregunta"   # requiere OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Carga .env si existe (sin dependencia externa).
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, str(Path(__file__).parent))
from rag import F1Rag  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pregunta al RAG de F1.")
    parser.add_argument("question", nargs="?", help="Pregunta (si se omite, modo interactivo).")
    parser.add_argument("--data", default="data/f1_data.jsonl", help="JSONL con el conocimiento.")
    parser.add_argument("--retriever", choices=["bm25", "embeddings"], default=None,
                        help="Recuperador (por defecto RAG_RETRIEVER o bm25).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Salida JSON con fuentes.")
    args = parser.parse_args()

    data = Path(args.data)
    if not data.exists() and args.data == "data/f1_data.jsonl":
        data = Path("data/sample_data.jsonl")
    rag = F1Rag(data, retriever=args.retriever)

    def run(question: str) -> None:
        result = rag.ask(question, top_k=args.top_k)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result["answer"])

    if args.question:
        run(args.question)
        return

    print("RAG de Formula 1 listo. Escribe tu pregunta (o 'salir').")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"salir", "exit", "quit", ""}:
            break
        run(question)


if __name__ == "__main__":
    main()
