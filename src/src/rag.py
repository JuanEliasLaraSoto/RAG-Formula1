"""Núcleo del RAG: recuperación TF-IDF + generación opcional con LLM.

Sin dependencias externas. Si hay OPENAI_API_KEY, las respuestas se
generan con un LLM usando solo el contexto recuperado; si no, se
responde de forma extractiva con los fragmentos más relevantes.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# Stopwords en español e inglés (los datos de F1 mezclan ambos).
STOPWORDS = set("""
el la los las un una unos unas de del al a en con por para sin sobre entre
y o u ni que se su sus es son fue fueron ser estar haber este esta estos estas
como mas muy ya lo le les me te nos os mi tu yo el ella ellos ellas
the a an and or of to in on for with by is are was were be been at from as it its
""".split())


def normalize(text: str) -> str:
    """Minúsculas, sin tildes, solo caracteres alfanuméricos."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text)


def stem(token: str) -> str:
    """Despluralizacion ligera para espanol/ingles: une singulares y plurales."""
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in normalize(text).split() if t and t not in STOPWORDS]


@dataclass
class Document:
    """Un fragmento de conocimiento indexable."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


class BM25Index:
    """Indice BM25 en memoria, en Python puro.

    BM25 mejora al TF-IDF clasico: satura la frecuencia de terminos y
    normaliza la longitud de los documentos de forma suave, por lo que
    un documento corto no gana solo por ser corto.
    """

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_tokens = [tokenize(d.text) for d in documents]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) or 1.0
        self.n_docs = len(documents)
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            self.tf.append(counts)
            for tok in counts:
                self.df[tok] = self.df.get(tok, 0) + 1

    def _idf(self, token: str) -> float:
        # IDF de Robertson-Spärck Jones, acotado a valores positivos.
        df = self.df.get(token, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        q_terms = set(tokenize(query))
        if not q_terms:
            return []
        scored = []
        for doc, tf, dl in zip(self.documents, self.tf, self.doc_len):
            score = 0.0
            matched = 0
            for tok in q_terms:
                freq = tf.get(tok, 0)
                if not freq:
                    continue
                matched += 1
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self._idf(tok) * (freq * (self.k1 + 1)) / denom
            if score > 0:
                # Bonus de cobertura: premia documentos que cubren mas
                # terminos distintos de la consulta.
                coverage = matched / len(q_terms)
                scored.append((doc, score * math.sqrt(coverage)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# Alias por compatibilidad: antes existia TfidfIndex.
TfidfIndex = BM25Index


def load_documents(path: str | Path) -> list[Document]:
    """Carga documentos desde un JSONL: {"id": ..., "text": ..., "metadata": {...}}"""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                docs.append(Document(id=row["id"], text=row["text"],
                                     metadata=row.get("metadata", {})))
    return docs


SYSTEM_PROMPT = (
    "Eres un asistente experto en Formula 1. Responde SOLO con la informacion "
    "del contexto proporcionado. Si el contexto no contiene la respuesta, di "
    "claramente que no tienes ese dato. Responde en espanol, de forma concisa."
)


def build_prompt(question: str, results: list[tuple[Document, float]]) -> str:
    context = "\n\n".join(
        f"[Fragmento {i}] {doc.text}" for i, (doc, _) in enumerate(results, 1)
    )
    return (
        f"Contexto recuperado:\n{context}\n\n"
        f"Pregunta: {question}\n\n"
        "Respuesta (basada unicamente en el contexto):"
    )


def generate_with_llm(prompt: str) -> str:
    """Genera la respuesta con OpenAI. Requiere OPENAI_API_KEY."""
    from openai import OpenAI  # import perezoso: dependencia opcional

    client = OpenAI()
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def extractive_answer(results: list[tuple[Document, float]]) -> str:
    """Respuesta sin LLM: devuelve los fragmentos recuperados."""
    lines = ["No tengo un LLM configurado, pero esto es lo mas relevante que encontre:\n"]
    for i, (doc, score) in enumerate(results, 1):
        lines.append(f"{i}. {doc.text}")
    lines.append("\n(Configura OPENAI_API_KEY para respuestas redactadas.)")
    return "\n".join(lines)


class F1Rag:
    """Pipeline completo: carga documentos, indexa y responde."""

    def __init__(self, data_path: str | Path):
        self.documents = load_documents(data_path)
        if not self.documents:
            raise ValueError(f"Sin documentos en {data_path}")
        self.index = TfidfIndex(self.documents)

    def ask(self, question: str, top_k: int = 5) -> dict:
        results = self.index.search(question, top_k=top_k)
        if not results:
            return {
                "answer": "No encontre nada relevante en la base de conocimiento.",
                "sources": [],
            }
        prompt = build_prompt(question, results)
        if os.environ.get("OPENAI_API_KEY"):
            answer = generate_with_llm(prompt)
        else:
            answer = extractive_answer(results)
        return {
            "answer": answer,
            "sources": [
                {"id": doc.id, "score": round(score, 4),
                 "text": doc.text, "metadata": doc.metadata}
                for doc, score in results
            ],
        }
