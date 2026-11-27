"""Núcleo del RAG: recuperación (BM25, embeddings o híbrida) + generación opcional con LLM.

Sin dependencias externas. Si hay OPENAI_API_KEY, las respuestas se
generan con un LLM usando solo el contexto recuperado; si no, se
responde de forma extractiva con los fragmentos más relevantes.
"""

from __future__ import annotations

import hashlib
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
quien quienes cual cuales cuanto cuanta cuantos cuantas donde cuando cuyo cuya
he has ha han hemos habeis
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


# Sinonimos del dominio de F1, sobre tokens ya normalizados y con stem.
# Se aplican SOLO a la consulta (los documentos se indexan tal cual): un
# sinonimo encontrado suma con peso reducido, asi matiza sin sustituir al
# termino exacto.
_RAW_SYNONYMS = {
    "mundial": ["campeonato", "titulo", "mundo"],
    "campeonato": ["mundial", "titulo"],
    "titulo": ["campeonato", "mundial"],
    "escuderia": ["equipo", "constructor"],
    "equipo": ["escuderia", "constructor"],
    "constructor": ["equipo", "escuderia"],
    "gp": ["grand", "prix"],
    "ganar": ["victoria", "ganador", "triunfo", "gano"],
    "gano": ["victoria", "ganador", "triunfo", "ganar"],
    "victoria": ["ganar", "gano", "ganador", "triunfo"],
    "ganador": ["ganar", "gano", "victoria", "vencedor"],
    "ganado": ["victoria", "triunfo", "gano", "ganar", "ganador"],
}
SYNONYMS = {stem(k): [stem(v) for v in vals] for k, vals in _RAW_SYNONYMS.items()}
SYNONYM_WEIGHT = 0.6

# Apodos de pilotos: tokens de consulta que no aparecen en los documentos
# pero identifican a un piloto concreto ("Checo" es Sergio Perez). A
# diferencia de un sinonimo, el apodo REFIERE al piloto, asi que se expande
# a los tokens de su nombre completo con peso casi pleno. Solo entran
# apodos sin ambiguedad: "Kimi" no vale, porque en el corpus hay dos
# (Kimi Raikkonen y Andrea Kimi Antonelli) y el nombre ya aparece en sus
# fichas, asi que la temporada de la consulta los desempata sola.
_RAW_ALIASES = {
    "checo": "sergio perez",
    "nano": "fernando alonso",
    "magic": "fernando alonso",
    "mad": "max verstappen",
    "supermax": "max verstappen",
    "seb": "sebastian vettel",
    "schumi": "michael schumacher",
}
ALIASES = {stem(k): [stem(t) for t in v.split()] for k, v in _RAW_ALIASES.items()}
ALIAS_WEIGHT = 0.9

# Ajuste por temporada: si la consulta menciona un ano, los documentos con
# metadata de esa temporada puntuan mas y los de otras temporadas, menos.
SEASON_BOOST = 1.5
SEASON_DAMP = 0.7
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Actualidad: "ultimo", "reciente", "actual"... sin ano en la consulta se
# interpretan como la temporada mas reciente del corpus, y entre carreras
# gana la de fecha mas reciente dentro del alcance temporal activo.
RECENCY_TERMS = {"ultimo", "ultima", "reciente", "actual", "vigente"}
RACE_RECENCY_BOOST = 4.0
_RACE_DATE_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")


def extract_years(text: str) -> set[str]:
    """Anos (19xx/20xx) mencionados en un texto, como strings."""
    return set(_YEAR_RE.findall(normalize(text)))



def diversify_groups(scored: list, top_k: int, max_per_group: int = 2) -> list:
    """Variedad en el top-k: como mucho `max_per_group` documentos del mismo
    grupo (tipo de documento + temporada de su metadata).

    La base tiene series de documentos casi identicos (p. ej. una ficha por
    piloto y temporada): sin limite, una consulta amplia llenaba el top-k
    con la misma plantilla repetida y el LLM recibia contexto redundante.
    Con el limite, el ranking se recorre en orden y, una vez lleno el cupo
    de un grupo, se pasa al mejor documento de OTRO grupo; si no hay
    suficientes grupos distintos, se rellena con los descartados en su
    orden original. Los documentos sin tipo+temporada nunca se limitan.
    El primer puesto nunca cambia y el orden por relevancia se respeta
    dentro de cada grupo.
    """
    if not scored:
        return []

    def group(doc) -> object:
        t = doc.metadata.get("type")
        s = doc.metadata.get("season")
        return (t, s) if t is not None and s is not None else ("__unico__", doc.id)

    selected = []
    deferred = []
    counts: dict = {}
    for doc, s in scored:
        g = group(doc)
        if counts.get(g, 0) < max_per_group:
            selected.append((doc, s))
            counts[g] = counts.get(g, 0) + 1
        else:
            deferred.append((doc, s))
        if len(selected) == top_k:
            break
    if len(selected) < top_k:
        selected.extend(deferred[: top_k - len(selected)])
    return selected


def has_recency_intent(question: str) -> bool:
    """La consulta habla de actualidad ("ultimo", "reciente", "actual"...)."""
    return bool(set(tokenize(question)) & RECENCY_TERMS)


def corpus_coverage(documents: list) -> dict | None:
    """Cobertura temporal del corpus: la carrera mas reciente indexada y su
    fecha. Sirve para que las respuestas de actualidad digan hasta donde
    llegan los datos en vez de darlos por hoy."""
    races = []
    for d in documents:
        if d.metadata.get("type") == "race":
            m = _RACE_DATE_RE.search(d.text)
            if m:
                name = re.search(r"^[Ee]l (.+?) de la temporada", d.text)
                races.append((m.group(0), name.group(1) if name else d.id))
    if not races:
        return None
    fecha, nombre = max(races)
    return {"ultima_carrera": nombre, "fecha": fecha}


def _resolve_entity(question: str, documents: list) -> tuple:
    """Piloto o equipo unico al que se refiere la consulta.

    Los apodos valen ("Checo" es Sergio Perez); un apellido compartido por
    varios pilotos no se fuerza (misma regla que con Kimi: sin contexto no
    se elige). Devuelve (nombre, es_equipo) o (None, False) si no hay un
    referente claro. Lo usan las respuestas calculadas (count_wins,
    standing_lookup).
    """
    q_tokens = set(tokenize(question))
    # Los apodos valen tambien para calcular ("cuantas carreras gano Checo").
    expanded = set(q_tokens)
    for t in q_tokens:
        expanded.update(ALIASES.get(t, ()))

    drivers = {d.metadata["name"] for d in documents
               if d.metadata.get("type") == "driver" and d.metadata.get("name")}
    teams = {d.metadata["name"] for d in documents
             if d.metadata.get("type") in ("constructor", "team") and d.metadata.get("name")}

    def ntok(name: str) -> set[str]:
        return set(tokenize(name))

    def matches(names: set[str]) -> list[str]:
        # Coincide si algun token largo del nombre esta en la consulta
        # (p. ej. el apellido o el nombre del equipo).
        return sorted(n for n in names
                      if any(tok in expanded for tok in ntok(n) if len(tok) > 3))

    m_drivers = matches(drivers)
    m_teams = matches(teams)
    if len(m_drivers) == 1:
        return m_drivers[0], False
    if len(m_drivers) == 0 and len(m_teams) == 1:
        return m_teams[0], True
    if len(m_drivers) > 1:
        # El apellido desempata: "Kimi" toca a dos pilotos, "Antonelli" a uno.
        by_surname = [n for n in m_drivers if tokenize(n)[-1:] and tokenize(n)[-1] in expanded]
        if len(by_surname) == 1:
            return by_surname[0], False
    return None, False


_RACE_WIN_RE = re.compile(r"Gano (.+?) con (.+?) en un tiempo")


def count_wins(question: str, documents: list, coverage: dict | None = None) -> str | None:
    """Respuesta calculada a "cuantas carreras/victorias gano X [en <ano>]".

    Cuenta las carreras del corpus ganadas por ese piloto ("Gano <nombre>")
    o por ese equipo ("con <equipo>") dentro del alcance temporal de la
    consulta, y lo dice en una frase; el detalle lo dan los fragmentos.
    Devuelve None si la consulta no es de recuento o el piloto/equipo es
    ambiguo (misma regla que con Kimi: sin contexto no se fuerza).

    Una temporada pedida sin carreras indexadas es ausencia de datos, no
    un cero: la respuesta dice que faltan esos datos y hasta donde llega
    la base. Y cuando el recuento toca la ultima temporada del corpus,
    que puede ir incompleta, declara la cobertura (ultima carrera y fecha)
    aunque la pregunta no use palabras de actualidad.
    """
    if not re.search(r"\bcuant[ao]s?\b", normalize(question)):
        return None
    q_tokens = set(tokenize(question))
    if "carrera" not in q_tokens and "victoria" not in q_tokens:
        return None
    entity, is_team = _resolve_entity(question, documents)
    if entity is None:
        return None

    coverage = coverage if coverage is not None else corpus_coverage(documents)
    years = extract_years(question)
    recency = has_recency_intent(question)
    seasons = sorted({str(d.metadata.get("season")) for d in documents
                      if d.metadata.get("type") == "race"})
    if not years and recency and seasons:
        years = {seasons[-1]}
    # Temporadas pedidas sin carreras indexadas: ausencia de datos.
    cubiertas = years & set(seasons)
    ausentes = years - set(seasons)
    if years and not cubiertas:
        msg = (f"No tengo carreras indexadas de "
               f"{' y '.join(sorted(years))} en los datos")
        if seasons:
            msg += f": la base llega hasta la temporada {seasons[-1]}"
        if coverage:
            msg += (f" (ultima carrera indexada: "
                    f"{coverage['ultima_carrera']}, {coverage['fecha']})")
        return msg + "."
    years = cubiertas
    wins = []
    for d in documents:
        if d.metadata.get("type") != "race":
            continue
        if years and str(d.metadata.get("season", "")) not in years:
            continue
        m = _RACE_WIN_RE.search(d.text)
        if not m:
            continue
        winner, team = m.group(1), m.group(2)
        if is_team:
            tq, te = set(tokenize(team)), set(tokenize(entity))
            hit = tq <= te or te <= tq
        else:
            hit = winner == entity
        if hit:
            gp = re.search(r"^[Ee]l (.+?) de la temporada", d.text)
            fecha = _RACE_DATE_RE.search(d.text)
            wins.append((fecha.group(0) if fecha else "", gp.group(1) if gp else d.id))
    wins.sort()

    if years:
        scope = "en " + " y ".join(sorted(years))
    elif seasons:
        scope = f"en las temporadas {seasons[0]}-{seasons[-1]} de los datos"
    else:
        scope = "en los datos"
    n = len(wins)
    if n == 0:
        msg = f"Segun los datos indexados, {entity} no gano ninguna carrera {scope}."
    else:
        lista = ", ".join(gp for _, gp in wins[:8])
        sufijo = f": {lista}" if n <= 8 else ""
        msg = (f"Segun los datos indexados, {entity} gano {n} "
               f"carrera{'s' if n != 1 else ''} {scope}{sufijo}.")
    if ausentes:
        msg += f" (sin carreras indexadas de {' y '.join(sorted(ausentes))})"
    # Si el recuento toca la ultima temporada del corpus, esa temporada
    # puede ir incompleta: la respuesta declara hasta que carrera llega.
    if coverage and seasons and (not years or seasons[-1] in years):
        msg += (f"\n\n(Datos indexados hasta el "
                f"{coverage['ultima_carrera']}, {coverage['fecha']}.)")
    return msg


_STANDING_RE = re.compile(
    r"En el campeonato de pilotos (\d{4}), (.+?) "
    r"(ocupa la posicion|termino en posicion) (\d+) con ([\d.]+) puntos"
    r"(?: y (\d+) victorias)?")


def standing_lookup(question: str, documents: list,
                    coverage: dict | None = None) -> str | None:
    """Respuesta calculada a "cuantos puntos tiene X" / "en que posicion va X".

    Lee la clasificacion indexada (documentos tipo standing, uno por piloto
    y temporada) y responde con la posicion, los puntos y las victorias de
    ese piloto; el detalle lo dan los fragmentos. Devuelve None si la
    consulta no es de puntos/posicion o el piloto es ambiguo (misma regla
    que con Kimi: sin contexto no se fuerza); los equipos no tienen
    clasificacion indexada y caen al camino normal de busqueda.

    Sin ano se responde con la temporada mas reciente de los datos (una
    clasificacion es una foto por temporada, no un acumulado). Una
    temporada pedida sin clasificacion indexada es ausencia de datos, y
    cuando la consulta toca la ultima temporada del corpus, que puede ir
    incompleta, la respuesta declara la cobertura (ultima carrera y fecha)
    aunque no use palabras de actualidad.
    """
    q = normalize(question)
    q_tokens = set(tokenize(question))
    puntos = bool(re.search(r"\bcuant[ao]s?\b", q)) and "punto" in q_tokens
    posicion = "posicion" in q_tokens or "puesto" in q_tokens
    if not puntos and not posicion:
        return None
    entity, is_team = _resolve_entity(question, documents)
    if entity is None or is_team:
        return None

    coverage = coverage if coverage is not None else corpus_coverage(documents)
    seasons = sorted({str(d.metadata.get("season")) for d in documents
                      if d.metadata.get("type") == "standing"})
    if not seasons:
        return None
    years = extract_years(question)
    cubiertas = years & set(seasons)
    if years and not cubiertas:
        msg = (f"No tengo clasificacion indexada de "
               f"{' y '.join(sorted(years))} en los datos")
        msg += f": la base llega hasta la temporada {seasons[-1]}"
        if coverage:
            msg += (f" (ultima carrera indexada: "
                    f"{coverage['ultima_carrera']}, {coverage['fecha']})")
        return msg + "."
    # Sin ano, la clasificacion que interesa es la mas reciente de los
    # datos; con varios anos, la mas reciente de los pedidos.
    year = max(cubiertas) if cubiertas else seasons[-1]

    row = None
    for d in documents:
        if d.metadata.get("type") != "standing":
            continue
        if str(d.metadata.get("season", "")) != year:
            continue
        m = _STANDING_RE.search(d.text)
        if m and m.group(2) == entity:
            row = m
            break
    if row is None:
        # Piloto conocido sin fila en la clasificacion de esa temporada:
        # ausencia en los datos, no un cero de puntos.
        return (f"Segun los datos indexados, {entity} no aparece en la "
                f"clasificacion de pilotos de {year}.")

    en_curso = row.group(3).startswith("ocupa")
    pos, pts, wins = row.group(4), row.group(5), row.group(6)
    verbo = "ocupa" if en_curso else "termino en"
    msg = (f"Segun los datos indexados, {entity} {verbo} la posicion {pos} "
           f"del campeonato de pilotos de {year} con {pts} puntos")
    if wins and wins != "0":
        msg += f" y {wins} victorias"
    msg += "."
    if coverage and year == seasons[-1]:
        msg += (f"\n\n(Datos indexados hasta el "
                f"{coverage['ultima_carrera']}, {coverage['fecha']}.)")
    return msg


def diversify_enabled(explicit: bool | None) -> bool:
    """Diversificacion MMR activa por defecto; RAG_DIVERSIFY=off la quita."""
    if explicit is not None:
        return explicit
    setting = os.environ.get("RAG_DIVERSIFY", "on").strip().lower()
    return setting not in {"off", "no", "0", "false"}


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
        self.latest_season = max(
            (d.metadata.get("season") for d in documents
             if isinstance(d.metadata.get("season"), int)),
            default=None,
        )
        # Fecha de cada carrera (solo documentos tipo race), para "la ultima
        # carrera": entre las del alcance temporal activo gana la mas reciente.
        self.race_dates: list[str | None] = []
        for d in documents:
            if d.metadata.get("type") == "race":
                m = _RACE_DATE_RE.search(d.text)
                self.race_dates.append(m.group(0) if m else None)
            else:
                self.race_dates.append(None)
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

    def search(self, query: str, top_k: int = 5,
               diversify: bool | None = None) -> list[tuple[Document, float]]:
        q_terms = set(tokenize(query))
        if not q_terms:
            return []
        years = extract_years(query)
        recency = bool(q_terms & RECENCY_TERMS)
        if recency and not years and self.latest_season is not None:
            # "ultimo/reciente/actual" sin ano: la temporada mas reciente.
            years = {str(self.latest_season)}
        # Expansion de sinonimos de la consulta: el termino original pesa 1.0
        # y cada sinonimo un peso reducido.
        weighted = {t: 1.0 for t in q_terms}
        for term in q_terms:
            for syn in SYNONYMS.get(term, ()):
                weighted.setdefault(syn, SYNONYM_WEIGHT)
            for alias in ALIASES.get(term, ()):
                weighted.setdefault(alias, ALIAS_WEIGHT)
        scored = []
        for doc, tf, dl, rdate in zip(self.documents, self.tf, self.doc_len,
                                      self.race_dates):  # rdate solo en races
            score = 0.0
            covered: set[str] = set()
            for tok, weight in weighted.items():
                freq = tf.get(tok, 0)
                if not freq:
                    continue
                if tok in q_terms:
                    covered.add(tok)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += weight * self._idf(tok) * (freq * (self.k1 + 1)) / denom
            # Un apodo queda cubierto si el documento nombra al piloto:
            # "Checo" se satisface con un texto que dice "Sergio Perez".
            for term in q_terms:
                if term not in covered and any(tf.get(a, 0) for a in ALIASES.get(term, ())):
                    covered.add(term)
            # "carrera" tambien la cubre un documento que ES una carrera,
            # aunque su texto no use la palabra (las fichas dicen "Grand
            # Prix" y "se disputo", no "carrera").
            if "carrera" in q_terms and doc.metadata.get("type") == "race":
                covered.add("carrera")
            if score > 0:
                # Cobertura: un documento que no cubre todos los terminos
                # originales de la consulta se penaliza en proporcion, asi
                # un sinonimo suelto no adelanta a una coincidencia plena.
                coverage = len(covered) / len(q_terms)
                score *= coverage
                # Ajuste por temporada: con un ano en la consulta, los
                # documentos de esa temporada suben y los de otras bajan.
                if years:
                    season = str(doc.metadata.get("season", ""))
                    if season:
                        score *= SEASON_BOOST if season in years else SEASON_DAMP
                scored.append([doc, score, rdate])
        if recency:
            # La carrera mas reciente ENTRE las que encajan con la consulta:
            # "la ultima victoria de Hamilton" es su ultima carrera ganada,
            # no necesariamente la ultima del calendario.
            dates = [rdate for doc, s, rdate in scored if rdate and years
                     and str(doc.metadata.get("season", "")) in years]
            if dates:
                latest = max(dates)
                for entry in scored:
                    if entry[2] == latest:
                        entry[1] *= RACE_RECENCY_BOOST
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = [(doc, s) for doc, s, _ in scored]
        if diversify_enabled(diversify):
            return diversify_groups(scored, top_k)
        return scored[:top_k]


def cosine(a: list[float], b: list[float]) -> float:
    """Similitud de coseno entre dos vectores, en Python puro."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def openai_embed(texts: list[str]) -> list[list[float]]:
    """Embeddings con OpenAI (text-embedding-3-small por defecto).

    Requiere OPENAI_API_KEY y el paquete openai. Se usa un unico
    lote por llamada; para bases grandes convendria paginar.
    """
    from openai import OpenAI  # import perezoso: dependencia opcional

    client = OpenAI()
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


class EmbeddingIndex:
    """Indice por embeddings: cada documento es un vector y la busqueda
    es similitud de coseno. Captura sinonimos y parafrasis que BM25 no ve
    ("maximo ganador" ~ "piloto con mas titulos").

    La funcion de embedding es inyectable para poder probar sin red.

    Con `cache_path` los vectores se guardan en un JSON local: en la
    siguiente ejecucion solo se calculan los embeddings de los documentos
    nuevos o modificados, en vez de recalcular toda la base (ahorro de
    tiempo y de llamadas a la API). La clave de cada vector es el hash del
    modelo y del texto, asi que un documento editado se recalcula solo y
    los vectores de documentos borrados se purgan al guardar.
    """

    def __init__(self, documents: list[Document], embed_fn=None, cache_path: str | Path | None = None):
        self.documents = documents
        self.embed_fn = embed_fn or openai_embed
        self.model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        texts = [d.text for d in documents]
        keys = [self._cache_key(t) for t in texts]
        cache = self._load_cache(cache_path)
        self.vectors: list[list[float]] = [[] for _ in texts]
        missing = []
        for i, key in enumerate(keys):
            if key in cache:
                self.vectors[i] = cache[key]
            else:
                missing.append(i)
        if missing:
            nuevos = self.embed_fn([texts[i] for i in missing])
            for i, vec in zip(missing, nuevos):
                self.vectors[i] = vec
                cache[keys[i]] = vec
            self._save_cache(cache_path, cache, set(keys))

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{text}".encode("utf-8")).hexdigest()

    @staticmethod
    def _load_cache(cache_path: str | Path | None) -> dict:
        if not cache_path:
            return {}
        try:
            data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            return data.get("vectors", {}) if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _save_cache(cache_path: str | Path | None, cache: dict, valid_keys: set) -> None:
        if not cache_path:
            return
        cache = {k: v for k, v in cache.items() if k in valid_keys}
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"vectors": cache}), encoding="utf-8")
        tmp.replace(path)  # escritura atomica: nunca deja un cache a medias

    def search(self, query: str, top_k: int = 5,
               diversify: bool | None = None) -> list[tuple[Document, float]]:
        if not query.strip():
            return []
        q_vec = self.embed_fn([query])[0]
        scored = [(doc, cosine(q_vec, vec)) for doc, vec in zip(self.documents, self.vectors)]
        scored = [(doc, s) for doc, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        if diversify_enabled(diversify):
            return diversify_groups(scored, top_k)
        return scored[:top_k]


class HybridIndex:
    """Combina BM25 y embeddings con fusion de rango reciproco (RRF).

    BM25 acierta con terminos exactos ("GP de Monaco 2024") y los
    embeddings con sinonimos y parafrasis ("maximo ganador"); el hibrido
    junta ambos mundos. RRF suma 1/(k + rango) de cada lista, asi que no
    hace falta calibrar las escalas de puntuacion de cada motor: un
    documento bien posicionado en las dos listas queda el primero, y uno
    que solo ve un motor puede entrar si el otro no lo encontro.

    Toma los `candidatos` primeros de cada motor y devuelve el top_k
    fusionado. La puntuacion devuelta es la suma RRF (sin unidad).
    """

    def __init__(self, documents: list[Document], embed_fn=None,
                 cache_path: str | Path | None = None, rrf_k: int = 60,
                 candidatos: int = 10):
        self.bm25 = BM25Index(documents)
        self.embeddings = EmbeddingIndex(documents, embed_fn=embed_fn, cache_path=cache_path)
        self.rrf_k = rrf_k
        self.candidatos = candidatos

    def search(self, query: str, top_k: int = 5,
               diversify: bool | None = None) -> list[tuple[Document, float]]:
        if not tokenize(query):
            return []
        fusion: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for resultados in (self.bm25.search(query, top_k=self.candidatos, diversify=False),
                           self.embeddings.search(query, top_k=self.candidatos, diversify=False)):
            for rango, (doc, _score) in enumerate(resultados):
                fusion[doc.id] = fusion.get(doc.id, 0.0) + 1.0 / (self.rrf_k + rango + 1)
                docs[doc.id] = doc
        orden = sorted(fusion.items(), key=lambda x: x[1], reverse=True)
        fused = [(docs[doc_id], score) for doc_id, score in orden]
        if diversify_enabled(diversify):
            return diversify_groups(fused, top_k)
        return fused[:top_k]


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
    "claramente que no tienes ese dato. Responde en espanol, de forma concisa. "
    "Cuando uses un dato de un fragmento, citalo entre corchetes al final de la "
    "frase, p. ej. [Fragmento 2]."
)


def build_prompt(question: str, results: list[tuple[Document, float]]) -> str:
    # Cada fragmento lleva el id de su documento (fuente) para que la
    # respuesta redactada pueda citar de donde sale cada dato.
    context = "\n\n".join(
        f"[Fragmento {i}] (fuente: {doc.id}) {doc.text}"
        for i, (doc, _) in enumerate(results, 1)
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
    """Respuesta sin LLM: devuelve los fragmentos recuperados, con su fuente."""
    lines = ["No tengo un LLM configurado, pero esto es lo mas relevante que encontre:\n"]
    for i, (doc, score) in enumerate(results, 1):
        lines.append(f"{i}. {doc.text} (fuente: {doc.id})")
    lines.append("\n(Configura OPENAI_API_KEY para respuestas redactadas.)")
    return "\n".join(lines)


class F1Rag:
    """Pipeline completo: carga documentos, indexa y responde.

    El recuperador se elige con `retriever` o la variable RAG_RETRIEVER:
    "bm25" (por defecto, sin dependencias), "embeddings" o "hybrid"
    (estos dos requieren OPENAI_API_KEY y el paquete openai).
    """

    def __init__(self, data_path: str | Path, retriever: str | None = None, embed_fn=None):
        self.documents = load_documents(data_path)
        if not self.documents:
            raise ValueError(f"Sin documentos en {data_path}")
        self.coverage = corpus_coverage(self.documents)
        retriever = (retriever or os.environ.get("RAG_RETRIEVER", "bm25")).lower()
        if retriever == "embeddings":
            self.index = EmbeddingIndex(self.documents, embed_fn=embed_fn,
                                        cache_path=self._embedding_cache_path(data_path))
        elif retriever == "hybrid":
            self.index = HybridIndex(self.documents, embed_fn=embed_fn,
                                     cache_path=self._embedding_cache_path(data_path))
        elif retriever == "bm25":
            self.index = TfidfIndex(self.documents)
        else:
            raise ValueError(
                f"Recuperador desconocido: {retriever} "
                "(usa 'bm25', 'embeddings' u 'hybrid')"
            )

    @staticmethod
    def _embedding_cache_path(data_path: str | Path) -> Path | None:
        """Ruta del cache de embeddings para una base de datos.

        Por defecto, junto a los datos (data/f1_data.embeddings.json).
        RAG_EMBEDDING_CACHE permite otra ruta, o desactivarlo con "off".
        """
        setting = os.environ.get("RAG_EMBEDDING_CACHE", "auto").strip()
        if setting.lower() in {"off", "no", "0", "false"}:
            return None
        if setting and setting.lower() != "auto":
            return Path(setting)
        return Path(data_path).with_suffix(".embeddings.json")

    def ask(self, question: str, top_k: int = 5,
            diversify: bool | None = None) -> dict:
        results = self.index.search(question, top_k=top_k, diversify=diversify)
        if not results:
            return {
                "answer": "No encontre nada relevante en la base de conocimiento.",
                "sources": [],
            }
        counting = count_wins(question, self.documents, coverage=self.coverage)
        standing = standing_lookup(question, self.documents, coverage=self.coverage)
        calculado = counting or standing  # una consulta no dispara ambos
        prompt = build_prompt(question, results)
        if os.environ.get("OPENAI_API_KEY"):
            if calculado:
                prompt += f"\n\nDato calculado de los documentos (usalo tal cual): {calculado}"
            answer = generate_with_llm(prompt)
        else:
            answer = extractive_answer(results)
            if calculado:
                answer = calculado + "\n\n" + answer
        if self.coverage and has_recency_intent(question):
            # Las preguntas de actualidad se responden con los datos
            # indexados, que pueden ir por detras del calendario real:
            # la respuesta dice hasta que carrera llegan.
            nota = (f"(Datos indexados hasta el "
                    f"{self.coverage['ultima_carrera']}, "
                    f"{self.coverage['fecha']}.)")
            if nota not in answer:  # el recuento ya la pudo incluir
                answer += "\n\n" + nota
        return {
            "answer": answer,
            "sources": [
                {"id": doc.id, "score": round(score, 4),
                 "text": doc.text, "metadata": doc.metadata}
                for doc, score in results
            ],
        }
