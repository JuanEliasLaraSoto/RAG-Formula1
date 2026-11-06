"""Descarga datos reales de F1 desde la API Jolpica (sucesora de Ergast)
y los convierte en documentos JSONL listos para indexar.

Uso:
    python src/data_ingest.py --season 2024 --output data/f1_data.jsonl

Sin argumentos descarga la temporada actual. Combina los datos descargados
con el fichero de ejemplo data/sample_data.jsonl si existe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

API_BASE = "https://api.jolpi.ca/ergast/f1"


def fetch(endpoint: str) -> dict:
    import requests  # dependencia opcional

    url = f"{API_BASE}/{endpoint}.json?limit=100"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["MRData"]


def driver_documents(season: int) -> list[dict]:
    data = fetch(f"{season}/drivers")
    docs = []
    for d in data["DriverTable"]["Drivers"]:
        name = f"{d['givenName']} {d['familyName']}"
        text = (
            f"{name} es un piloto de Formula 1 ({d.get('nationality', 'nacionalidad desconocida')}), "
            f"numero {d.get('permanentNumber', 'sin numero fijo')}, "
            f"codigo {d.get('code', 'N/A')}, nacido el {d.get('dateOfBirth', 'fecha desconocida')}. "
            f"Compitio en la temporada {season}."
        )
        docs.append({"id": f"driver-{season}-{d['driverId']}", "text": text,
                     "metadata": {"type": "driver", "season": season, "name": name}})
    return docs


def constructor_documents(season: int) -> list[dict]:
    data = fetch(f"{season}/constructors")
    docs = []
    for c in data["ConstructorTable"]["Constructors"]:
        text = (
            f"{c['name']} es un equipo (constructor) de Formula 1 "
            f"de nacionalidad {c.get('nationality', 'desconocida')}, "
            f"participante en la temporada {season}."
        )
        docs.append({"id": f"constructor-{season}-{c['constructorId']}", "text": text,
                     "metadata": {"type": "constructor", "season": season, "name": c["name"]}})
    return docs


def race_documents(season: int) -> list[dict]:
    data = fetch(f"{season}/results/1")  # ganador de cada carrera
    docs = []
    for race in data["RaceTable"]["Races"]:
        winner = race["Results"][0]
        drv = winner["Driver"]
        con = winner["Constructor"]
        text = (
            f"El {race['raceName']} de la temporada {season} "
            f"(circuito {race['Circuit']['circuitName']}, {race['Circuit']['Location']['locality']}, "
            f"{race['Circuit']['Location']['country']}) se disputo el {race['date']}. "
            f"Gano {drv['givenName']} {drv['familyName']} con {con['name']} "
            f"en un tiempo de {winner.get('Time', {}).get('time', 'no registrado')}."
        )
        docs.append({"id": f"race-{season}-r{race['round']}", "text": text,
                     "metadata": {"type": "race", "season": season, "round": int(race["round"])}})
    return docs


def champion_documents(season: int) -> list[dict]:
    """Documento resumen del campeon de pilotos y de constructores del ano:
    es la pregunta mas habitual y el documento de clasificacion suelto
    no siempre gana al resto en la busqueda."""
    docs = []
    data = fetch(f"{season}/driverStandings")
    lists = data["StandingsTable"]["StandingsLists"]
    if lists:
        s = lists[0]["DriverStandings"][0]
        drv, con = s["Driver"], s["Constructors"][0]
        name = f"{drv['givenName']} {drv['familyName']}"
        text = (
            f"{name} ({con['name']}) gano el campeonato del mundo de pilotos de "
            f"Formula 1 de {season} con {s['points']} puntos y {s['wins']} victorias. "
            f"Es el campeon del mundo de F1 de {season}."
        )
        docs.append({"id": f"champion-{season}-drivers", "text": text,
                     "metadata": {"type": "champion", "season": season, "name": name}})
    data = fetch(f"{season}/constructorStandings")
    lists = data["StandingsTable"]["StandingsLists"]
    if lists:
        s = lists[0]["ConstructorStandings"][0]
        con = s["Constructor"]
        text = (
            f"{con['name']} gano el campeonato del mundo de constructores de "
            f"Formula 1 de {season} con {s['points']} puntos y {s['wins']} victorias."
        )
        docs.append({"id": f"champion-{season}-constructors", "text": text,
                     "metadata": {"type": "champion", "season": season, "name": con["name"]}})
    return docs


def standings_documents(season: int) -> list[dict]:
    docs = []
    data = fetch(f"{season}/driverStandings")
    lists = data["StandingsTable"]["StandingsLists"]
    if lists:
        for s in lists[0]["DriverStandings"]:
            drv = s["Driver"]
            text = (
                f"En el campeonato de pilotos {season}, {drv['givenName']} {drv['familyName']} "
                f"termino en posicion {s['position']} con {s['points']} puntos "
                f"y {s['wins']} victorias."
            )
            docs.append({"id": f"standing-{season}-{drv['driverId']}", "text": text,
                         "metadata": {"type": "standing", "season": season}})
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga datos de F1 y genera JSONL para el RAG.")
    parser.add_argument("--season", type=int, default=None, help="Temporada (por defecto, la actual).")
    parser.add_argument("--output", default="data/f1_data.jsonl")
    parser.add_argument("--include-sample", action="store_true",
                        help="Anadir el fichero de ejemplo data/sample_data.jsonl al resultado.")
    args = parser.parse_args()

    import datetime
    season = args.season or datetime.date.today().year

    print(f"Descargando datos de la temporada {season} desde {API_BASE} ...", file=sys.stderr)
    docs: list[dict] = []
    for label, fn in [("pilotos", driver_documents), ("equipos", constructor_documents),
                      ("carreras", race_documents), ("clasificacion", standings_documents),
                      ("campeones", champion_documents)]:
        try:
            got = fn(season)
            print(f"  {label}: {len(got)} documentos", file=sys.stderr)
            docs.extend(got)
        except Exception as exc:  # red o API caida: avisar y seguir
            print(f"  {label}: ERROR ({exc})", file=sys.stderr)

    sample_path = Path("data/sample_data.jsonl")
    if args.include_sample and sample_path.exists():
        with open(sample_path, encoding="utf-8") as f:
            sample = [json.loads(line) for line in f if line.strip()]
        print(f"  ejemplo incluido: {len(sample)} documentos", file=sys.stderr)
        docs.extend(sample)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Escritos {len(docs)} documentos en {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

