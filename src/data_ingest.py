"""Descarga datos reales de F1 desde la API Jolpica (sucesora de Ergast)
y los convierte en documentos JSONL listos para indexar.

Uso:
    python src/data_ingest.py --season 2024 --output data/f1_data.jsonl

Sin argumentos descarga la temporada actual. Combina los datos descargados
con el fichero de ejemplo data/sample_data.jsonl si existe.

Las clasificaciones distinguen entre temporadas terminadas y en curso:
- Temporada terminada: se describe al campeon y las posiciones finales.
- Temporada en curso: se describe al lider y las posiciones actuales con la
  carrera de referencia y la fecha; nunca se habla de resultados finales.
- Titulo decidido antes de tiempo: solo se afirma el campeonato cuando la
  ventaja del lider supera los puntos maximos que aun puede sumar el segundo
  (titulo clinchado matematicamente), y se dice que se decidio de forma
  anticipada.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

API_BASE = "https://api.jolpi.ca/ergast/f1"

# Puntos maximos por gran premio. Hasta 2024 la vuelta rapida daba 1 punto
# extra; desde 2025 ya no. Sprints: 8 al ganador (15 maximo por equipo).
MAX_GP_POINTS = {"drivers": (26, 25), "constructors": (44, 43)}
MAX_SPRINT_POINTS = {"drivers": 8, "constructors": 15}


def fetch(endpoint: str) -> dict:
    import requests  # dependencia opcional

    url = f"{API_BASE}/{endpoint}.json?limit=100"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["MRData"]


def _max_points_per_gp(season: int, kind: str) -> int:
    antes_2025, desde_2025 = MAX_GP_POINTS[kind]
    return antes_2025 if season <= 2024 else desde_2025


def season_status(season: int, today: str | None = None) -> dict:
    """Estado de la temporada: en curso o terminada, con la carrera de
    referencia y las carreras y sprints que quedan por disputarse."""
    today = today or datetime.date.today().isoformat()
    races = fetch(f"{season}/races")["RaceTable"]["Races"]
    done = [r for r in races if r.get("date", "9999-12-31") <= today]
    remaining = [r for r in races if r.get("date", "9999-12-31") > today]
    remaining_sprints = 0
    try:
        sprints = fetch(f"{season}/sprint")["RaceTable"]["Races"]
        remaining_sprints = sum(1 for r in sprints if r.get("date", "9999-12-31") > today)
    except Exception:  # temporadas sin datos de sprint
        pass
    return {
        "season": season,
        "total": len(races),
        "round": len(done),
        "last_race": done[-1] if done else None,
        "completed": bool(races) and not remaining,
        "remaining_gp": len(remaining),
        "remaining_sprints": remaining_sprints,
        "as_of": today,
    }


def title_clinched(standings: list[dict], status: dict, kind: str) -> bool:
    """True si el lider ya no puede ser alcanzado: su ventaja supera los
    puntos maximos que aun quedan en juego (grandes premios y sprints)."""
    if status["completed"] or len(standings) < 2:
        return False
    gap = float(standings[0]["points"]) - float(standings[1]["points"])
    max_remaining = (
        status["remaining_gp"] * _max_points_per_gp(status["season"], kind)
        + status["remaining_sprints"] * MAX_SPRINT_POINTS[kind]
    )
    return gap > max_remaining


def _contexto_en_curso(status: dict) -> str:
    """Frase de contexto para una temporada en curso: carrera de referencia
    y fecha de los datos."""
    if status["last_race"]:
        last = status["last_race"]
        return (f"tras la carrera {status['round']} de {status['total']} "
                f"({last['raceName']}, {last['date']}), a fecha de {status['as_of']}")
    return f"antes de la primera carrera, a fecha de {status['as_of']}"


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


def champion_documents(season: int, status: dict | None = None,
                       today: str | None = None) -> list[dict]:
    """Documento resumen del campeon (o del lider) de pilotos y de
    constructores: es la pregunta mas habitual y el documento de
    clasificacion suelto no siempre gana al resto en la busqueda.

    Solo se habla de campeon cuando la temporada ha terminado o el titulo
    esta decidido matematicamente; si no, se describe al lider con la
    carrera de referencia y la fecha."""
    status = status or season_status(season, today=today)
    docs = []
    data = fetch(f"{season}/driverStandings")
    lists = data["StandingsTable"]["StandingsLists"]
    standings = lists[0]["DriverStandings"] if lists else []
    if standings:
        s = standings[0]
        drv, con = s["Driver"], s["Constructors"][0]
        name = f"{drv['givenName']} {drv['familyName']}"
        if status["completed"]:
            text = (
                f"{name} ({con['name']}) gano el campeonato del mundo de pilotos de "
                f"Formula 1 de {season} con {s['points']} puntos y {s['wins']} victorias. "
                f"Es el campeon del mundo de F1 de {season}."
            )
        elif title_clinched(standings, status, "drivers"):
            text = (
                f"{name} ({con['name']}) se proclamo matematicamente campeon del mundo de "
                f"pilotos de Formula 1 de {season} {_contexto_en_curso(status)}, "
                f"con {s['points']} puntos y {s['wins']} victorias, a falta de "
                f"{status['remaining_gp']} grandes premios. Gano el campeonato de pilotos de "
                f"{season} de forma anticipada: es el campeon del mundo de F1 de {season}."
            )
        else:
            text = (
                f"{name} ({con['name']}) lidera el campeonato del mundo de pilotos de "
                f"Formula 1 de {season} con {s['points']} puntos y {s['wins']} victorias "
                f"{_contexto_en_curso(status)}. La temporada {season} aun esta en curso y "
                f"el campeonato no esta decidido."
            )
        docs.append({"id": f"champion-{season}-drivers", "text": text,
                     "metadata": {"type": "champion", "season": season, "name": name}})
    data = fetch(f"{season}/constructorStandings")
    lists = data["StandingsTable"]["StandingsLists"]
    standings = lists[0]["ConstructorStandings"] if lists else []
    if standings:
        s = standings[0]
        con = s["Constructor"]
        if status["completed"]:
            text = (
                f"{con['name']} gano el campeonato del mundo de constructores de "
                f"Formula 1 de {season} con {s['points']} puntos y {s['wins']} victorias."
            )
        elif title_clinched(standings, status, "constructors"):
            text = (
                f"{con['name']} se proclamo matematicamente campeon del mundo de "
                f"constructores de Formula 1 de {season} {_contexto_en_curso(status)}, "
                f"con {s['points']} puntos y {s['wins']} victorias, a falta de "
                f"{status['remaining_gp']} grandes premios. Gano el campeonato de "
                f"constructores de {season} de forma anticipada."
            )
        else:
            text = (
                f"{con['name']} lidera el campeonato del mundo de constructores de "
                f"Formula 1 de {season} con {s['points']} puntos y {s['wins']} victorias "
                f"{_contexto_en_curso(status)}. La temporada {season} aun esta en curso y "
                f"el campeonato no esta decidido."
            )
        docs.append({"id": f"champion-{season}-constructors", "text": text,
                     "metadata": {"type": "champion", "season": season, "name": con["name"]}})
    return docs


def standings_documents(season: int, status: dict | None = None,
                        today: str | None = None) -> list[dict]:
    status = status or season_status(season, today=today)
    docs = []
    data = fetch(f"{season}/driverStandings")
    lists = data["StandingsTable"]["StandingsLists"]
    if lists:
        for s in lists[0]["DriverStandings"]:
            drv = s["Driver"]
            if status["completed"]:
                text = (
                    f"En el campeonato de pilotos {season}, {drv['givenName']} {drv['familyName']} "
                    f"termino en posicion {s['position']} con {s['points']} puntos "
                    f"y {s['wins']} victorias."
                )
            else:
                text = (
                    f"En el campeonato de pilotos {season}, {drv['givenName']} {drv['familyName']} "
                    f"ocupa la posicion {s['position']} con {s['points']} puntos "
                    f"y {s['wins']} victorias {_contexto_en_curso(status)}. "
                    f"La temporada aun esta en curso."
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

    season = args.season or datetime.date.today().year

    print(f"Descargando datos de la temporada {season} desde {API_BASE} ...", file=sys.stderr)
    docs: list[dict] = []
    status = None
    for label, fn in [("pilotos", driver_documents), ("equipos", constructor_documents),
                      ("carreras", race_documents), ("clasificacion", standings_documents),
                      ("campeones", champion_documents)]:
        try:
            if fn in (standings_documents, champion_documents):
                if status is None:
                    status = season_status(season)
                    estado = "terminada" if status["completed"] else "en curso"
                    print(f"  estado de la temporada: {estado} "
                          f"({status['round']} de {status['total']} carreras)", file=sys.stderr)
                got = fn(season, status=status)
            else:
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
