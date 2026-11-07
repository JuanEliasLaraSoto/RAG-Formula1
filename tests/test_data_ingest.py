"""Pruebas de los documentos de clasificacion y campeon de data_ingest.

Regresion: una temporada en curso nunca debe describirse con lenguaje de
resultados finales ("termino", "campeon del mundo"); una temporada terminada
si. Un titulo decidido antes de tiempo (clinchado) si puede afirmarse.

Ejecutar: python tests/test_data_ingest.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import data_ingest  # noqa: E402

SEASON = 2099  # temporada ficticia; las fechas de las fixtures fijan el estado


def _race(round_no, date, name="Gran Premio de Pruebas"):
    return {"round": str(round_no), "raceName": f"{name} {round_no}", "date": date,
            "Circuit": {"circuitName": "Circuito de Pruebas",
                        "Location": {"locality": "Pruebas", "country": "Testlandia"}}}


def _driver_standing(driver_id, name, surname, pos, points, wins, team="Equipo A"):
    return {"position": str(pos), "points": str(points), "wins": str(wins),
            "Driver": {"driverId": driver_id, "givenName": name, "familyName": surname},
            "Constructors": [{"name": team}]}


def _constructor_standing(cid, name, pos, points, wins):
    return {"position": str(pos), "points": str(points), "wins": str(wins),
            "Constructor": {"constructorId": cid, "name": name}}


def _routes(races, driver_standings, constructor_standings, sprints=None, round_no="1"):
    return {
        f"{SEASON}/races": {"RaceTable": {"Races": races}},
        f"{SEASON}/sprint": {"RaceTable": {"Races": sprints or []}},
        f"{SEASON}/driverStandings": {"StandingsTable": {"StandingsLists": [
            {"season": str(SEASON), "round": round_no, "DriverStandings": driver_standings}]}},
        f"{SEASON}/constructorStandings": {"StandingsTable": {"StandingsLists": [
            {"season": str(SEASON), "round": round_no, "ConstructorStandings": constructor_standings}]}},
    }


def _use(routes):
    data_ingest.fetch = lambda endpoint: routes[endpoint]


def test_tras_primera_carrera_no_hay_campeon():
    # Tras la carrera 1 de 24 nadie es campeon y nadie ha "terminado" nada.
    races = [_race(1, "2026-03-08")] + [_race(n, f"2026-{3 + (n // 4):02d}-15") for n in range(2, 25)]
    drivers = [_driver_standing("piloto_a", "Ana", "Alonso", 1, 25, 1),
               _driver_standing("piloto_b", "Berto", "Bermudez", 2, 18, 0)]
    teams = [_constructor_standing("equipo_a", "Equipo A", 1, 25, 1),
             _constructor_standing("equipo_b", "Equipo B", 2, 18, 0)]
    _use(_routes(races, drivers, teams))
    today = "2026-03-10"

    status = data_ingest.season_status(SEASON, today=today)
    assert not status["completed"] and status["round"] == 1 and status["total"] == 24

    for doc in data_ingest.champion_documents(SEASON, today=today):
        assert "campeon del mundo de F1" not in doc["text"], doc["text"]
        assert "gano el campeonato" not in doc["text"], doc["text"]
        assert "lidera" in doc["text"] and "no esta decidido" in doc["text"], doc["text"]
        assert "carrera 1 de 24" in doc["text"] and "a fecha de 2026-03-10" in doc["text"], doc["text"]

    for doc in data_ingest.standings_documents(SEASON, today=today):
        assert "termino" not in doc["text"], doc["text"]
        assert "ocupa la posicion" in doc["text"], doc["text"]
        assert "aun esta en curso" in doc["text"], doc["text"]


def test_temporada_terminada_describe_campeon():
    races = [_race(1, "2025-03-09")] + [_race(n, f"2025-{3 + (n // 4):02d}-15") for n in range(2, 25)]
    drivers = [_driver_standing("piloto_a", "Ana", "Alonso", 1, 400, 10),
               _driver_standing("piloto_b", "Berto", "Bermudez", 2, 300, 5)]
    teams = [_constructor_standing("equipo_a", "Equipo A", 1, 600, 12),
             _constructor_standing("equipo_b", "Equipo B", 2, 500, 8)]
    _use(_routes(races, drivers, teams, round_no="24"))
    today = "2025-12-20"

    status = data_ingest.season_status(SEASON, today=today)
    assert status["completed"] and status["round"] == 24

    champs = {d["id"]: d["text"] for d in data_ingest.champion_documents(SEASON, today=today)}
    assert "Ana Alonso" in champs[f"champion-{SEASON}-drivers"]
    assert "gano el campeonato del mundo de pilotos" in champs[f"champion-{SEASON}-drivers"]
    assert f"Es el campeon del mundo de F1 de {SEASON}." in champs[f"champion-{SEASON}-drivers"]
    assert "Equipo A gano el campeonato del mundo de constructores" in champs[f"champion-{SEASON}-constructors"]

    standings = data_ingest.standings_documents(SEASON, today=today)
    assert any("termino en posicion 1" in d["text"] for d in standings)
    assert all("aun esta en curso" not in d["text"] for d in standings)


def test_titulo_clinchado_antes_del_final():
    # 19 carreras disputadas de 23, ventaja de 119 puntos: quedan 100 en
    # juego (4 grandes premios, sin sprints), asi que el titulo ya es suyo.
    races = [_race(n, f"2026-0{1 + (n // 5)}-10") for n in range(1, 20)]
    races += [_race(20, "2026-11-08"), _race(21, "2026-11-22"),
              _race(22, "2026-11-29"), _race(23, "2026-12-06")]
    drivers = [_driver_standing("piloto_a", "Ana", "Alonso", 1, 380, 9),
               _driver_standing("piloto_b", "Berto", "Bermudez", 2, 261, 3)]
    # Constructores: ventaja de 126 con 172 puntos en juego: NO decidido.
    teams = [_constructor_standing("equipo_a", "Equipo A", 1, 630, 12),
             _constructor_standing("equipo_b", "Equipo B", 2, 504, 5)]
    _use(_routes(races, drivers, teams, round_no="19"))
    today = "2026-11-07"

    status = data_ingest.season_status(SEASON, today=today)
    assert not status["completed"] and status["remaining_gp"] == 4
    assert status["remaining_sprints"] == 0

    champs = {d["id"]: d["text"] for d in data_ingest.champion_documents(SEASON, today=today)}
    drivers_doc = champs[f"champion-{SEASON}-drivers"]
    assert "se proclamo matematicamente campeon del mundo" in drivers_doc, drivers_doc
    assert f"es el campeon del mundo de F1 de {SEASON}" in drivers_doc, drivers_doc
    assert "Gano el campeonato de pilotos" in drivers_doc, drivers_doc
    assert "a falta de 4 grandes premios" in drivers_doc, drivers_doc
    constructors_doc = champs[f"champion-{SEASON}-constructors"]
    assert "lidera" in constructors_doc and "no esta decidido" in constructors_doc, constructors_doc
    assert "gano el campeonato" not in constructors_doc, constructors_doc


def test_sprint_restante_retrasa_el_clinch():
    # Misma ventaja de 119, pero quedan 3 grandes premios y 2 sprints:
    # 3x25 + 2x8 = 91 < 119 sigue clinchado; con 4 GP y 2 sprints (116) tambien;
    # con 5 GP y 2 sprints (141) ya no.
    races = [_race(1, "2026-03-01")] + [_race(n, f"2026-1{n}-0{n % 9 + 1}") for n in range(2, 7)]
    drivers = [_driver_standing("piloto_a", "Ana", "Alonso", 1, 380, 9),
               _driver_standing("piloto_b", "Berto", "Bermudez", 2, 261, 3)]
    sprints = [_race(5, "2026-12-05"), _race(6, "2026-12-06")]
    _use(_routes(races, drivers, [], sprints=sprints))
    status = data_ingest.season_status(SEASON, today="2026-03-10")
    assert status["remaining_gp"] == 5 and status["remaining_sprints"] == 2
    assert not data_ingest.title_clinched(drivers, status, "drivers")
    doc = data_ingest.champion_documents(SEASON, today="2026-03-10")[0]
    assert "lidera" in doc["text"] and "no esta decidido" in doc["text"]


def test_corpus_real_coherente_con_el_estado_de_la_temporada():
    # Guardia sobre el corpus generado: mientras la temporada en curso no
    # haya terminado, ningun documento suyo puede hablar en pasado final.
    real = Path(__file__).parent.parent / "data" / "f1_data.jsonl"
    if not real.exists():
        return
    import datetime
    hoy = datetime.date.today()
    docs = [json.loads(line) for line in open(real, encoding="utf-8") if line.strip()]
    seasons = {d["metadata"].get("season") for d in docs if d["metadata"].get("season")}
    for season in seasons:
        try:
            status = data_ingest.season_status(season)
        except Exception:
            continue  # sin red: no se puede comprobar esa temporada
        if status["completed"]:
            continue
        del_season = [d for d in docs if d["metadata"].get("season") == season
                      and d["metadata"].get("type") in ("standing", "champion")]
        for d in del_season:
            assert "termino" not in d["text"], d["text"]
            if d["metadata"]["type"] == "champion":
                # En curso: o lider o titulo clinchado, nunca "gano" a secas.
                assert ("lidera" in d["text"]
                        or "se proclamo matematicamente campeon" in d["text"]), d["text"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK {t.__name__}")
    print(f"\n{len(tests)} pruebas superadas")
