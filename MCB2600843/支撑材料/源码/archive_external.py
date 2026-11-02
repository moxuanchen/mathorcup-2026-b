"""Offline verification of archived external responses.

The competition run is offline. Raw API responses retrieved during the single
network session are archived under 临时产物/外部数据/; this script re-computes
their SHA-256 and writes the provenance record without any network access.
The optional online refresh helper is not included in this offline package
and is not part of run_all.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "临时产物" / "外部数据"
OUT = ROOT / "求解结果" / "数据" / "外部"

SOURCES = [
    {
        "tag": "open_meteo_era5_port_utc",
        "raw_file": "open_meteo_era5_port_utc.json",
        "product": "Open-Meteo Archive API, model=era5",
        "request": {
            "latitude": 28.2797,
            "longitude": 85.3777,
            "start_date": "2026-08-20",
            "end_date": "2026-08-27",
            "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
            "timezone": "UTC",
            "elevation": "nan (automatic downscaling disabled)",
        },
        "scope": "event-window external comparison only; not a model input",
    },
    {
        "tag": "nasa_power_port_hourly_utc",
        "raw_file": "nasa_power_port_hourly_utc.json",
        "product": "NASA POWER hourly point, MERRA-2 based",
        "request": {
            "parameters": ["T2M", "PRECTOTCORR"],
            "longitude": 85.3777,
            "latitude": 28.2797,
            "start": "20260819",
            "end": "20260828",
            "time-standard": "UTC",
        },
        "scope": "event-window external comparison only; not a model input",
    },
    {
        "tag": "nasa_power_station_daily_utc",
        "raw_file": "nasa_power_station_daily_utc.json",
        "product": "NASA POWER daily point, MERRA-2 based",
        "request": {
            "parameters": ["T2M", "PRECTOTCORR"],
            "longitude": 85.30,
            "latitude": 28.40,
            "start": "20210101",
            "end": "20250802",
            "time-standard": "UTC",
        },
        "scope": "assumed representative station point; not verified installation metadata",
    },
    {
        "tag": "historical_forecast_ifs",
        "raw_file": "historical_forecast_ifs.json",
        "product": "Open-Meteo Historical Forecast API, model=ecmwf_ifs04",
        "request": {
            "latitude": 28.2797,
            "longitude": 85.3777,
            "start_date": "2026-08-20",
            "end_date": "2026-08-27",
            "hourly": ["temperature_2m", "precipitation"],
            "timezone": "UTC",
            "elevation": "nan",
        },
        "scope": "valid-time archive only; no stored initialization or lead time, so not an issued-forecast validation",
    },
]

DOCUMENTATION = [
    ("open_meteo_product_docs", "open_meteo_docs.html", "Open-Meteo historical weather API documentation"),
    ("nasa_daily_time_standard", "nasa_power_daily_docs.html", "NASA POWER daily temporal API documentation"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sources, missing = [], []
    for item in SOURCES:
        path = RAW / item["raw_file"]
        if not path.is_file():
            missing.append(item["raw_file"])
            continue
        raw = path.read_bytes()
        try:
            keys = list(json.loads(raw))[:15]
        except (ValueError, UnicodeError):
            keys = None
        sources.append({
            "tag": item["tag"],
            "product": item["product"],
            "request": item["request"],
            "raw_file": path.relative_to(ROOT).as_posix(),
            "bytes": len(raw),
            "sha256": sha256(path),
            "json_keys": keys,
            "retrieval": "archived during the single online session; this record is recomputed offline",
            "status": "archived" if keys is not None else "archived_unparsed",
            "scope": item["scope"],
        })
    documentation = []
    for tag, name, product in DOCUMENTATION:
        path = RAW / name
        documentation.append({
            "tag": tag,
            "product": product,
            "raw_file": path.relative_to(ROOT).as_posix() if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
            "status": "archived" if path.is_file() else "missing",
        })
    result = {
        "schema_version": 2,
        "version": "v5.2",
        "network_access": "none during this run; hashes recomputed from archived responses",
        "sources": sources,
        "documentation": documentation,
        "missing_raw_files": missing,
        "withdrawn_claims": {
            "open_meteo_default_product": "unqualified default is Best Match (IFS HRES + ERA5 + ERA5-Land); an unqualified ERA5 claim is not supportable",
            "nasa_default_day_boundary": "default is local standard time, so a UTC+8 versus default-LST daily comparison is not a valid same-day comparison",
            "event_parameters_from_media": "reported runout, travel time, front speed and peak discharge have no verifiable original text or observation definition for this event and are withdrawn from validation",
            "station_metadata": "public station metadata endpoint returned no usable record; assumed coordinates, inferred altitude, UTC+8 and hour-label convention remain hypotheses",
            "issued_forecast": "archived forecast response carries valid times only; without initialization and lead time it cannot demonstrate pre-event availability or skill",
        },
    }
    (OUT / "06_外部来源核实.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"external provenance archived: {len(sources)} sources, {len(documentation)} documents, {len(missing)} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
