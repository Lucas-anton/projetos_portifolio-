"""
src/fetcher.py
--------------
Coleta dados climáticos da Open Meteo API.

API utilizada: Open Meteo (https://open-meteo.com)
  - 100% gratuita, sem autenticação, sem cadastro
  - Dados horários de temperatura, vento, chuva, umidade
  - Histórico + previsão futura

Endpoints:
  /v1/forecast   → previsão + histórico recente por coordenadas
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 15
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0

# Cidades pré-configuradas — fácil de expandir
CITIES: dict[str, dict] = {
    "São Paulo":       {"lat": -23.5505, "lon": -46.6333, "tz": "America/Sao_Paulo"},
    "Rio de Janeiro":  {"lat": -22.9068, "lon": -43.1729, "tz": "America/Sao_Paulo"},
    "Brasília":        {"lat": -15.7801, "lon": -47.9292, "tz": "America/Sao_Paulo"},
    "Fortaleza":       {"lat": -3.7172,  "lon": -38.5433, "tz": "America/Fortaleza"},
    "Manaus":          {"lat": -3.1190,  "lon": -60.0217, "tz": "America/Manaus"},
    "Porto Alegre":    {"lat": -30.0346, "lon": -51.2177, "tz": "America/Sao_Paulo"},
    "Salvador":        {"lat": -12.9714, "lon": -38.5014, "tz": "America/Bahia"},
    "Recife":          {"lat": -8.0476,  "lon": -34.8770, "tz": "America/Recife"},
    "Petrolina":       {"lat": -9.3891,  "lon": -40.5026, "tz": "America/Recife"},
    "Curitiba":        {"lat": -25.4284, "lon": -49.2733, "tz": "America/Sao_Paulo"},
}

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "apparent_temperature",
    "weather_code",
]

# WMO weather codes → descrição legível
WMO_CODES: dict[int, str] = {
    0: "Céu limpo", 1: "Predominantemente limpo", 2: "Parcialmente nublado",
    3: "Nublado", 45: "Neblina", 48: "Neblina com gelo",
    51: "Garoa leve", 53: "Garoa moderada", 55: "Garoa intensa",
    61: "Chuva leve", 63: "Chuva moderada", 65: "Chuva forte",
    71: "Neve leve", 73: "Neve moderada", 75: "Neve intensa",
    80: "Pancadas leves", 81: "Pancadas moderadas", 82: "Pancadas fortes",
    95: "Tempestade", 96: "Tempestade com granizo", 99: "Tempestade intensa",
}


@dataclass
class FetchResult:
    """Dados brutos retornados pela API para uma cidade."""
    city: str
    lat: float
    lon: float
    df: pd.DataFrame
    timezone: str


class OpenMeteoClient:
    """Wrapper HTTP com retry para a Open Meteo API."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def fetch_city(self, city: str, past_days: int = 7) -> Optional[FetchResult]:
        """
        Busca dados horários de uma cidade.

        Parâmetros
        ----------
        city      : nome da cidade (deve estar em CITIES)
        past_days : quantos dias de histórico incluir (máx: 92)
        """
        if city not in CITIES:
            raise ValueError(f"Cidade '{city}' não encontrada. Opções: {list(CITIES)}")

        info = CITIES[city]
        params = {
            "latitude":         info["lat"],
            "longitude":        info["lon"],
            "hourly":           ",".join(HOURLY_VARIABLES),
            "timezone":         info["tz"],
            "past_days":        past_days,
            "forecast_days":    3,
            "wind_speed_unit":  "kmh",
        }

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                logger.info("Buscando dados de %s (tentativa %d)...", city, attempt)
                resp = self.session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                df = self._parse(data, city)
                logger.info("  ✓ %s — %d registros horários.", city, len(df))
                return FetchResult(city=city, lat=info["lat"], lon=info["lon"],
                                   df=df, timezone=info["tz"])

            except requests.exceptions.ConnectionError as e:
                if attempt == RETRY_ATTEMPTS:
                    raise RuntimeError(
                        f"Sem conexão com a Open Meteo API.\nDetalhe: {e}"
                    ) from e
                time.sleep(RETRY_BACKOFF * attempt)

            except requests.exceptions.HTTPError as e:
                raise RuntimeError(f"Erro HTTP {e.response.status_code}: {e}") from e

            except requests.exceptions.Timeout:
                if attempt == RETRY_ATTEMPTS:
                    raise RuntimeError(f"Timeout após {REQUEST_TIMEOUT}s.")
                time.sleep(RETRY_BACKOFF)

        return None

    @staticmethod
    def _parse(data: dict, city: str) -> pd.DataFrame:
        """Converte o JSON da API em DataFrame limpo."""
        hourly = data.get("hourly", {})
        df = pd.DataFrame(hourly)

        df = df.rename(columns={
            "time":                  "datetime",
            "temperature_2m":        "temperatura",
            "relative_humidity_2m":  "umidade",
            "precipitation":         "precipitacao",
            "wind_speed_10m":        "vento_kmh",
            "wind_direction_10m":    "vento_direcao",
            "apparent_temperature":  "sensacao_termica",
            "weather_code":          "codigo_tempo",
        })

        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"]     = df["datetime"].dt.date
        df["hour"]     = df["datetime"].dt.hour
        df["city"]     = city

        # Descrição do tempo em português
        df["condicao"] = df["codigo_tempo"].map(WMO_CODES).fillna("Desconhecido")

        # Classifica se é previsão ou histórico
        now = pd.Timestamp.now().normalize()
        df["tipo"] = df["datetime"].apply(
            lambda x: "Previsão" if x.normalize() > now else "Histórico"
        )

        return df


class WeatherFetcher:
    """
    Coleta dados de múltiplas cidades e consolida em um único DataFrame.

    Uso típico
    ----------
    >>> fetcher = WeatherFetcher(cities=["São Paulo", "Rio de Janeiro"], past_days=7)
    >>> df = fetcher.run()
    """

    def __init__(
        self,
        cities: Optional[list[str]] = None,
        past_days: int = 7,
    ) -> None:
        self.cities = cities or list(CITIES.keys())
        self.past_days = past_days
        self.client = OpenMeteoClient()

    def run(self) -> pd.DataFrame:
        """Coleta todas as cidades e retorna DataFrame consolidado."""
        frames: list[pd.DataFrame] = []

        for city in self.cities:
            try:
                result = self.client.fetch_city(city, self.past_days)
                if result:
                    frames.append(result.df)
            except RuntimeError as e:
                logger.error("Falha ao buscar %s: %s", city, e)

        if not frames:
            raise RuntimeError("Nenhum dado coletado. Verifique sua conexão.")

        df = pd.concat(frames, ignore_index=True)
        logger.info("Total consolidado: %d registros de %d cidades.",
                    len(df), len(frames))
        return df