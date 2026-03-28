"""
src/analyzer.py
---------------
Responsável por:
  - Buscar dados reais da NASA Open API (sem autenticação pesada)
  - Limpar e estruturar os dados em DataFrames prontos para análise
  - Calcular métricas e retornar um AnalysisReport tipado

APIs utilizadas:
  - NeoWs (Near Earth Object Web Service): asteroides próximos à Terra
    Docs: https://api.nasa.gov/#NeoWS
  - APOD (Astronomy Picture of the Day): metadados das imagens diárias
    Docs: https://api.nasa.gov/#apod

Nenhuma chave pessoal é necessária — usamos DEMO_KEY (limite: 30 req/hora).
Para aumentar o limite, cadastre-se gratuitamente em https://api.nasa.gov
e exporte: export NASA_API_KEY=sua_chave
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
BASE_URL = "https://api.nasa.gov"
API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")  # funciona sem cadastro
REQUEST_TIMEOUT = 15  # segundos
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0  # segundos entre tentativas


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------
@dataclass
class AnalysisReport:
    """Contêiner tipado com todos os dados processados e métricas calculadas."""

    # — Dados brutos processados —
    neo_df: pd.DataFrame          # asteroides próximos à Terra
    apod_df: pd.DataFrame         # metadados APOD

    # — Métricas NeoWs —
    total_asteroids: int
    potentially_hazardous_count: int
    hazardous_rate: float
    avg_diameter_km: float
    max_diameter_km: float
    closest_approach_km: float
    fastest_asteroid_kmh: float

    # — Métricas APOD —
    total_apod_entries: int
    media_type_counts: pd.Series
    top_keywords: pd.Series

    # — Metadados —
    date_range_start: str
    date_range_end: str
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  RELATÓRIO DE ANÁLISE — NASA OPEN DATA",
            "=" * 62,
            f"  Período analisado    : {self.date_range_start} → {self.date_range_end}",
            f"  Asteroides coletados : {self.total_asteroids}",
            f"  Potencialmente perig.: {self.potentially_hazardous_count} "
            f"({self.hazardous_rate:.1%})",
            f"  Diâmetro médio       : {self.avg_diameter_km:.3f} km",
            f"  Maior asteroide      : {self.max_diameter_km:.3f} km",
            f"  Aproximação mais próx: {self.closest_approach_km:,.0f} km da Terra",
            f"  Mais rápido          : {self.fastest_asteroid_kmh:,.0f} km/h",
            f"  Entradas APOD        : {self.total_apod_entries}",
            "",
            "  Avisos:",
        ]
        if self.warnings:
            lines += [f"    ⚠  {w}" for w in self.warnings]
        else:
            lines.append("    ✓  Nenhum problema encontrado.")
        lines.append("=" * 62)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cliente HTTP com retry
# ---------------------------------------------------------------------------
class NasaApiClient:
    """
    Wrapper sobre requests com retry automático e tratamento de erros.

    Centraliza toda comunicação HTTP — facilita mock em testes unitários.
    """

    def __init__(self, api_key: str = API_KEY) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """GET com retry exponencial. Lança RuntimeError após esgotar tentativas."""
        url = f"{BASE_URL}{endpoint}"
        params = {**(params or {}), "api_key": self.api_key}

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                logger.info("GET %s (tentativa %d/%d)", endpoint, attempt, RETRY_ATTEMPTS)
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "?"
                # 429 = rate limit — espera mais antes de tentar novamente
                if status == 429:
                    wait = RETRY_BACKOFF * attempt * 3
                    logger.warning("Rate limit atingido. Aguardando %.0fs...", wait)
                    time.sleep(wait)
                elif attempt == RETRY_ATTEMPTS:
                    raise RuntimeError(f"HTTP {status} em {url}: {e}") from e

            except requests.exceptions.ConnectionError as e:
                if attempt == RETRY_ATTEMPTS:
                    raise RuntimeError(
                        f"Sem conexão com a API da NASA.\n"
                        f"Verifique sua internet e tente novamente.\nDetalhe: {e}"
                    ) from e
                time.sleep(RETRY_BACKOFF * attempt)

            except requests.exceptions.Timeout:
                if attempt == RETRY_ATTEMPTS:
                    raise RuntimeError(f"Timeout após {REQUEST_TIMEOUT}s em {url}")
                time.sleep(RETRY_BACKOFF)

        raise RuntimeError("Falha após todas as tentativas.")  # pragma: no cover


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------
class NasaAnalyzer:
    """
    Orquestra a coleta e análise de dados da NASA Open API.

    Uso típico
    ----------
    >>> analyzer = NasaAnalyzer(days=7)
    >>> report = analyzer.run()
    >>> print(report.summary())

    Parâmetros
    ----------
    days : int
        Janela de busca de asteroides (máx. 7 dias pela API NeoWs).
    cache_dir : Path | None
        Se fornecido, salva JSONs brutos para reuso offline e testes.
    """

    def __init__(
        self,
        days: int = 7,
        cache_dir: Optional[Path] = None,
    ) -> None:
        if days > 7:
            raise ValueError("A API NeoWs suporta no máximo 7 dias por requisição.")
        self.days = days
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.client = NasaApiClient()
        self._neo_raw: dict = {}
        self._apod_raw: list[dict] = []
        self._neo_df: pd.DataFrame = pd.DataFrame()
        self._apod_df: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Etapa 1 — Coleta
    # ------------------------------------------------------------------
    def fetch(self) -> "NasaAnalyzer":
        """Busca dados das duas APIs e armazena internamente."""
        end_date = date.today()
        start_date = end_date - timedelta(days=self.days - 1)
        self._start_str = start_date.isoformat()
        self._end_str = end_date.isoformat()

        self._fetch_neo(self._start_str, self._end_str)
        self._fetch_apod(self._start_str, self._end_str)
        return self

    def _fetch_neo(self, start: str, end: str) -> None:
        """Coleta asteroides próximos à Terra no período especificado."""
        logger.info("Coletando dados NeoWs: %s → %s", start, end)
        self._neo_raw = self.client.get(
            "/neo/rest/v1/feed",
            {"start_date": start, "end_date": end, "detailed": "false"},
        )
        total = self._neo_raw.get("element_count", 0)
        logger.info("NeoWs: %d asteroides retornados.", total)

    def _fetch_apod(self, start: str, end: str) -> None:
        """Coleta metadados do APOD no mesmo período."""
        logger.info("Coletando dados APOD: %s → %s", start, end)
        data = self.client.get(
            "/planetary/apod",
            {"start_date": start, "end_date": end, "thumbs": "false"},
        )
        # A API retorna lista ou dict (quando é 1 entrada)
        self._apod_raw = data if isinstance(data, list) else [data]
        logger.info("APOD: %d entradas retornadas.", len(self._apod_raw))

    # ------------------------------------------------------------------
    # Etapa 2 — Transformação
    # ------------------------------------------------------------------
    def transform(self) -> "NasaAnalyzer":
        """Achata os JSONs em DataFrames limpos e tipados."""
        self._neo_df = self._transform_neo()
        self._apod_df = self._transform_apod()
        return self

    def _transform_neo(self) -> pd.DataFrame:
        """
        Achata a estrutura aninhada do NeoWs.

        Estrutura original: {near_earth_objects: {data: [asteroides]}}
        """
        records = []
        neo_by_date = self._neo_raw.get("near_earth_objects", {})

        for day, asteroids in neo_by_date.items():
            for ast in asteroids:
                # Pega o close approach mais próximo (pode haver múltiplos)
                approaches = ast.get("close_approach_data", [{}])
                closest = min(
                    approaches,
                    key=lambda x: float(x.get("miss_distance", {}).get("kilometers", float("inf"))),
                    default={},
                )

                diam = ast.get("estimated_diameter", {}).get("kilometers", {})
                records.append({
                    "date": day,
                    "id": ast.get("id"),
                    "name": ast.get("name"),
                    "is_potentially_hazardous": ast.get("is_potentially_hazardous_asteroid", False),
                    "diameter_min_km": float(diam.get("estimated_diameter_min", 0)),
                    "diameter_max_km": float(diam.get("estimated_diameter_max", 0)),
                    "diameter_avg_km": (
                        float(diam.get("estimated_diameter_min", 0))
                        + float(diam.get("estimated_diameter_max", 0))
                    ) / 2,
                    "miss_distance_km": float(
                        closest.get("miss_distance", {}).get("kilometers", 0)
                    ),
                    "relative_velocity_kmh": float(
                        closest.get("relative_velocity", {}).get("kilometers_per_hour", 0)
                    ),
                    "orbiting_body": closest.get("orbiting_body", "Earth"),
                    "absolute_magnitude": ast.get("absolute_magnitude_h"),
                })

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("NeoWs retornou dados vazios.")
            return df

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        logger.info("NeoWs transformado: %d linhas.", len(df))
        return df

    def _transform_apod(self) -> pd.DataFrame:
        """Normaliza o JSON do APOD em um DataFrame plano."""
        if not self._apod_raw:
            return pd.DataFrame()

        df = pd.json_normalize(self._apod_raw)

        # Garante colunas mínimas
        for col in ["date", "title", "media_type", "explanation", "url"]:
            if col not in df.columns:
                df[col] = None

        df["date"] = pd.to_datetime(df["date"])
        df["title_word_count"] = df["title"].str.split().str.len()
        df["explanation_length"] = df["explanation"].str.len()

        logger.info("APOD transformado: %d linhas.", len(df))
        return df

    # ------------------------------------------------------------------
    # Etapa 3 — Análise
    # ------------------------------------------------------------------
    def analyze(self) -> AnalysisReport:
        """Calcula métricas e monta o AnalysisReport."""
        if self._neo_df.empty and self._apod_df.empty:
            raise RuntimeError("Execute .fetch().transform() antes de .analyze().")

        warnings: list[str] = []
        neo = self._neo_df
        apod = self._apod_df

        # — Métricas NeoWs —
        total = len(neo)
        hazardous_count = int(neo["is_potentially_hazardous"].sum())
        hazardous_rate = hazardous_count / total if total > 0 else 0.0
        avg_diam = neo["diameter_avg_km"].mean() if total > 0 else 0.0
        max_diam = neo["diameter_max_km"].max() if total > 0 else 0.0
        closest_km = neo["miss_distance_km"].min() if total > 0 else 0.0
        fastest = neo["relative_velocity_kmh"].max() if total > 0 else 0.0

        if hazardous_rate > 0.3:
            warnings.append(f"Alto índice de asteroides perigosos: {hazardous_rate:.1%}")

        # — Métricas APOD —
        total_apod = len(apod)
        media_counts = apod["media_type"].value_counts() if total_apod > 0 else pd.Series(dtype=int)

        # Top palavras nos títulos (stopwords simples)
        stopwords = {"a", "of", "the", "in", "and", "an", "to", "with", "from", "for", "on"}
        if total_apod > 0 and "title" in apod.columns:
            all_words = (
                apod["title"]
                .dropna()
                .str.lower()
                .str.replace(r"[^a-z\s]", "", regex=True)
                .str.split()
                .explode()
            )
            top_keywords = (
                all_words[~all_words.isin(stopwords)]
                .value_counts()
                .head(10)
            )
        else:
            top_keywords = pd.Series(dtype=int)

        logger.info("Análise concluída.")

        return AnalysisReport(
            neo_df=neo,
            apod_df=apod,
            total_asteroids=total,
            potentially_hazardous_count=hazardous_count,
            hazardous_rate=hazardous_rate,
            avg_diameter_km=avg_diam,
            max_diameter_km=max_diam,
            closest_approach_km=closest_km,
            fastest_asteroid_kmh=fastest,
            total_apod_entries=total_apod,
            media_type_counts=media_counts,
            top_keywords=top_keywords,
            date_range_start=self._start_str,
            date_range_end=self._end_str,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Conveniência
    # ------------------------------------------------------------------
    def run(self) -> AnalysisReport:
        """Pipeline completo: fetch → transform → analyze."""
        return self.fetch().transform().analyze()