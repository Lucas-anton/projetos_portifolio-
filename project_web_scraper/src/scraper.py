"""
src/scraper.py
--------------
Coleta notícias do Hacker News via API oficial pública.

API utilizada: Hacker News Firebase API
  Docs   : https://github.com/HackerNews/API
  Base   : https://hacker-news.firebaseio.com/v0/
  Licença: Dados públicos, sem autenticação necessária

Endpoints usados:
  /topstories.json   → IDs das top stories (até 500)
  /item/{id}.json    → Detalhes de cada item (título, url, score, etc.)

Design:
  - HackerNewsClient  → toda comunicação HTTP (fácil de mockar em testes)
  - HackerNewsScraper → orquestra coleta em paralelo com ThreadPoolExecutor
  - ScrapeResult      → dataclass tipada com os dados brutos coletados
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
BASE_URL = "https://hacker-news.firebaseio.com/v0"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={id}"
REQUEST_TIMEOUT = 10
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 1.5
MAX_WORKERS = 10   # threads para busca paralela de itens


# ---------------------------------------------------------------------------
# Dataclass de resultado bruto
# ---------------------------------------------------------------------------
@dataclass
class ScrapeResult:
    """Dados brutos coletados, antes de qualquer análise."""

    items: list[dict]
    total_fetched: int
    total_requested: int
    fetch_errors: int
    elapsed_seconds: float
    scraped_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def success_rate(self) -> float:
        return self.total_fetched / self.total_requested if self.total_requested else 0.0


# ---------------------------------------------------------------------------
# Cliente HTTP
# ---------------------------------------------------------------------------
class HackerNewsClient:
    """
    Wrapper sobre requests com retry e backoff exponencial.

    Separado do Scraper para facilitar mock em testes unitários:
        client = MagicMock(spec=HackerNewsClient)
        client.get_top_ids.return_value = [1, 2, 3]
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_top_ids(self, category: str = "topstories") -> list[int]:
        """
        Retorna lista de IDs das top stories.

        Parâmetros
        ----------
        category : str
            topstories | newstories | beststories | askstories | showstories
        """
        valid = {"topstories", "newstories", "beststories", "askstories", "showstories"}
        if category not in valid:
            raise ValueError(f"Categoria inválida. Use uma de: {valid}")

        url = f"{BASE_URL}/{category}.json"
        data = self._get(url)
        return data if isinstance(data, list) else []

    def get_item(self, item_id: int) -> Optional[dict]:
        """Retorna os dados de um item pelo ID. Retorna None em caso de erro."""
        url = f"{BASE_URL}/item/{item_id}.json"
        try:
            return self._get(url)
        except Exception as e:
            logger.debug("Erro ao buscar item %d: %s", item_id, e)
            return None

    def _get(self, url: str) -> any:
        """GET com retry e backoff exponencial."""
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as e:
                if attempt == RETRY_ATTEMPTS:
                    raise RuntimeError(f"HTTP {e.response.status_code}: {url}") from e
            except requests.exceptions.ConnectionError as e:
                if attempt == RETRY_ATTEMPTS:
                    raise RuntimeError(
                        "Sem conexão com a API do Hacker News.\n"
                        f"Verifique sua internet. Detalhe: {e}"
                    ) from e
            except requests.exceptions.Timeout:
                if attempt == RETRY_ATTEMPTS:
                    raise RuntimeError(f"Timeout após {REQUEST_TIMEOUT}s: {url}")

            time.sleep(RETRY_BACKOFF ** attempt)

        raise RuntimeError("Falha após todas as tentativas.")


# ---------------------------------------------------------------------------
# Scraper principal
# ---------------------------------------------------------------------------
class HackerNewsScraper:
    """
    Coleta notícias do Hacker News em paralelo via ThreadPoolExecutor.

    Uso típico
    ----------
    >>> scraper = HackerNewsScraper(limit=100)
    >>> result = scraper.run()
    >>> print(f"Coletadas: {result.total_fetched} notícias")

    Parâmetros
    ----------
    limit    : int  — quantas stories coletar (padrão: 100, máx: 500)
    category : str  — topstories | newstories | beststories
    workers  : int  — threads paralelas (padrão: 10)
    """

    def __init__(
        self,
        limit: int = 100,
        category: str = "topstories",
        workers: int = MAX_WORKERS,
    ) -> None:
        if not 1 <= limit <= 500:
            raise ValueError("limit deve estar entre 1 e 500.")
        self.limit = limit
        self.category = category
        self.workers = workers
        self.client = HackerNewsClient()

    # ------------------------------------------------------------------
    # Etapa 1 — Busca IDs
    # ------------------------------------------------------------------
    def _fetch_ids(self) -> list[int]:
        logger.info("Buscando IDs da categoria '%s'...", self.category)
        ids = self.client.get_top_ids(self.category)
        selected = ids[: self.limit]
        logger.info("%d IDs selecionados de %d disponíveis.", len(selected), len(ids))
        return selected

    # ------------------------------------------------------------------
    # Etapa 2 — Busca itens em paralelo
    # ------------------------------------------------------------------
    def _fetch_items(self, ids: list[int]) -> tuple[list[dict], int]:
        """Busca todos os itens em paralelo. Retorna (itens_válidos, erros)."""
        items: list[dict] = []
        errors = 0

        logger.info("Buscando %d itens com %d threads...", len(ids), self.workers)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.client.get_item, id_): id_ for id_ in ids}

            for i, future in enumerate(as_completed(futures), 1):
                if i % 20 == 0:
                    logger.info("  %d/%d itens processados...", i, len(ids))
                result = future.result()
                if result and result.get("type") == "story" and result.get("title"):
                    items.append(self._enrich(result))
                elif result is None:
                    errors += 1

        logger.info("Coleta concluída: %d válidos, %d erros.", len(items), errors)
        return items, errors

    # ------------------------------------------------------------------
    # Enriquecimento do item
    # ------------------------------------------------------------------
    @staticmethod
    def _enrich(item: dict) -> dict:
        """
        Adiciona campos derivados ao item bruto.

        Campos adicionados
        ------------------
        - hn_url        : link direto para a discussão no HN
        - domain        : domínio extraído da URL original
        - published_at  : timestamp Unix → datetime ISO
        - published_date: apenas a data (YYYY-MM-DD)
        - published_hour: hora de publicação (0-23)
        """
        raw_url = item.get("url", "")
        try:
            domain = urlparse(raw_url).netloc.replace("www.", "")
        except Exception:
            domain = ""

        ts = item.get("time")
        published_at = datetime.fromtimestamp(ts).isoformat() if ts else None
        published_date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else None
        published_hour = datetime.fromtimestamp(ts).hour if ts else None

        return {
            # Campos originais da API
            "id":            item.get("id"),
            "title":         item.get("title", "").strip(),
            "url":           raw_url,
            "score":         item.get("score", 0),
            "by":            item.get("by", ""),
            "descendants":   item.get("descendants", 0),   # nº de comentários
            "type":          item.get("type", "story"),
            # Campos enriquecidos
            "hn_url":        HN_ITEM_URL.format(id=item.get("id")),
            "domain":        domain,
            "published_at":  published_at,
            "published_date": published_date,
            "published_hour": published_hour,
        }

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def run(self) -> ScrapeResult:
        """Executa coleta completa e retorna ScrapeResult."""
        t0 = time.time()
        ids = self._fetch_ids()
        items, errors = self._fetch_items(ids)
        elapsed = round(time.time() - t0, 2)

        return ScrapeResult(
            items=items,
            total_fetched=len(items),
            total_requested=len(ids),
            fetch_errors=errors,
            elapsed_seconds=elapsed,
        )