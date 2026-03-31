"""
src/pipeline.py
---------------
Pipeline de dados de criptomoedas.

Fluxo:
  1. Coleta preços reais da CoinGecko API (gratuita, sem auth)
  2. Processa e enriquece os dados
  3. Persiste no SQLite (output/crypto.db)
  4. Exporta JSON para o dashboard web consumir

API: https://api.coingecko.com/api/v3
  - Gratuita, sem autenticação
  - Rate limit: 10-30 req/min (plano free)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DB_PATH   = Path("output/crypto.db")
JSON_PATH = Path("web/data.json")
BASE_URL  = "https://api.coingecko.com/api/v3"

COINS = [
    "bitcoin", "ethereum", "solana", "binancecoin",
    "cardano", "ripple", "dogecoin", "polkadot",
]

COIN_SYMBOLS = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL",
    "binancecoin": "BNB", "cardano": "ADA", "ripple": "XRP",
    "dogecoin": "DOGE", "polkadot": "DOT",
}

COIN_COLORS = {
    "bitcoin": "#F7931A", "ethereum": "#627EEA", "solana": "#9945FF",
    "binancecoin": "#F3BA2F", "cardano": "#0033AD", "ripple": "#00AAE4",
    "dogecoin": "#C2A633", "polkadot": "#E6007A",
}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Cria o banco e tabelas se não existirem."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            coin_id         TEXT    NOT NULL,
            symbol          TEXT    NOT NULL,
            price_usd       REAL    NOT NULL,
            market_cap      REAL,
            volume_24h      REAL,
            change_1h       REAL,
            change_24h      REAL,
            change_7d       REAL,
            high_24h        REAL,
            low_24h         REAL,
            collected_at    TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ran_at       TEXT NOT NULL,
            coins_saved  INTEGER,
            status       TEXT,
            error        TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_coin_time
        ON prices (coin_id, collected_at)
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class CoinGeckoCollector:
    """Coleta dados da CoinGecko API."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "crypto-pipeline/1.0",
        })

    def fetch(self, coins: list[str] = COINS) -> list[dict]:
        """Busca preços, market cap, volume e variações."""
        logger.info("Coletando dados de %d moedas...", len(coins))
        params = {
            "ids":                   ",".join(coins),
            "vs_currency":           "usd",
            "include_market_cap":    "true",
            "include_24hr_vol":      "true",
            "include_24hr_change":   "true",
            "include_last_updated_at": "true",
            "price_change_percentage": "1h,24h,7d",
        }
        for attempt in range(1, 4):
            try:
                resp = self.session.get(
                    f"{BASE_URL}/coins/markets",
                    params={**params, "per_page": len(coins), "page": 1},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info("✓ %d moedas coletadas.", len(data))
                return data
            except requests.exceptions.ConnectionError as e:
                if attempt == 3:
                    raise RuntimeError(f"Sem conexão com CoinGecko: {e}") from e
                time.sleep(2 ** attempt)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limit — aguardando 60s...")
                    time.sleep(60)
                else:
                    raise
        return []


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------
def process(raw: list[dict]) -> list[dict]:
    """Limpa e enriquece os dados brutos."""
    now = datetime.now().isoformat(timespec="seconds")
    records = []
    for item in raw:
        coin_id = item.get("id", "")
        records.append({
            "coin_id":      coin_id,
            "symbol":       COIN_SYMBOLS.get(coin_id, item.get("symbol", "").upper()),
            "price_usd":    item.get("current_price") or 0.0,
            "market_cap":   item.get("market_cap") or 0.0,
            "volume_24h":   item.get("total_volume") or 0.0,
            "change_1h":    item.get("price_change_percentage_1h_in_currency") or 0.0,
            "change_24h":   item.get("price_change_percentage_24h") or 0.0,
            "change_7d":    item.get("price_change_percentage_7d_in_currency") or 0.0,
            "high_24h":     item.get("high_24h") or 0.0,
            "low_24h":      item.get("low_24h") or 0.0,
            "collected_at": now,
        })
    return records


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def save_to_db(records: list[dict], conn: sqlite3.Connection) -> int:
    """Persiste os registros no SQLite."""
    conn.executemany("""
        INSERT INTO prices
        (coin_id, symbol, price_usd, market_cap, volume_24h,
         change_1h, change_24h, change_7d, high_24h, low_24h, collected_at)
        VALUES
        (:coin_id, :symbol, :price_usd, :market_cap, :volume_24h,
         :change_1h, :change_24h, :change_7d, :high_24h, :low_24h, :collected_at)
    """, records)
    conn.commit()
    return len(records)


def export_json(conn: sqlite3.Connection, json_path: Path = JSON_PATH) -> None:
    """
    Exporta dados para JSON consumido pelo dashboard.

    Estrutura exportada:
      - latest:   último snapshot de cada moeda
      - history:  últimas 100 leituras por moeda (para gráficos)
      - runs:     últimas 20 execuções do pipeline
      - meta:     timestamp e total de registros
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Último preço de cada moeda
    latest_rows = conn.execute("""
        SELECT p.*
        FROM prices p
        INNER JOIN (
            SELECT coin_id, MAX(collected_at) as max_at
            FROM prices GROUP BY coin_id
        ) latest ON p.coin_id = latest.coin_id AND p.collected_at = latest.max_at
        ORDER BY market_cap DESC
    """).fetchall()

    cols = [d[0] for d in conn.execute("SELECT * FROM prices LIMIT 0").description]
    latest = [dict(zip(cols, row)) for row in latest_rows]

    # Histórico por moeda
    history: dict[str, list] = {}
    for coin in COINS:
        rows = conn.execute("""
            SELECT collected_at, price_usd, change_24h
            FROM prices
            WHERE coin_id = ?
            ORDER BY collected_at DESC
            LIMIT 100
        """, (coin,)).fetchall()
        history[coin] = [
            {"t": r[0], "price": r[1], "change": r[2]}
            for r in reversed(rows)
        ]

    # Últimas execuções
    runs = conn.execute("""
        SELECT * FROM pipeline_runs
        ORDER BY ran_at DESC LIMIT 20
    """).fetchall()
    run_cols = [d[0] for d in conn.execute(
        "SELECT * FROM pipeline_runs LIMIT 0"
    ).description]

    total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]

    payload = {
        "meta": {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "total_records": total,
            "coins": COINS,
            "colors": COIN_COLORS,
            "symbols": COIN_SYMBOLS,
        },
        "latest":  latest,
        "history": history,
        "runs":    [dict(zip(run_cols, r)) for r in runs],
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("JSON exportado: %s (%d registros)", json_path, total)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class CryptoPipeline:
    """
    Orquestra coleta → processamento → persistência → exportação.

    Uso típico
    ----------
    >>> pipeline = CryptoPipeline()
    >>> pipeline.run_once()          # executa uma vez
    >>> pipeline.run_loop(interval=300)  # executa a cada 5 minutos
    """

    def __init__(self) -> None:
        self.collector = CoinGeckoCollector()
        self.conn = init_db()

    def run_once(self) -> dict:
        """Executa uma coleta completa."""
        ran_at = datetime.now().isoformat(timespec="seconds")
        try:
            raw     = self.collector.fetch()
            records = process(raw)
            saved   = save_to_db(records, self.conn)
            export_json(self.conn)

            self.conn.execute(
                "INSERT INTO pipeline_runs (ran_at, coins_saved, status) VALUES (?,?,?)",
                (ran_at, saved, "success"),
            )
            self.conn.commit()
            logger.info("✅ Pipeline OK — %d moedas salvas às %s", saved, ran_at)
            return {"status": "success", "saved": saved, "ran_at": ran_at}

        except Exception as e:
            self.conn.execute(
                "INSERT INTO pipeline_runs (ran_at, coins_saved, status, error) VALUES (?,?,?,?)",
                (ran_at, 0, "error", str(e)),
            )
            self.conn.commit()
            logger.error("❌ Pipeline falhou: %s", e)
            return {"status": "error", "error": str(e), "ran_at": ran_at}

    def run_loop(self, interval: int = 300) -> None:
        """Loop contínuo — executa a cada `interval` segundos."""
        logger.info("🚀 Pipeline iniciado — intervalo: %ds", interval)
        while True:
            result = self.run_once()
            if result["status"] == "success":
                print(f"[{result['ran_at']}] ✅ {result['saved']} moedas coletadas")
            else:
                print(f"[{result['ran_at']}] ❌ Erro: {result.get('error')}")
            logger.info("Próxima coleta em %ds...", interval)
            time.sleep(interval)