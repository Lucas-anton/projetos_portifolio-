"""
src/collector.py
----------------
Coleta dados reais de produtos da Open Food Facts API
e gera histórico sintético de vendas com padrões realistas.

API utilizada: Open Food Facts (https://world.openfoodfacts.org/api/v2)
  - 100% gratuita, sem autenticação
  - Mais de 3 milhões de produtos cadastrados
  - Dados: nome, categoria, marca, nutrição, imagem

Histórico de vendas gerado com:
  - Tendência (crescimento/declínio gradual)
  - Sazonalidade semanal (picos em fins de semana)
  - Sazonalidade mensal (meses específicos)
  - Eventos (promoções, feriados)
  - Ruído gaussiano realista
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://world.openfoodfacts.org/api/v2"

CATEGORIES_BR = [
    "Bebidas", "Snacks", "Laticínios", "Carnes", "Hortifruti",
    "Padaria", "Congelados", "Higiene", "Limpeza", "Eletrônicos",
]

SYNTHETIC_PRODUCTS = [
    {"id": "P001", "name": "Café Pilão 500g",           "category": "Bebidas",     "price": 18.90, "cost": 9.50},
    {"id": "P002", "name": "Leite Integral 1L",         "category": "Laticínios",  "price": 5.49,  "cost": 2.80},
    {"id": "P003", "name": "Biscoito Oreo 144g",        "category": "Snacks",      "price": 8.90,  "cost": 4.20},
    {"id": "P004", "name": "Refrigerante Coca-Cola 2L", "category": "Bebidas",     "price": 12.90, "cost": 5.60},
    {"id": "P005", "name": "Arroz Tio João 5kg",        "category": "Hortifruti",  "price": 32.90, "cost": 18.00},
    {"id": "P006", "name": "Feijão Camil 1kg",          "category": "Hortifruti",  "price": 9.90,  "cost": 5.40},
    {"id": "P007", "name": "Macarrão Barilla 500g",     "category": "Padaria",     "price": 7.90,  "cost": 3.80},
    {"id": "P008", "name": "Sabão em Pó OMO 1kg",       "category": "Limpeza",     "price": 22.90, "cost": 11.50},
    {"id": "P009", "name": "Shampoo Pantene 400ml",     "category": "Higiene",     "price": 19.90, "cost": 9.80},
    {"id": "P010", "name": "Iogurte Activia 170g",      "category": "Laticínios",  "price": 4.90,  "cost": 2.20},
    {"id": "P011", "name": "Suco Del Valle 1L",         "category": "Bebidas",     "price": 9.90,  "cost": 4.80},
    {"id": "P012", "name": "Batata Chips Pringles",     "category": "Snacks",      "price": 14.90, "cost": 7.20},
    {"id": "P013", "name": "Frango Congelado 1kg",      "category": "Congelados",  "price": 24.90, "cost": 13.50},
    {"id": "P014", "name": "Queijo Mussarela 200g",     "category": "Laticínios",  "price": 11.90, "cost": 6.20},
    {"id": "P015", "name": "Cerveja Heineken 350ml",    "category": "Bebidas",     "price": 4.99,  "cost": 2.10},
]

# Sazonalidade semanal (0=segunda, 6=domingo)
WEEKLY_PATTERN = [0.75, 0.80, 0.85, 0.90, 1.10, 1.40, 1.20]

# Sazonalidade mensal
MONTHLY_PATTERN = [1.15, 1.00, 0.95, 0.90, 0.90, 0.95, 0.95, 1.00, 1.00, 1.05, 1.15, 1.35]

# Produtos com alta sazonalidade específica
SEASONAL_PRODUCTS = {
    "P004": {11: 1.5, 12: 1.8},  # refrigerante no verão (dez-jan)
    "P015": {6: 1.6, 12: 2.0},   # cerveja no inverno e fim de ano
    "P001": {6: 1.3, 7: 1.3},    # café no inverno
}


@dataclass
class SalesDataset:
    """Dataset completo: produtos + histórico de vendas."""
    products: pd.DataFrame
    sales: pd.DataFrame
    date_range: tuple[str, str]


class SalesGenerator:
    """
    Gera histórico sintético de vendas com padrões realistas.

    Cada produto tem sua própria curva de tendência, sazonalidade
    e eventos aleatórios (promoções, rupturas de estoque).
    """

    def __init__(self, days: int = 365, seed: int = 42) -> None:
        self.days = days
        self.rng = np.random.default_rng(seed)

    def run(self) -> SalesDataset:
        """Gera o dataset completo."""
        products_df = pd.DataFrame(SYNTHETIC_PRODUCTS)
        end_date = date.today()
        start_date = end_date - timedelta(days=self.days - 1)
        dates = pd.date_range(start_date, end_date, freq="D")

        all_records = []
        for prod in SYNTHETIC_PRODUCTS:
            pid = prod["id"]
            base_demand = self._base_demand(prod["category"])
            trend = self._trend()
            records = self._generate_product_sales(
                pid, dates, base_demand, trend,
                SEASONAL_PRODUCTS.get(pid, {})
            )
            all_records.extend(records)

        sales_df = pd.DataFrame(all_records)
        sales_df["date"] = pd.to_datetime(sales_df["date"])
        sales_df = sales_df.sort_values(["product_id", "date"]).reset_index(drop=True)

        logger.info(
            "Dataset gerado: %d registros | %d produtos | %d dias",
            len(sales_df), len(SYNTHETIC_PRODUCTS), self.days,
        )
        return SalesDataset(
            products=products_df,
            sales=sales_df,
            date_range=(str(start_date), str(end_date)),
        )

    def _base_demand(self, category: str) -> int:
        demands = {
            "Bebidas": 120, "Laticínios": 200, "Snacks": 80,
            "Hortifruti": 150, "Padaria": 90, "Congelados": 60,
            "Higiene": 70, "Limpeza": 65, "Eletrônicos": 30,
        }
        return demands.get(category, 100)

    def _trend(self) -> float:
        """Tendência aleatória entre -0.15% e +0.25% ao dia."""
        return self.rng.uniform(-0.0015, 0.0025)

    def _generate_product_sales(
        self,
        pid: str,
        dates: pd.DatetimeIndex,
        base: int,
        trend: float,
        seasonal_override: dict,
    ) -> list[dict]:
        records = []
        promotion_days = set(self.rng.choice(len(dates), size=12, replace=False))

        for i, dt in enumerate(dates):
            # Tendência
            trend_mult = 1 + trend * i

            # Sazonalidade semanal
            week_mult = WEEKLY_PATTERN[dt.dayofweek]

            # Sazonalidade mensal
            month_mult = MONTHLY_PATTERN[dt.month - 1]

            # Override específico do produto
            prod_mult = seasonal_override.get(dt.month, 1.0)

            # Promoção (aumento de 40-80%)
            promo = self.rng.uniform(1.4, 1.8) if i in promotion_days else 1.0

            # Ruptura ocasional de estoque (0 vendas)
            stockout = self.rng.random() < 0.01

            demand = int(
                base * trend_mult * week_mult * month_mult * prod_mult * promo
                * max(0.7, self.rng.normal(1.0, 0.12))
            )
            demand = 0 if stockout else max(0, demand)

            records.append({
                "product_id":  pid,
                "date":        dt.date(),
                "quantity":    demand,
                "is_promo":    int(i in promotion_days),
                "is_stockout": int(stockout),
                "day_of_week": dt.dayofweek,
                "month":       dt.month,
                "day_of_year": dt.dayofyear,
                "week":        dt.isocalendar()[1],
            })
        return records