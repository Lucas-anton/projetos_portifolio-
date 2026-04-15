"""
src/api.py
----------
API REST para o sistema de previsão de demanda.

Endpoints
---------
GET  /health                → Status e métricas do modelo
GET  /products              → Lista de produtos
GET  /products/{id}/history → Histórico de vendas de um produto
GET  /products/{id}/forecast → Previsão para os próximos N dias
GET  /dashboard/summary     → Resumo para o dashboard (KPIs + top produtos)
GET  /dashboard/overview    → Dados completos para o dashboard
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Demand Forecast API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/app", include_in_schema=False)
def frontend(): return FileResponse("web/index.html")

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_forecaster = None
_sales_df: pd.DataFrame | None = None
_products_df: pd.DataFrame | None = None

SALES_PATH    = Path("output/sales.pkl")
PRODUCTS_PATH = Path("output/products.pkl")


def get_forecaster():
    global _forecaster
    if _forecaster is None:
        from model import DemandForecaster
        _forecaster = DemandForecaster()
        _forecaster.load()
    return _forecaster


def get_sales() -> pd.DataFrame:
    global _sales_df
    if _sales_df is None:
        if not SALES_PATH.exists():
            raise RuntimeError("Dados não encontrados. Execute python main.py --train")
        _sales_df = pd.read_pickle(SALES_PATH)
        _sales_df["date"] = pd.to_datetime(_sales_df["date"])
    return _sales_df


def get_products() -> pd.DataFrame:
    global _products_df
    if _products_df is None:
        _products_df = pd.read_pickle(PRODUCTS_PATH)
    return _products_df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root(): return {"name": "Demand Forecast API", "docs": "/docs", "app": "/app"}


@app.get("/health")
def health():
    try:
        fc = get_forecaster()
        art = fc.artifacts
        return {
            "status":      "ok",
            "n_products":  len(art.models) if art else 0,
            "train_end":   art.train_end if art else None,
            "avg_mae":     round(sum(v["mae"] for v in art.metrics.values()) / max(1, len(art.metrics)), 2) if art else None,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/products")
def list_products():
    products = get_products()
    fc = get_forecaster()
    result = []
    for _, row in products.iterrows():
        pid = row["id"]
        stats = fc.artifacts.product_stats.get(pid, {}) if fc.artifacts else {}
        metrics = fc.artifacts.metrics.get(pid, {}) if fc.artifacts else {}
        result.append({
            "id":       pid,
            "name":     row["name"],
            "category": row["category"],
            "price":    row["price"],
            "cost":     row["cost"],
            "avg_daily_sales": round(stats.get("mean", 0), 1),
            "total_sales":     stats.get("total", 0),
            "mae":             metrics.get("mae", 0),
            "mape":            round(metrics.get("mape", 0) * 100, 1),
        })
    return {"total": len(result), "products": result}


@app.get("/products/{product_id}/history")
def get_history(
    product_id: str,
    days: int = Query(90, ge=7, le=365),
):
    sales = get_sales()
    df = sales[sales["product_id"] == product_id].copy()
    if df.empty:
        raise HTTPException(404, f"Produto {product_id} não encontrado.")

    df = df.sort_values("date").tail(days)
    # Agrega por semana para visualização
    df_weekly = df.set_index("date").resample("W")["quantity"].sum().reset_index()

    return {
        "product_id": product_id,
        "daily": [
            {"date": str(r.date), "quantity": int(r.quantity), "is_promo": int(r.is_promo)}
            for _, r in df.iterrows()
        ],
        "weekly": [
            {"date": str(r.date.date()), "quantity": int(r.quantity)}
            for _, r in df_weekly.iterrows()
        ],
    }


@app.get("/products/{product_id}/forecast")
def get_forecast(
    product_id: str,
    horizon: int = Query(30, ge=7, le=90),
):
    fc = get_forecaster()
    sales = get_sales()
    try:
        result = fc.predict(product_id, sales, horizon=horizon)
    except ValueError as e:
        raise HTTPException(404, str(e))

    return {
        "product_id": product_id,
        "horizon":    horizon,
        "mae":        result.mae,
        "mape":       round(result.mape * 100, 2),
        "model":      result.model_name,
        "forecast": [
            {
                "date":      str(r.date),
                "predicted": int(r.predicted),
                "lower":     int(r.lower),
                "upper":     int(r.upper),
            }
            for _, r in result.forecast_df.iterrows()
        ],
    }


@app.get("/dashboard/summary")
def dashboard_summary():
    """KPIs principais para o topo do dashboard."""
    sales  = get_sales()
    products = get_products()
    fc = get_forecaster()

    total_revenue = 0.0
    for _, prod in products.iterrows():
        pid = prod["id"]
        prod_sales = sales[sales["product_id"] == pid]["quantity"].sum()
        total_revenue += prod_sales * prod["price"]

    last_30 = sales[sales["date"] >= (sales["date"].max() - pd.Timedelta(days=30))]
    prev_30 = sales[
        (sales["date"] >= (sales["date"].max() - pd.Timedelta(days=60))) &
        (sales["date"] < (sales["date"].max() - pd.Timedelta(days=30)))
    ]

    last_qty = int(last_30["quantity"].sum())
    prev_qty = int(prev_30["quantity"].sum())
    growth = round((last_qty - prev_qty) / max(1, prev_qty) * 100, 1)

    # Top produtos por vendas últimos 30 dias
    top = (
        last_30.groupby("product_id")["quantity"].sum()
        .sort_values(ascending=False).head(5)
    )
    top_products = []
    for pid, qty in top.items():
        prod_row = products[products["id"] == pid]
        name = prod_row["name"].values[0] if not prod_row.empty else pid
        price = float(prod_row["price"].values[0]) if not prod_row.empty else 0
        top_products.append({
            "id": pid, "name": name,
            "quantity": int(qty), "revenue": round(qty * price, 2),
        })

    return {
        "total_products":    len(products),
        "total_revenue":     round(total_revenue, 2),
        "last_30d_units":    last_qty,
        "units_growth":      growth,
        "avg_daily_units":   round(last_qty / 30, 1),
        "top_products":      top_products,
        "date_range":        {
            "start": str(sales["date"].min().date()),
            "end":   str(sales["date"].max().date()),
        },
    }


@app.get("/dashboard/overview")
def dashboard_overview():
    """Dados completos para o dashboard principal."""
    sales    = get_sales()
    products = get_products()

    # Série temporal agregada (todas as categorias)
    daily_total = (
        sales.groupby("date")["quantity"].sum()
        .reset_index()
        .sort_values("date")
    )
    daily_total["date"] = daily_total["date"].dt.strftime("%Y-%m-%d")

    # Por categoria
    cat_map = dict(zip(products["id"], products["category"]))
    sales["category"] = sales["product_id"].map(cat_map)
    cat_sales = (
        sales.groupby("category")["quantity"].sum()
        .sort_values(ascending=False).to_dict()
    )

    # Por dia da semana
    dow_sales = (
        sales.groupby("day_of_week")["quantity"].mean()
        .round(1).to_dict()
    )

    # Por mês
    monthly = (
        sales.groupby("month")["quantity"].sum()
        .to_dict()
    )

    return {
        "daily_total": daily_total.to_dict(orient="records"),
        "category_sales": cat_sales,
        "dow_sales": dow_sales,
        "monthly_sales": monthly,
    }