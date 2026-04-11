"""
src/api.py
----------
API REST para o detector de anomalias financeiras.

Endpoints
---------
GET  /health           → Status e métricas do modelo
GET  /transactions     → Lista transações com scores (paginada)
GET  /anomalies        → Só transações anômalas
POST /predict          → Classifica uma nova transação
GET  /stats            → Estatísticas agregadas para o dashboard
GET  /timeline         → Volume de anomalias por hora/dia
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Serve frontend
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/app", include_in_schema=False)
def frontend(): return FileResponse("web/index.html")

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_detector = None
_df: pd.DataFrame | None = None

DATA_PATH = Path("output/transactions.pkl")


def get_detector():
    global _detector
    if _detector is None:
        from model import HybridAnomalyDetector
        _detector = HybridAnomalyDetector()
        _detector.load()
    return _detector


def get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        if not DATA_PATH.exists():
            raise RuntimeError("Dataset não encontrado. Execute python main.py --train")
        raw = pd.read_pickle(DATA_PATH)
        _df = get_detector().predict(raw)
        _df["timestamp"] = pd.to_datetime(_df["timestamp"])
    return _df


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Serializa DataFrame para JSON de forma segura."""
    out = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if hasattr(val, 'isoformat'):
                record[col] = val.isoformat()
            elif hasattr(val, 'item'):
                record[col] = val.item()
            elif pd.isna(val) if not isinstance(val, str) else False:
                record[col] = None
            else:
                record[col] = str(val) if not isinstance(val, (int, float, str, bool)) else val
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TransactionInput(BaseModel):
    amount:             float = Field(..., gt=0, examples=[1250.00])
    hour:               int   = Field(..., ge=0, le=23, examples=[3])
    day_of_week:        int   = Field(..., ge=0, le=6, examples=[1])
    distance_from_home: float = Field(default=0, examples=[450.0])
    n_transactions_1h:  int   = Field(default=1, examples=[8])
    avg_amount_30d:     float = Field(default=200.0, examples=[180.0])
    is_foreign:         int   = Field(default=0, examples=[0])
    is_night:           int   = Field(default=0, examples=[1])
    amount_ratio:       float = Field(default=1.0, examples=[6.9])
    v1: float = Field(default=0.0); v2: float = Field(default=0.0)
    v3: float = Field(default=0.0); v4: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"name": "Fraud Detection API", "docs": "/docs", "app": "/app"}


@app.get("/health")
def health():
    try:
        det = get_detector()
        return {"status": "ok", "model": "hybrid", "metrics": det.metrics}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/transactions")
def get_transactions(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    risk: str = Query(None),
):
    df = get_df()
    if risk:
        df = df[df["risk_level"].astype(str) == risk]
    total = len(df)
    start = (page - 1) * size
    page_df = df.sort_values("timestamp", ascending=False).iloc[start:start+size]
    return {"total": total, "page": page, "size": size,
            "data": df_to_records(page_df)}


@app.get("/anomalies")
def get_anomalies(limit: int = Query(100, le=500)):
    df = get_df()
    anomalies = df[df["is_anomaly"] == 1].sort_values(
        "hybrid_score", ascending=False
    ).head(limit)
    return {"total": len(anomalies), "data": df_to_records(anomalies)}


@app.post("/predict")
def predict_transaction(tx: TransactionInput):
    det = get_detector()
    import pandas as pd
    row = pd.DataFrame([tx.model_dump()])
    result = det.predict(row).iloc[0]
    risk = str(result["risk_level"])
    return {
        "is_anomaly":   bool(result["is_anomaly"]),
        "risk_level":   risk,
        "hybrid_score": float(result["hybrid_score"]),
        "if_score":     float(result["if_score"]),
        "ae_score":     float(result["ae_score"]),
        "recommendation": {
            "Crítico": "🚨 Bloquear transação imediatamente",
            "Alto":    "⚠️  Solicitar autenticação adicional",
            "Médio":   "👁️  Monitorar e notificar cliente",
            "Baixo":   "✅  Aprovar — risco baixo",
        }.get(risk, "✅ Aprovar"),
    }


@app.get("/stats")
def get_stats():
    df = get_df()
    total = len(df)
    n_anomaly = int(df["is_anomaly"].sum())
    risk_dist = df["risk_level"].astype(str).value_counts().to_dict()
    cat_anomaly = (
        df[df["is_anomaly"] == 1]["category"]
        .value_counts().head(5).to_dict()
    )
    avg_fraud_amount = float(df[df["is_anomaly"] == 1]["amount"].mean() or 0)
    avg_normal_amount = float(df[df["is_anomaly"] == 0]["amount"].mean() or 0)
    total_at_risk = float(df[df["is_anomaly"] == 1]["amount"].sum())

    return {
        "total_transactions": total,
        "total_anomalies":    n_anomaly,
        "anomaly_rate":       round(n_anomaly / total * 100, 2),
        "total_at_risk":      round(total_at_risk, 2),
        "avg_fraud_amount":   round(avg_fraud_amount, 2),
        "avg_normal_amount":  round(avg_normal_amount, 2),
        "risk_distribution":  risk_dist,
        "top_fraud_categories": cat_anomaly,
    }


@app.get("/timeline")
def get_timeline():
    df = get_df().copy()
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour

    # Por dia
    daily = df.groupby("date").agg(
        total=("transaction_id", "count"),
        anomalies=("is_anomaly", "sum"),
        volume=("amount", "sum"),
    ).reset_index()

    # Por hora
    hourly = df.groupby("hour").agg(
        total=("transaction_id", "count"),
        anomalies=("is_anomaly", "sum"),
    ).reset_index()

    return {
        "daily":  daily.to_dict(orient="records"),
        "hourly": hourly.to_dict(orient="records"),
    }