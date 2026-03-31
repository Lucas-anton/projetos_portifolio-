"""
src/api.py
----------
API REST para classificação de sentimentos em tempo real.

Endpoints
---------
GET  /              → Info da API
GET  /health        → Status do modelo
POST /predict       → Classifica um texto
POST /predict/batch → Classifica múltiplos textos (máx. 100)
GET  /model/info    → Métricas do modelo treinado
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"
METRICS_PATH = Path("output/metrics.json")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TextInput(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000,
                      examples=["This game is absolutely broken after the patch!"])


class BatchInput(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100,
                              examples=[["Amazing update!", "Game is trash now."]])


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    lexicon_signals: list[str]
    sentiment_emoji: str


class BatchResponse(BaseModel):
    total: int
    results: list[SentimentResponse]
    summary: dict[str, int]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="🎮 Games Sentiment Analysis API",
    description="""
API REST para análise de sentimentos em textos de games e e-sports.

## Como usar
1. `POST /predict` com qualquer comentário de game
2. Receba classificação: **positive**, **negative** ou **neutral**

## Modelo
- **Algoritmo**: Logistic Regression + TF-IDF (1-2 grams)
- **Dados**: Comentários reais de subreddits de games
- **Classes**: positive · negative · neutral
    """,
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None

EMOJI_MAP = {"positive": "😊", "negative": "😤", "neutral": "😐"}


def get_model():
    global _model
    if _model is None:
        from model import SentimentModel
        _model = SentimentModel()
        try:
            _model.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Modelo não encontrado. Execute 'python main.py --train' primeiro.",
            )
    return _model


def to_response(result) -> SentimentResponse:
    return SentimentResponse(
        text=result.text[:200],
        sentiment=result.sentiment,
        confidence=result.confidence,
        probabilities=result.probabilities,
        lexicon_signals=result.lexicon_signals,
        sentiment_emoji=EMOJI_MAP.get(result.sentiment, "❓"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Games Sentiment Analysis API",
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": {
            "predict":       "POST /predict",
            "batch_predict": "POST /predict/batch",
            "model_info":    "GET /model/info",
            "health":        "GET /health",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    model_ready = Path("output/sentiment_model.joblib").exists()
    return {
        "status": "ok" if model_ready else "model_not_trained",
        "model_loaded": model_ready,
        "api_version": API_VERSION,
    }


@app.post("/predict", response_model=SentimentResponse, tags=["Análise"])
def predict(body: TextInput):
    """Classifica o sentimento de um texto."""
    model = get_model()
    result = model.predict(body.text)
    return to_response(result)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Análise"])
def predict_batch(body: BatchInput):
    """Classifica múltiplos textos (máx. 100)."""
    model = get_model()
    results = model.predict_batch(body.texts)
    responses = [to_response(r) for r in results]
    summary = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        summary[r.sentiment] = summary.get(r.sentiment, 0) + 1
    return BatchResponse(total=len(responses), results=responses, summary=summary)


@app.get("/model/info", tags=["Modelo"])
def model_info():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {
        "algorithm": "Logistic Regression + TF-IDF",
        "note": "Execute main.py --train para métricas detalhadas.",
    }