"""
src/api.py
----------
API REST para previsão de aluguel de imóveis brasileiros.

Endpoints
---------
GET  /                  → Informações da API
GET  /health            → Status e versão do modelo
POST /predict           → Previsão de aluguel para um imóvel
POST /predict/batch     → Previsão em lote (múltiplos imóveis)
GET  /model/info        → Métricas e features do modelo
GET  /cities            → Cidades suportadas

Documentação interativa automática:
  http://localhost:8000/docs      (Swagger UI)
  http://localhost:8000/redoc     (ReDoc)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cidades suportadas (mesmas do trainer)
# ---------------------------------------------------------------------------
SUPPORTED_CITIES = [
    "São Paulo", "Rio de Janeiro", "Belo Horizonte",
    "Porto Alegre", "Curitiba", "Campinas",
]

MODEL_PATH = Path("output/model.joblib")
API_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Schemas Pydantic — validação automática de entrada e saída
# ---------------------------------------------------------------------------
class ImovelInput(BaseModel):
    """Dados de entrada para previsão de aluguel."""

    city: str = Field(
        ...,
        description="Cidade do imóvel",
        examples=["São Paulo"],
    )
    area: int = Field(
        ..., ge=10, le=1000,
        description="Área total em m²",
        examples=[75],
    )
    rooms: int = Field(
        ..., ge=1, le=10,
        description="Número de quartos",
        examples=[2],
    )
    bathrooms: int = Field(
        ..., ge=1, le=10,
        description="Número de banheiros",
        examples=[1],
    )
    parking_spaces: int = Field(
        default=0, ge=0, le=10,
        description="Vagas de garagem",
        examples=[1],
    )
    floor: int = Field(
        default=0, ge=0, le=50,
        description="Andar do imóvel (0 = térreo)",
        examples=[3],
    )
    animal: bool = Field(
        default=False,
        description="Aceita animais?",
        examples=[True],
    )
    furniture: bool = Field(
        default=False,
        description="Imóvel mobiliado?",
        examples=[False],
    )
    hoa: float = Field(
        default=0.0, ge=0,
        description="Condomínio mensal (R$)",
        examples=[450.0],
    )
    property_tax: float = Field(
        default=0.0, ge=0,
        description="IPTU mensal (R$)",
        examples=[120.0],
    )
    fire_insurance: float = Field(
        default=0.0, ge=0,
        description="Seguro incêndio mensal (R$)",
        examples=[25.0],
    )

    @field_validator("city")
    @classmethod
    def validate_city(cls, v: str) -> str:
        if v not in SUPPORTED_CITIES:
            raise ValueError(
                f"Cidade '{v}' não suportada. "
                f"Cidades disponíveis: {SUPPORTED_CITIES}"
            )
        return v


class PredictionResponse(BaseModel):
    """Resposta de previsão."""
    predicted_rent: float = Field(description="Aluguel previsto em R$")
    predicted_rent_formatted: str = Field(description="Aluguel formatado (ex: R$ 2.500,00)")
    confidence_range: dict[str, float] = Field(
        description="Intervalo de confiança estimado (±15%)"
    )
    input_summary: dict = Field(description="Resumo dos dados de entrada")


class BatchInput(BaseModel):
    """Entrada para previsão em lote."""
    imoveis: list[ImovelInput] = Field(
        ..., min_length=1, max_length=100,
        description="Lista de imóveis (máx. 100 por requisição)",
    )


class BatchResponse(BaseModel):
    """Resposta de previsão em lote."""
    total: int
    predictions: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str
    supported_cities: list[str]


class RootResponse(BaseModel):
    """Resposta do endpoint raiz."""
    name: str
    version: str
    docs: str
    endpoints: dict[str, str]


# ---------------------------------------------------------------------------
# Inicialização da API
# ---------------------------------------------------------------------------
app = FastAPI(
    title="🏠 API de Previsão de Aluguel — Imóveis Brasil",
    description="""
API REST para previsão de aluguel de imóveis em cidades brasileiras.

## Como usar

1. Consulte `/cities` para ver as cidades disponíveis
2. Envie os dados do imóvel para `/predict`
3. Receba a previsão de aluguel em R$

## Modelo
- **Algoritmo**: Random Forest Regressor
- **Features**: área, quartos, banheiros, vagas, andar, mobília, condomínio, IPTU
- **Dataset**: Brazilian Houses Dataset (distribuições reais)
    """,
    version=API_VERSION,
    contact={
        "name": "Portfólio Data Science",
        "url": "https://github.com/seu-usuario",
    },
    license_info={"name": "MIT"},
)

# CORS — permite chamadas do browser e de outros domínios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega modelo em memória na inicialização
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Modelo não encontrado. Execute 'python main.py --train' primeiro.",
            )
        _pipeline = joblib.load(MODEL_PATH)
        logger.info("Modelo carregado de %s", MODEL_PATH)
    return _pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_model=RootResponse, tags=["Info"])
def root():
    """Informações gerais da API."""
    return RootResponse(
        name="API de Previsão de Aluguel — Imóveis Brasil",
        version=API_VERSION,
        docs="/docs",
        endpoints={
            "predict":       "POST /predict",
            "batch_predict": "POST /predict/batch",
            "model_info":    "GET /model/info",
            "cities":        "GET /cities",
            "health":        "GET /health",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Verifica status da API e do modelo."""
    return HealthResponse(
        status="ok",
        model_loaded=MODEL_PATH.exists(),
        api_version=API_VERSION,
        supported_cities=SUPPORTED_CITIES,
    )


@app.get("/cities", tags=["Info"])
def list_cities():
    """Lista todas as cidades suportadas pelo modelo."""
    return {
        "total": len(SUPPORTED_CITIES),
        "cities": SUPPORTED_CITIES,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Previsão"])
def predict(imovel: ImovelInput):
    """
    Prevê o valor de aluguel de um imóvel.

    Retorna o valor previsto em R$, com intervalo de confiança de ±15%.
    """
    pipeline = get_pipeline()

    df = pd.DataFrame([{
        "city":           imovel.city,
        "area":           imovel.area,
        "rooms":          imovel.rooms,
        "bathrooms":      imovel.bathrooms,
        "parking_spaces": imovel.parking_spaces,
        "floor":          imovel.floor,
        "animal":         int(imovel.animal),
        "furniture":      int(imovel.furniture),
        "hoa":            imovel.hoa,
        "property_tax":   imovel.property_tax,
        "fire_insurance": imovel.fire_insurance,
    }])

    predicted = float(pipeline.predict(df)[0])
    predicted = max(500.0, round(predicted, 2))

    margin = predicted * 0.15
    return PredictionResponse(
        predicted_rent=predicted,
        predicted_rent_formatted=f"R$ {predicted:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        confidence_range={
            "min": round(predicted - margin, 2),
            "max": round(predicted + margin, 2),
        },
        input_summary={
            "city":     imovel.city,
            "area_m2":  imovel.area,
            "rooms":    imovel.rooms,
            "furnished": imovel.furniture,
        },
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Previsão"])
def predict_batch(batch: BatchInput):
    """
    Prevê o aluguel de múltiplos imóveis em uma única requisição.

    Máximo de 100 imóveis por chamada.
    """
    pipeline = get_pipeline()

    records = [
        {
            "city":           im.city,
            "area":           im.area,
            "rooms":          im.rooms,
            "bathrooms":      im.bathrooms,
            "parking_spaces": im.parking_spaces,
            "floor":          im.floor,
            "animal":         int(im.animal),
            "furniture":      int(im.furniture),
            "hoa":            im.hoa,
            "property_tax":   im.property_tax,
            "fire_insurance":  im.fire_insurance,
        }
        for im in batch.imoveis
    ]

    df = pd.DataFrame(records)
    predictions = pipeline.predict(df)

    results = []
    for i, (im, pred) in enumerate(zip(batch.imoveis, predictions)):
        pred = max(500.0, round(float(pred), 2))
        results.append({
            "index":          i,
            "city":           im.city,
            "area":           im.area,
            "rooms":          im.rooms,
            "predicted_rent": pred,
            "formatted":      f"R$ {pred:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        })

    return BatchResponse(total=len(results), predictions=results)


@app.get("/model/info", tags=["Modelo"])
def model_info():
    """Retorna informações técnicas e métricas do modelo treinado."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Modelo não encontrado.")

    info_path = Path("output/metrics.json")
    if info_path.exists():
        import json
        with open(info_path) as f:
            return json.load(f)

    return {
        "algorithm":  "Random Forest Regressor",
        "features":   [
            "city", "area", "rooms", "bathrooms", "parking_spaces",
            "floor", "animal", "furniture", "hoa", "property_tax", "fire_insurance",
        ],
        "target":      "rent_amount (R$)",
        "note":        "Execute main.py --train para ver métricas detalhadas.",
    }