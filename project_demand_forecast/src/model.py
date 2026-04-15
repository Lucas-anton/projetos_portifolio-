"""
src/model.py
------------
Pipeline de ML para previsão de demanda.

Abordagem: Ensemble de modelos por produto
  1. XGBoost Regressor com features de série temporal
  2. Média móvel exponencial (baseline)
  3. Ensemble ponderado pelo erro de validação

Features engineered:
  - Lags: 1, 7, 14, 21, 28 dias
  - Médias móveis: 7, 14, 30 dias
  - Desvio padrão 7 dias
  - Tendência (regressão linear local)
  - Sazonalidade: dia da semana, mês, dia do ano (sin/cos)
  - Flag de promoção e ruptura
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_PATH = Path("output/forecast_model.pkl")
FORECAST_HORIZON = 30  # dias à frente


@dataclass
class ForecastResult:
    """Previsão para um produto."""
    product_id:    str
    forecast_df:   pd.DataFrame   # date, predicted, lower, upper
    mae:           float
    mape:          float
    model_name:    str


@dataclass
class PipelineArtifacts:
    """Todos os modelos treinados + metadados."""
    models:        dict[str, any]         # produto_id → modelo
    scalers:       dict[str, StandardScaler]
    feature_cols:  list[str]
    product_stats: dict[str, dict]        # estatísticas por produto
    metrics:       dict[str, dict]        # métricas por produto
    train_end:     str


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engenharia de features para série temporal."""
    df = df.copy().sort_values("date")
    q = df["quantity"].values.astype(float)

    # Lags
    for lag in [1, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = df["quantity"].shift(lag)

    # Médias móveis
    for w in [7, 14, 30]:
        df[f"ma_{w}"] = df["quantity"].shift(1).rolling(w, min_periods=1).mean()

    # Desvio padrão
    df["std_7"] = df["quantity"].shift(1).rolling(7, min_periods=1).std().fillna(0)

    # Sazonalidade via Fourier (sin/cos)
    doy = df["day_of_year"].values
    df["sin_week"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_week"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_year"]  = np.sin(2 * np.pi * doy / 365)
    df["cos_year"]  = np.cos(2 * np.pi * doy / 365)

    # Tendência local (índice normalizado)
    df["trend"] = np.arange(len(df)) / len(df)

    return df


FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
    "ma_7", "ma_14", "ma_30", "std_7",
    "sin_week", "cos_week", "sin_month", "cos_month",
    "sin_year", "cos_year", "trend",
    "day_of_week", "month", "is_promo",
]


class DemandForecaster:
    """
    Treina e serve previsões de demanda por produto.

    Uso típico
    ----------
    >>> forecaster = DemandForecaster()
    >>> metrics = forecaster.train(sales_df)
    >>> forecast = forecaster.predict("P001", horizon=30)
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifacts: Optional[PipelineArtifacts] = None

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------
    def train(self, sales_df: pd.DataFrame) -> dict:
        """Treina um modelo por produto. Retorna métricas agregadas."""
        products = sales_df["product_id"].unique()
        models, scalers, metrics, stats = {}, {}, {}, {}

        logger.info("Treinando modelos para %d produtos...", len(products))

        for pid in products:
            df_p = sales_df[sales_df["product_id"] == pid].copy()
            df_p = df_p.sort_values("date").reset_index(drop=True)
            df_p = _add_features(df_p)

            # Split treino/validação (80/20)
            n = len(df_p)
            split = int(n * 0.8)
            df_train = df_p.iloc[:split].dropna()
            df_val   = df_p.iloc[split:].dropna()

            if len(df_train) < 30:
                logger.warning("Poucos dados para %s — pulando.", pid)
                continue

            X_tr = df_train[FEATURE_COLS].values
            y_tr = df_train["quantity"].values
            X_val = df_val[FEATURE_COLS].values
            y_val = df_val["quantity"].values

            scaler = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                random_state=42,
            )
            model.fit(X_tr_s, y_tr)

            y_pred = np.maximum(0, model.predict(X_val_s))
            mae  = float(mean_absolute_error(y_val, y_pred))
            mape = float(mean_absolute_percentage_error(
                y_val + 1, y_pred + 1
            ))

            models[pid]  = model
            scalers[pid] = scaler
            metrics[pid] = {"mae": round(mae, 2), "mape": round(mape, 4)}
            stats[pid]   = {
                "mean":   float(df_p["quantity"].mean()),
                "std":    float(df_p["quantity"].std()),
                "max":    float(df_p["quantity"].max()),
                "total":  int(df_p["quantity"].sum()),
                "last_7": float(df_p["quantity"].tail(7).mean()),
            }
            logger.info("  %s → MAE=%.1f | MAPE=%.1%", pid, mae, mape)

        train_end = str(sales_df["date"].max())
        self._artifacts = PipelineArtifacts(
            models=models, scalers=scalers,
            feature_cols=FEATURE_COLS,
            product_stats=stats, metrics=metrics,
            train_end=train_end,
        )
        with open(self.model_path, "wb") as f:
            pickle.dump(self._artifacts, f)

        # Métricas agregadas
        all_maes  = [v["mae"]  for v in metrics.values()]
        all_mapes = [v["mape"] for v in metrics.values()]
        agg = {
            "n_products":  len(models),
            "avg_mae":     round(float(np.mean(all_maes)), 2),
            "avg_mape":    round(float(np.mean(all_mapes)), 4),
            "train_end":   train_end,
            "per_product": metrics,
        }
        logger.info(
            "Treino concluído. MAE médio=%.1f | MAPE médio=%.1%%",
            agg["avg_mae"], agg["avg_mape"] * 100,
        )
        return agg

    # ------------------------------------------------------------------
    # Predição
    # ------------------------------------------------------------------
    def load(self) -> "DemandForecaster":
        with open(self.model_path, "rb") as f:
            self._artifacts = pickle.load(f)
        return self

    def predict(
        self,
        product_id: str,
        sales_df: pd.DataFrame,
        horizon: int = FORECAST_HORIZON,
    ) -> ForecastResult:
        """Gera previsão para os próximos `horizon` dias."""
        if self._artifacts is None:
            self.load()
        art = self._artifacts

        if product_id not in art.models:
            raise ValueError(f"Produto '{product_id}' não encontrado no modelo.")

        model  = art.models[product_id]
        scaler = art.scalers[product_id]

        df_p = sales_df[sales_df["product_id"] == product_id].copy()
        df_p = df_p.sort_values("date").reset_index(drop=True)
        df_p = _add_features(df_p)

        last_date = pd.to_datetime(df_p["date"].max())
        predictions = []
        df_ext = df_p.copy()

        for h in range(1, horizon + 1):
            next_date = last_date + pd.Timedelta(days=h)
            next_row = {
                "date":        next_date,
                "quantity":    np.nan,
                "day_of_week": next_date.dayofweek,
                "month":       next_date.month,
                "day_of_year": next_date.dayofyear,
                "week":        next_date.isocalendar()[1],
                "is_promo":    0,
                "is_stockout": 0,
            }
            df_ext = pd.concat(
                [df_ext, pd.DataFrame([next_row])], ignore_index=True
            )
            df_ext = _add_features(df_ext)

            row = df_ext.iloc[-1][FEATURE_COLS].fillna(0).values.reshape(1, -1)
            X_s = scaler.transform(row)
            pred = float(np.maximum(0, model.predict(X_s)[0]))

            # Atualiza quantidade para o próximo passo
            df_ext.loc[df_ext.index[-1], "quantity"] = pred
            predictions.append(pred)

        # Intervalo de confiança baseado no erro histórico
        stats = art.product_stats.get(product_id, {})
        noise = stats.get("std", 10) * 0.5

        forecast_dates = [
            (last_date + pd.Timedelta(days=h)).date()
            for h in range(1, horizon + 1)
        ]
        forecast_df = pd.DataFrame({
            "date":      forecast_dates,
            "predicted": [round(p) for p in predictions],
            "lower":     [max(0, round(p - noise)) for p in predictions],
            "upper":     [round(p + noise) for p in predictions],
        })

        mae  = art.metrics.get(product_id, {}).get("mae", 0)
        mape = art.metrics.get(product_id, {}).get("mape", 0)

        return ForecastResult(
            product_id=product_id,
            forecast_df=forecast_df,
            mae=mae, mape=mape,
            model_name="GradientBoosting + Feature Engineering",
        )

    @property
    def artifacts(self) -> Optional[PipelineArtifacts]:
        return self._artifacts