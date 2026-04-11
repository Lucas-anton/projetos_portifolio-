"""
src/model.py
------------
Detector de anomalias híbrido para transações de cartão de crédito.

Pipeline
--------
1. Geração de dados sintéticos com distribuições reais do
   Credit Card Fraud Dataset (Kaggle / ULB Machine Learning Group)

2. Isolation Forest
   - Detecta anomalias globais por isolamento de pontos raros
   - Rápido, sem suposição de distribuição

3. Autoencoder (NumPy puro — sem TensorFlow/PyTorch)
   - Aprende a reconstruir transações normais
   - Alto erro de reconstrução = anomalia
   - Mais sensível a padrões locais

4. Ensemble híbrido
   - Score final = α × IF_score + (1-α) × AE_score
   - Threshold adaptativo baseado no percentil configurado
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_PATH = Path("output/model.pkl")
DATA_PATH  = Path("output/transactions.pkl")

# ---------------------------------------------------------------------------
# Gerador de dataset sintético realista
# ---------------------------------------------------------------------------
CATEGORIES = ["Supermercado", "Restaurante", "E-commerce", "Combustível",
               "Farmácia", "Eletrônicos", "Viagem", "Entretenimento",
               "Saúde", "Educação"]

MERCHANTS = {
    "Supermercado":   ["Carrefour", "Extra", "Pão de Açúcar", "Atacadão"],
    "Restaurante":    ["iFood", "McDonald's", "Outback", "Subway"],
    "E-commerce":     ["Mercado Livre", "Amazon", "Shopee", "Magazine Luiza"],
    "Combustível":    ["Shell", "Ipiranga", "Posto BR", "Petrobras"],
    "Farmácia":       ["Drogasil", "Ultrafarma", "Panvel", "Pacheco"],
    "Eletrônicos":    ["Apple Store", "Samsung", "Fast Shop", "Kabum"],
    "Viagem":         ["Booking", "Latam", "Gol", "Azul"],
    "Entretenimento": ["Netflix", "Spotify", "Steam", "Cinema"],
    "Saúde":          ["Unimed", "Fleury", "DASA", "Einstein"],
    "Educação":       ["Coursera", "Udemy", "Alura", "Descomplica"],
}

FRAUD_PATTERNS = [
    "Valor muito alto para categoria",
    "Múltiplas transações em sequência",
    "Horário atípico (madrugada)",
    "Localização inconsistente",
    "Merchant desconhecido",
    "Valor fracionado (structuring)",
]


def generate_transactions(
    n_normal: int = 2000,
    n_fraud: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Gera dataset sintético com distribuições baseadas no
    Credit Card Fraud Dataset (ULB / Kaggle).

    Features geradas
    ----------------
    amount, hour, day_of_week, category, merchant,
    distance_from_home, n_transactions_1h, avg_amount_30d,
    is_foreign, is_night, amount_ratio, v1..v8 (PCA-like features)
    """
    rng = np.random.default_rng(seed)

    def make_normal(n):
        cats = rng.choice(CATEGORIES, n)
        _hp = [1,1,1,1,1,2,4,6,7,7,7,7,7,7,7,7,6,6,5,5,4,3,2,1]
        _hp = [x/sum(_hp) for x in _hp]
        hours = rng.choice(range(24), n, p=_hp)
        amounts = np.clip(rng.lognormal(4.2, 0.9, n), 5, 3000).round(2)
        return pd.DataFrame({
            "amount":              amounts,
            "hour":                hours,
            "day_of_week":         rng.integers(0, 7, n),
            "category":            cats,
            "merchant":            [rng.choice(MERCHANTS[c]) for c in cats],
            "distance_from_home":  np.clip(rng.exponential(15, n), 0, 200).round(1),
            "n_transactions_1h":   rng.choice([1,2,3], n, p=[0.7,0.2,0.1]),
            "avg_amount_30d":      np.clip(rng.normal(180, 80, n), 20, 800).round(2),
            "is_foreign":          rng.choice([0,1], n, p=[0.95,0.05]),
            "is_night":            ((hours >= 22) | (hours <= 5)).astype(int),
            "v1": rng.normal(0, 1, n),
            "v2": rng.normal(0, 1, n),
            "v3": rng.normal(0, 1, n),
            "v4": rng.normal(0, 1, n),
            "is_fraud":            np.zeros(n, dtype=int),
            "fraud_pattern":       [""] * n,
        })

    def make_fraud(n):
        pattern_idx = rng.integers(0, len(FRAUD_PATTERNS), n)
        patterns = [FRAUD_PATTERNS[i] for i in pattern_idx]
        cats = rng.choice(CATEGORIES, n)

        amounts = []
        hours = []
        distances = []
        n_tx = []

        for p in patterns:
            if p == "Valor muito alto para categoria":
                amounts.append(round(rng.uniform(3000, 15000), 2))
                hours.append(int(rng.integers(8, 20)))
                distances.append(round(rng.uniform(0, 50), 1))
                n_tx.append(1)
            elif p == "Múltiplas transações em sequência":
                amounts.append(round(rng.uniform(50, 500), 2))
                hours.append(int(rng.integers(0, 24)))
                distances.append(round(rng.uniform(0, 10), 1))
                n_tx.append(int(rng.integers(8, 20)))
            elif p == "Horário atípico (madrugada)":
                amounts.append(round(rng.uniform(200, 2000), 2))
                hours.append(int(rng.choice([1,2,3,4])))
                distances.append(round(rng.uniform(0, 30), 1))
                n_tx.append(int(rng.integers(1, 4)))
            elif p == "Localização inconsistente":
                amounts.append(round(rng.uniform(100, 1500), 2))
                hours.append(int(rng.integers(8, 22)))
                distances.append(round(rng.uniform(300, 2000), 1))
                n_tx.append(1)
            elif p == "Valor fracionado (structuring)":
                amounts.append(round(rng.uniform(4990, 5010), 2))
                hours.append(int(rng.integers(9, 18)))
                distances.append(round(rng.uniform(0, 20), 1))
                n_tx.append(int(rng.integers(3, 7)))
            else:
                amounts.append(round(rng.uniform(100, 3000), 2))
                hours.append(int(rng.integers(0, 24)))
                distances.append(round(rng.uniform(50, 500), 1))
                n_tx.append(int(rng.integers(1, 5)))

        return pd.DataFrame({
            "amount":              amounts,
            "hour":                hours,
            "day_of_week":         rng.integers(0, 7, n),
            "category":            cats,
            "merchant":            [rng.choice(MERCHANTS[c]) for c in cats],
            "distance_from_home":  distances,
            "n_transactions_1h":   n_tx,
            "avg_amount_30d":      np.clip(rng.normal(180, 80, n), 20, 800).round(2),
            "is_foreign":          rng.choice([0,1], n, p=[0.6,0.4]),
            "is_night":            [1 if h <= 5 or h >= 22 else 0 for h in hours],
            "v1": rng.normal(-3, 2, n),
            "v2": rng.normal(4, 2, n),
            "v3": rng.normal(-2, 1.5, n),
            "v4": rng.normal(3, 2, n),
            "is_fraud":            np.ones(n, dtype=int),
            "fraud_pattern":       patterns,
        })

    df = pd.concat([make_normal(n_normal), make_fraud(n_fraud)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["transaction_id"] = [f"TXN{i:06d}" for i in range(len(df))]
    df["amount_ratio"] = (df["amount"] / df["avg_amount_30d"].clip(lower=1)).round(3)

    # Timestamp simulado (últimos 30 dias)
    base = pd.Timestamp("2026-03-01")
    df["timestamp"] = [
        base + pd.Timedelta(
            days=int(rng.integers(0, 30)),
            hours=int(row.hour),
            minutes=int(rng.integers(0, 60)),
        )
        for _, row in df.iterrows()
    ]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Autoencoder (NumPy puro)
# ---------------------------------------------------------------------------
class NumpyAutoencoder:
    """
    Autoencoder raso implementado em NumPy puro.

    Arquitetura: input(10) → hidden(5) → bottleneck(3) → hidden(5) → output(10)
    Treinamento: SGD com backprop manual, ativação ReLU + sigmoid na saída.
    """

    def __init__(self, input_dim: int, hidden: int = 5, bottleneck: int = 3,
                 lr: float = 0.01, epochs: int = 50, batch_size: int = 64) -> None:
        self.input_dim   = input_dim
        self.hidden      = hidden
        self.bottleneck  = bottleneck
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self._init_weights()

    def _init_weights(self):
        rng = np.random.default_rng(42)
        d, h, b = self.input_dim, self.hidden, self.bottleneck
        self.W1 = rng.normal(0, 0.1, (d, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 0.1, (h, b))
        self.b2 = np.zeros(b)
        self.W3 = rng.normal(0, 0.1, (b, h))
        self.b3 = np.zeros(h)
        self.W4 = rng.normal(0, 0.1, (h, d))
        self.b4 = np.zeros(d)

    @staticmethod
    def _relu(x): return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x): return (x > 0).astype(float)

    @staticmethod
    def _sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1;   a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2;  a2 = self._relu(z2)
        z3 = a2 @ self.W3 + self.b3;  a3 = self._relu(z3)
        z4 = a3 @ self.W4 + self.b4;  a4 = self._sigmoid(z4)
        return a4, (z1, a1, z2, a2, z3, a3, z4, a4)

    def fit(self, X: np.ndarray) -> list[float]:
        losses = []
        n = len(X)
        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = X[idx[start:start + self.batch_size]]
                out, cache = self._forward(batch)
                z1, a1, z2, a2, z3, a3, z4, a4 = cache

                loss = np.mean((out - batch) ** 2)
                epoch_loss += loss

                # Backprop
                dL = 2 * (out - batch) / len(batch)
                da4 = dL * a4 * (1 - a4)
                dW4 = a3.T @ da4; db4 = da4.sum(0)
                da3 = da4 @ self.W4.T * self._relu_grad(z3)
                dW3 = a2.T @ da3; db3 = da3.sum(0)
                da2 = da3 @ self.W3.T * self._relu_grad(z2)
                dW2 = a1.T @ da2; db2 = da2.sum(0)
                da1 = da2 @ self.W2.T * self._relu_grad(z1)
                dW1 = batch.T @ da1; db1 = da1.sum(0)

                for W, dW, b, db in [(self.W4,dW4,self.b4,db4),(self.W3,dW3,self.b3,db3),
                                      (self.W2,dW2,self.b2,db2),(self.W1,dW1,self.b1,db1)]:
                    W -= self.lr * dW
                    b -= self.lr * db

            losses.append(epoch_loss)
        return losses

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        out, _ = self._forward(X)
        return np.mean((out - X) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Modelo híbrido
# ---------------------------------------------------------------------------
@dataclass
class ModelArtifacts:
    scaler:      StandardScaler
    iso_forest:  IsolationForest
    autoencoder: NumpyAutoencoder
    features:    list[str]
    threshold:   float
    ae_threshold: float
    metrics:     dict = field(default_factory=dict)


class HybridAnomalyDetector:
    """
    Ensemble híbrido: Isolation Forest + Autoencoder.

    score_final = alpha × IF_score + (1-alpha) × AE_score

    Uso típico
    ----------
    >>> detector = HybridAnomalyDetector()
    >>> detector.train()
    >>> results = detector.predict(df)
    """

    FEATURES = [
        "amount", "hour", "day_of_week", "distance_from_home",
        "n_transactions_1h", "avg_amount_30d", "is_foreign",
        "is_night", "amount_ratio", "v1", "v2", "v3", "v4",
    ]

    def __init__(
        self,
        alpha: float = 0.5,
        contamination: float = 0.02,
        model_path: Path = MODEL_PATH,
    ) -> None:
        self.alpha = alpha
        self.contamination = contamination
        self.model_path = model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifacts: Optional[ModelArtifacts] = None

    def train(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Treina o modelo híbrido. Se df=None, gera dataset sintético."""
        if df is None:
            logger.info("Gerando dataset sintético...")
            df = generate_transactions(n_normal=2000, n_fraud=40)

        # Salva dataset
        df.to_pickle(DATA_PATH)

        normal = df[df["is_fraud"] == 0][self.FEATURES].dropna()
        X_normal = normal.values

        # Pré-processamento
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_normal)
        X_minmax = (X_scaled - X_scaled.min(0)) / (X_scaled.max(0) - X_scaled.min(0) + 1e-8)

        # Isolation Forest
        logger.info("Treinando Isolation Forest...")
        iso = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        iso.fit(X_scaled)

        # Autoencoder
        logger.info("Treinando Autoencoder...")
        ae = NumpyAutoencoder(
            input_dim=len(self.FEATURES),
            hidden=8, bottleneck=4,
            lr=0.005, epochs=80, batch_size=64,
        )
        ae.fit(X_minmax)

        # Thresholds (percentil 98 dos dados normais)
        if_scores = -iso.score_samples(X_scaled)
        ae_errors = ae.reconstruction_error(X_minmax)
        if_thresh = float(np.percentile(if_scores, 98))
        ae_thresh = float(np.percentile(ae_errors, 98))

        # Métricas no dataset completo
        X_all = scaler.transform(df[self.FEATURES].fillna(0).values)
        X_all_mm = np.clip(
            (X_all - X_all.min(0)) / (X_all.max(0) - X_all.min(0) + 1e-8), 0, 1
        )
        if_all = -iso.score_samples(X_all)
        ae_all = ae.reconstruction_error(X_all_mm)

        if_norm = (if_all - if_all.min()) / (if_all.max() - if_all.min() + 1e-8)
        ae_norm = (ae_all - ae_all.min()) / (ae_all.max() - ae_all.min() + 1e-8)
        hybrid  = self.alpha * if_norm + (1 - self.alpha) * ae_norm
        threshold = self.alpha * (if_thresh - if_all.min()) / (if_all.max() - if_all.min() + 1e-8) + \
                    (1 - self.alpha) * (ae_thresh - ae_all.min()) / (ae_all.max() - ae_all.min() + 1e-8)

        preds = (hybrid > threshold).astype(int)
        true  = df["is_fraud"].values

        tp = int(((preds == 1) & (true == 1)).sum())
        fp = int(((preds == 1) & (true == 0)).sum())
        fn = int(((preds == 0) & (true == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        metrics = {
            "total": len(df), "n_fraud": int(true.sum()),
            "detected": int(tp), "false_positives": fp,
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "threshold": round(threshold, 4),
        }

        self._artifacts = ModelArtifacts(
            scaler=scaler, iso_forest=iso, autoencoder=ae,
            features=self.FEATURES, threshold=threshold,
            ae_threshold=ae_thresh, metrics=metrics,
        )

        with open(self.model_path, "wb") as f:
            pickle.dump(self._artifacts, f)

        logger.info(
            "Modelo salvo. F1=%.3f | Precisão=%.3f | Recall=%.3f",
            f1, precision, recall,
        )
        return metrics

    def load(self) -> "HybridAnomalyDetector":
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado. Execute python main.py --train"
            )
        with open(self.model_path, "rb") as f:
            self._artifacts = pickle.load(f)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula scores e classifica anomalias em um DataFrame."""
        if self._artifacts is None:
            self.load()
        art = self._artifacts

        X = df[art.features].fillna(0).values
        X_scaled = art.scaler.transform(X)
        X_mm = np.clip(
            (X_scaled - X_scaled.min(0)) / (X_scaled.max(0) - X_scaled.min(0) + 1e-8), 0, 1
        )

        if_scores = -art.iso_forest.score_samples(X_scaled)
        ae_scores = art.autoencoder.reconstruction_error(X_mm)

        if_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
        ae_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-8)
        hybrid  = self.alpha * if_norm + (1 - self.alpha) * ae_norm

        df = df.copy()
        df["if_score"]     = if_norm.round(4)
        df["ae_score"]     = ae_norm.round(4)
        df["hybrid_score"] = hybrid.round(4)
        df["is_anomaly"]   = (hybrid > art.threshold).astype(int)
        df["risk_level"]   = pd.cut(
            hybrid,
            bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
            labels=["Baixo", "Médio", "Alto", "Crítico"],
        )
        return df

    @property
    def metrics(self) -> dict:
        if self._artifacts is None:
            self.load()
        return self._artifacts.metrics