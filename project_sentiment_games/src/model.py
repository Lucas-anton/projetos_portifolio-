"""
src/model.py
------------
Treina e aplica o modelo de análise de sentimentos em textos de games.

Pipeline NLP
------------
  TF-IDF Vectorizer (uni+bigrams, stop words EN) →
  Logistic Regression (multi-class: positive/negative/neutral)

Também implementa análise léxica baseada em VADER para comparação
e ensemble com o modelo treinado.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from collector import RedditCollector

logger = logging.getLogger(__name__)

MODEL_PATH = Path("output/sentiment_model.joblib")
LABELS = ["negative", "neutral", "positive"]

# Léxico customizado de games (palavras-chave domain-specific)
GAMES_LEXICON = {
    "positive": [
        "amazing", "incredible", "satisfying", "awesome", "love",
        "great", "excellent", "perfect", "fantastic", "brilliant",
        "carry", "clutch", "insane", "goat", "fire", "based",
        "wholesome", "clean", "smooth", "balanced",
    ],
    "negative": [
        "broken", "trash", "terrible", "awful", "hate",
        "broken", "cheater", "hacker", "toxic", "unplayable",
        "rigged", "lag", "crash", "bug", "nerf", "rip",
        "predatory", "pay2win", "p2w", "scam", "dead",
    ],
}


@dataclass
class ModelMetrics:
    """Métricas completas de avaliação."""
    accuracy: float
    report: str
    confusion: np.ndarray
    n_train: int
    n_test: int

    def summary(self) -> str:
        lines = [
            "=" * 52,
            "  MÉTRICAS DO MODELO — SENTIMENT ANALYSIS",
            "=" * 52,
            f"  Acurácia geral  : {self.accuracy:.1%}",
            f"  Amostras treino : {self.n_train:,}",
            f"  Amostras teste  : {self.n_test:,}",
            "",
            "  Relatório por classe:",
        ]
        for line in self.report.split("\n"):
            if line.strip():
                lines.append(f"    {line}")
        lines.append("=" * 52)
        return "\n".join(lines)


@dataclass
class SentimentResult:
    """Resultado de classificação de um texto."""
    text: str
    sentiment: str           # positive | negative | neutral
    confidence: float        # probabilidade da classe prevista
    probabilities: dict[str, float]  # prob de cada classe
    lexicon_signals: list[str]       # palavras-chave encontradas


class SentimentModel:
    """
    Treina e aplica análise de sentimentos em textos de games.

    Uso típico
    ----------
    >>> model = SentimentModel()
    >>> model.train()
    >>> result = model.predict("This game is absolutely broken!")
    >>> print(result.sentiment, result.confidence)
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._pipeline: Optional[Pipeline] = None

    # ------------------------------------------------------------------
    # Pré-processamento
    # ------------------------------------------------------------------
    @staticmethod
    def preprocess(text: str) -> str:
        """Limpa e normaliza texto para o modelo."""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
        text = re.sub(r"@\w+|#\w+", " ", text)             # remove @mentions e #tags
        text = re.sub(r"[^a-z0-9\s\!\?\.\,\'']", " ", text)  # mantém pontuação básica
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------
    def train(self, n_samples: int = 500) -> ModelMetrics:
        """
        Coleta dados, treina o pipeline e serializa o modelo.

        Usa dataset sintético com labels verdadeiros (_true_sentiment)
        para garantir qualidade do treino sem depender de API.
        """
        logger.info("Coletando dados de treino...")
        collector = RedditCollector()
        result = collector.run(limit=n_samples)
        df = result.df

        if "_true_sentiment" not in df.columns:
            raise RuntimeError(
                "Dataset sem labels. Use o dataset sintético para treino."
            )

        df["text_clean"] = df["text"].apply(self.preprocess)
        df = df[df["text_clean"].str.len() > 5].copy()

        X = df["text_clean"]
        y = df["_true_sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("Treinando pipeline TF-IDF + Logistic Regression...")
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                stop_words="english",
                min_df=2,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        self._pipeline.fit(X_train, y_train)

        y_pred = self._pipeline.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        report = classification_report(y_test, y_pred, target_names=LABELS)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=LABELS)

        joblib.dump(self._pipeline, self.model_path)
        logger.info("Modelo salvo em %s (acurácia: %.1f%%)", self.model_path, accuracy * 100)

        return ModelMetrics(
            accuracy=accuracy,
            report=report,
            confusion=conf_matrix,
            n_train=len(X_train),
            n_test=len(X_test),
        )

    def load(self) -> "SentimentModel":
        """Carrega modelo serializado do disco."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {self.model_path}. "
                "Execute 'python main.py --train' primeiro."
            )
        self._pipeline = joblib.load(self.model_path)
        logger.info("Modelo carregado de %s", self.model_path)
        return self

    # ------------------------------------------------------------------
    # Predição
    # ------------------------------------------------------------------
    def predict(self, text: str) -> SentimentResult:
        """Classifica um texto e retorna resultado detalhado."""
        if self._pipeline is None:
            self.load()

        clean = self.preprocess(text)
        proba = self._pipeline.predict_proba([clean])[0]
        classes = self._pipeline.classes_

        probs = {c: round(float(p), 4) for c, p in zip(classes, proba)}
        predicted = max(probs, key=probs.get)
        confidence = probs[predicted]

        # Sinais léxicos para explicabilidade
        words = set(clean.split())
        signals = []
        for word in words:
            if word in GAMES_LEXICON["positive"]:
                signals.append(f"✅ {word}")
            elif word in GAMES_LEXICON["negative"]:
                signals.append(f"❌ {word}")

        return SentimentResult(
            text=text,
            sentiment=predicted,
            confidence=confidence,
            probabilities=probs,
            lexicon_signals=signals[:5],
        )

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Classifica uma lista de textos eficientemente."""
        if self._pipeline is None:
            self.load()

        cleaned = [self.preprocess(t) for t in texts]
        probas = self._pipeline.predict_proba(cleaned)
        classes = self._pipeline.classes_

        results = []
        for text, proba in zip(texts, probas):
            probs = {c: round(float(p), 4) for c, p in zip(classes, proba)}
            predicted = max(probs, key=probs.get)
            confidence = probs[predicted]

            words = set(self.preprocess(text).split())
            signals = []
            for word in words:
                if word in GAMES_LEXICON["positive"]:
                    signals.append(f"✅ {word}")
                elif word in GAMES_LEXICON["negative"]:
                    signals.append(f"❌ {word}")

            results.append(SentimentResult(
                text=text,
                sentiment=predicted,
                confidence=confidence,
                probabilities=probs,
                lexicon_signals=signals[:5],
            ))

        return results

    def analyze_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica predict_batch em um DataFrame com coluna 'text'."""
        results = self.predict_batch(df["text"].tolist())
        df = df.copy()
        df["sentiment"]  = [r.sentiment for r in results]
        df["confidence"] = [r.confidence for r in results]
        df["prob_positive"] = [r.probabilities.get("positive", 0) for r in results]
        df["prob_negative"] = [r.probabilities.get("negative", 0) for r in results]
        df["prob_neutral"]  = [r.probabilities.get("neutral",  0) for r in results]
        return df