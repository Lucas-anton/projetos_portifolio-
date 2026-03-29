"""
src/analyzer.py
---------------
Transforma um ScrapeResult em um DataFrame limpo e calcula métricas.

Responsabilidades:
  - Converter lista de dicts → DataFrame tipado
  - Limpeza e validação dos dados
  - Cálculo de métricas e rankings
  - Persistência: salva CSV e JSON em output/data/
  - Retorna AnalysisReport pronto para o Plotter e o relatório textual
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from scraper import ScrapeResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MIN_SCORE_THRESHOLD = 1   # remove itens sem pontuação (spam/duplicatas)


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------
@dataclass
class AnalysisReport:
    """Contêiner tipado com DataFrame limpo e todas as métricas calculadas."""

    # — Dados —
    df: pd.DataFrame

    # — Métricas gerais —
    total_stories: int
    avg_score: float
    median_score: float
    avg_comments: float
    top_score: int
    top_story_title: str
    top_story_url: str

    # — Rankings —
    top_domains: pd.Series         # domínios mais frequentes
    top_authors: pd.Series         # autores mais ativos
    score_by_hour: pd.Series       # pontuação média por hora do dia
    stories_by_hour: pd.Series     # volume por hora
    score_distribution: pd.Series  # quartis de score

    # — Metadados —
    scraped_at: str
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  RELATÓRIO DE ANÁLISE — HACKER NEWS SCRAPER",
            "=" * 62,
            f"  Stories coletadas   : {self.total_stories}",
            f"  Score médio         : {self.avg_score:.1f}",
            f"  Score mediano       : {self.median_score:.1f}",
            f"  Comentários médios  : {self.avg_comments:.1f}",
            f"  Story mais votada   : {self.top_score} pts",
            f"  Título              : {self.top_story_title[:55]}...",
            f"  Coletado em         : {self.scraped_at}",
            f"  Tempo de coleta     : {self.elapsed_seconds}s",
            "",
            "  Avisos:",
        ]
        if self.warnings:
            lines += [f"    ⚠  {w}" for w in self.warnings]
        else:
            lines.append("    ✓  Nenhum problema encontrado.")
        lines.append("=" * 62)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class HackerNewsAnalyzer:
    """
    Transforma ScrapeResult → AnalysisReport.

    Uso típico
    ----------
    >>> result = scraper.run()
    >>> report = HackerNewsAnalyzer(result, output_dir="output").run()
    >>> print(report.summary())

    Parâmetros
    ----------
    result     : ScrapeResult — dados brutos do scraper
    output_dir : Path | str   — onde salvar CSV e JSON
    """

    def __init__(self, result: ScrapeResult, output_dir: str | Path = "output") -> None:
        self.result = result
        self.output_dir = Path(output_dir)
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Etapa 1 — Construção e limpeza do DataFrame
    # ------------------------------------------------------------------
    def build_dataframe(self) -> "HackerNewsAnalyzer":
        """Converte lista de dicts em DataFrame limpo e tipado."""
        logger.info("Construindo DataFrame com %d itens...", len(self.result.items))

        df = pd.DataFrame(self.result.items)

        if df.empty:
            logger.warning("Nenhum dado para processar.")
            self._df = df
            return self

        # — Tipos —
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
        df["descendants"] = pd.to_numeric(df["descendants"], errors="coerce").fillna(0).astype(int)
        df["published_hour"] = pd.to_numeric(df["published_hour"], errors="coerce")

        # — Limpeza —
        before = len(df)
        df = df[df["title"].str.strip().ne("")]          # remove títulos vazios
        df = df[df["score"] >= MIN_SCORE_THRESHOLD]      # remove score zerado
        df = df.drop_duplicates(subset=["id"])           # remove duplicatas por ID
        after = len(df)

        if before != after:
            logger.info("Limpeza: %d → %d linhas (%d removidas).",
                        before, after, before - after)

        # — Feature: faixa de score —
        df["score_tier"] = pd.cut(
            df["score"],
            bins=[0, 50, 200, 500, float("inf")],
            labels=["Baixo (<50)", "Médio (50-200)", "Alto (200-500)", "Viral (500+)"],
        )

        # — Feature: tem URL externa? —
        df["has_external_url"] = df["url"].notna() & df["url"].ne("")

        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        self._df = df
        logger.info("DataFrame pronto: %d linhas, %d colunas.", *df.shape)
        return self

    # ------------------------------------------------------------------
    # Etapa 2 — Persistência
    # ------------------------------------------------------------------
    def save(self) -> "HackerNewsAnalyzer":
        """Salva os dados limpos em CSV e JSON."""
        if self._df is None or self._df.empty:
            logger.warning("Sem dados para salvar.")
            return self

        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV — fácil de abrir no Excel/Sheets
        csv_path = data_dir / f"hn_stories_{timestamp}.csv"
        self._df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("CSV salvo: %s", csv_path)

        # JSON — útil para APIs e pipelines downstream
        json_path = data_dir / f"hn_stories_{timestamp}.json"
        records = self._df.copy()
        records["published_at"] = records["published_at"].astype(str)
        records["score_tier"] = records["score_tier"].astype(str)
        json_path.write_text(
            json.dumps(records.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("JSON salvo: %s", json_path)

        return self

    # ------------------------------------------------------------------
    # Etapa 3 — Métricas
    # ------------------------------------------------------------------
    def analyze(self) -> AnalysisReport:
        """Calcula métricas e retorna AnalysisReport."""
        if self._df is None:
            raise RuntimeError("Execute .build_dataframe() antes de .analyze().")

        df = self._df
        warnings: list[str] = []

        if df.empty:
            warnings.append("DataFrame vazio — sem dados para analisar.")

        # — Métricas gerais —
        total = len(df)
        avg_score = df["score"].mean() if total else 0.0
        median_score = df["score"].median() if total else 0.0
        avg_comments = df["descendants"].mean() if total else 0.0
        top_row = df.iloc[0] if total else {}
        top_score = int(top_row.get("score", 0))
        top_title = str(top_row.get("title", ""))
        top_url = str(top_row.get("hn_url", ""))

        # — Rankings —
        top_domains = (
            df[df["domain"].ne("")]
            ["domain"].value_counts().head(10)
        )
        top_authors = df["by"].value_counts().head(10)

        # — Padrões temporais —
        score_by_hour = (
            df.groupby("published_hour")["score"].mean().round(1)
            if "published_hour" in df.columns else pd.Series(dtype=float)
        )
        stories_by_hour = (
            df.groupby("published_hour").size()
            if "published_hour" in df.columns else pd.Series(dtype=int)
        )

        # — Distribuição de score —
        score_distribution = df["score"].describe()

        if avg_score < 10:
            warnings.append("Score médio muito baixo — verifique os dados coletados.")

        logger.info("Análise concluída. Total: %d stories, score médio: %.1f.",
                    total, avg_score)

        return AnalysisReport(
            df=df,
            total_stories=total,
            avg_score=avg_score,
            median_score=median_score,
            avg_comments=avg_comments,
            top_score=top_score,
            top_story_title=top_title,
            top_story_url=top_url,
            top_domains=top_domains,
            top_authors=top_authors,
            score_by_hour=score_by_hour,
            stories_by_hour=stories_by_hour,
            score_distribution=score_distribution,
            scraped_at=self.result.scraped_at,
            elapsed_seconds=self.result.elapsed_seconds,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Conveniência
    # ------------------------------------------------------------------
    def run(self) -> AnalysisReport:
        """Pipeline completo: build → save → analyze."""
        return self.build_dataframe().save().analyze()