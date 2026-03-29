"""
src/plotter.py
--------------
Gera visualizações profissionais a partir do AnalysisReport.

Cada método público produz exatamente um PNG e retorna seu Path.
generate_all() é o ponto de entrada conveniente.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from analyzer import AnalysisReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paleta — tema escuro técnico (inspirado em terminais)
# ---------------------------------------------------------------------------
C = {
    "bg":       "#0f1117",
    "surface":  "#1a1d27",
    "border":   "#2a2d3e",
    "primary":  "#ff6600",    # laranja HN
    "secondary":"#3b82f6",
    "success":  "#22c55e",
    "warning":  "#f59e0b",
    "danger":   "#ef4444",
    "text":     "#e2e8f0",
    "subtext":  "#64748b",
    "grid":     "#1e2133",
}

plt.rcParams.update({
    "figure.facecolor":  C["bg"],
    "axes.facecolor":    C["surface"],
    "axes.edgecolor":    C["border"],
    "axes.labelcolor":   C["text"],
    "axes.titlecolor":   C["text"],
    "text.color":        C["text"],
    "xtick.color":       C["subtext"],
    "ytick.color":       C["subtext"],
    "grid.color":        C["grid"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "axes.grid":         True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
    "legend.facecolor":  C["surface"],
    "legend.edgecolor":  C["border"],
    "legend.labelcolor": C["text"],
    "font.family":       "DejaVu Sans",
})

TIER_COLORS = {
    "Baixo (<50)":     C["subtext"],
    "Médio (50-200)":  C["secondary"],
    "Alto (200-500)":  C["warning"],
    "Viral (500+)":    C["danger"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    logger.info("Gráfico salvo: %s", path)
    return path


def _title(ax: plt.Axes, text: str, sub: str = "") -> None:
    ax.set_title(text, fontsize=13, fontweight="bold",
                 pad=14, color=C["text"])
    if sub:
        ax.text(0.5, 1.015, sub, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8.5, color=C["subtext"])


def _source_note(fig: plt.Figure) -> None:
    fig.text(0.99, 0.01, "Fonte: Hacker News API — news.ycombinator.com",
             ha="right", va="bottom", fontsize=7.5, color=C["subtext"])


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------
class HackerNewsPlotter:
    """
    Gera visualizações a partir de um AnalysisReport do HN Scraper.
    """

    def __init__(self, report: AnalysisReport, output_dir: str | Path) -> None:
        self.report = report
        self.df = report.df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Top 15 Stories por Score (barras horizontais)
    # ------------------------------------------------------------------
    def plot_top_stories(self) -> Path:
        """As 15 stories com maior score — barras horizontais com gradiente."""
        top = self.df.head(15).copy()
        # Trunca títulos longos
        top["short_title"] = top["title"].str[:55] + "…"

        fig, ax = plt.subplots(figsize=(12, 7))

        norm_scores = (top["score"] - top["score"].min()) / (
            top["score"].max() - top["score"].min() + 1
        )
        colors = plt.cm.YlOrRd(0.3 + norm_scores * 0.7)

        bars = ax.barh(
            top["short_title"][::-1],
            top["score"][::-1],
            color=colors[::-1],
            edgecolor=C["bg"],
            linewidth=0.5,
            height=0.7,
        )

        for bar, score in zip(bars, top["score"][::-1]):
            ax.text(
                bar.get_width() + top["score"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:,}", va="center", fontsize=8.5,
                color=C["text"], fontweight="bold",
            )

        ax.set_xlabel("Score (pontos)")
        ax.tick_params(axis="y", labelsize=8.5)
        _title(ax, "Top 15 Stories por Score", sub="Hacker News — Top Stories")
        _source_note(fig)
        fig.tight_layout()
        return _save(fig, self.output_dir, "01_top_stories.png")

    # ------------------------------------------------------------------
    # 2. Distribuição de Score (histograma + KDE)
    # ------------------------------------------------------------------
    def plot_score_distribution(self) -> Path:
        """Distribuição de scores — evidencia a cauda longa típica do HN."""
        scores = self.df["score"].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # — Subplot A: escala linear —
        ax1 = axes[0]
        ax1.hist(scores, bins=40, color=C["primary"],
                 edgecolor=C["bg"], alpha=0.85, linewidth=0.4)
        ax1.set_xlabel("Score")
        ax1.set_ylabel("Quantidade de stories")
        _title(ax1, "Distribuição de Score (linear)")

        # — Subplot B: escala log —
        ax2 = axes[1]
        log_scores = np.log10(scores[scores > 0])
        ax2.hist(log_scores, bins=35, color=C["secondary"],
                 edgecolor=C["bg"], alpha=0.85, linewidth=0.4)
        ax2.set_xlabel("Score (log₁₀)")
        ax2.set_ylabel("Quantidade de stories")
        _title(ax2, "Distribuição de Score (log)", sub="Escala logarítmica — evidencia a cauda longa")

        _source_note(fig)
        fig.tight_layout()
        return _save(fig, self.output_dir, "02_score_distribution.png")

    # ------------------------------------------------------------------
    # 3. Score médio por hora do dia
    # ------------------------------------------------------------------
    def plot_score_by_hour(self) -> Path:
        """Score médio e volume de posts por hora — ideal para identificar horários de pico."""
        score_h = self.report.score_by_hour
        count_h = self.report.stories_by_hour

        if score_h.empty:
            logger.warning("Sem dados temporais para plot_score_by_hour.")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "Sem dados de hora disponíveis",
                    ha="center", va="center", transform=ax.transAxes, color=C["subtext"])
            return _save(fig, self.output_dir, "03_score_by_hour.png")

        fig, ax1 = plt.subplots(figsize=(12, 5))

        hours = score_h.index.astype(int)

        # Barras: volume por hora
        ax1.bar(hours, count_h.reindex(hours, fill_value=0),
                color=C["border"], alpha=0.6, label="Quantidade de posts", zorder=2)
        ax1.set_ylabel("Quantidade de posts", color=C["subtext"])
        ax1.tick_params(axis="y", labelcolor=C["subtext"])

        # Linha: score médio
        ax2 = ax1.twinx()
        ax2.plot(hours, score_h.values, color=C["primary"],
                 linewidth=2.5, marker="o", markersize=5,
                 label="Score médio", zorder=3)
        ax2.set_ylabel("Score médio", color=C["primary"])
        ax2.tick_params(axis="y", labelcolor=C["primary"])
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(C["primary"])

        ax1.set_xlabel("Hora do dia (UTC)")
        ax1.set_xticks(range(0, 24, 2))

        # Legenda combinada
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        _title(ax1, "Score Médio e Volume de Posts por Hora do Dia",
               sub="Horário em UTC")
        _source_note(fig)
        fig.tight_layout()
        return _save(fig, self.output_dir, "03_score_by_hour.png")

    # ------------------------------------------------------------------
    # 4. Top 10 Domínios
    # ------------------------------------------------------------------
    def plot_top_domains(self) -> Path:
        """Top 10 domínios mais citados — onde estão as notícias mais compartilhadas."""
        domains = self.report.top_domains

        if domains.empty:
            logger.warning("Sem dados de domínio.")
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.text(0.5, 0.5, "Sem dados de domínio", ha="center", va="center",
                    transform=ax.transAxes, color=C["subtext"])
            return _save(fig, self.output_dir, "04_top_domains.png")

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = [C["primary"] if i == 0 else C["secondary"]
                  for i in range(len(domains))]

        bars = ax.barh(domains.index[::-1], domains.values[::-1],
                       color=colors[::-1], edgecolor=C["bg"],
                       linewidth=0.5, height=0.65)

        for bar, val in zip(bars, domains.values[::-1]):
            ax.text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=9,
                    color=C["text"], fontweight="bold")

        ax.set_xlabel("Número de stories")
        ax.tick_params(axis="y", labelsize=9)
        _title(ax, "Top 10 Domínios Mais Compartilhados")
        _source_note(fig)
        fig.tight_layout()
        return _save(fig, self.output_dir, "04_top_domains.png")

    # ------------------------------------------------------------------
    # 5. Score vs Comentários (scatter)
    # ------------------------------------------------------------------
    def plot_score_vs_comments(self) -> Path:
        """Scatter: score × comentários, colorido por faixa de score."""
        df = self.df.dropna(subset=["score", "descendants", "score_tier"])

        fig, ax = plt.subplots(figsize=(10, 6))

        for tier, color in TIER_COLORS.items():
            subset = df[df["score_tier"].astype(str) == tier]
            ax.scatter(
                subset["score"], subset["descendants"],
                c=color, label=tier, alpha=0.65, s=35,
                edgecolors="none", zorder=3,
            )

        ax.set_xlabel("Score (pontos)")
        ax.set_ylabel("Número de comentários")
        ax.legend(title="Faixa de score", title_fontsize=8)
        _title(ax, "Score vs Número de Comentários",
               sub="Cada ponto = 1 story")
        _source_note(fig)
        fig.tight_layout()
        return _save(fig, self.output_dir, "05_score_vs_comments.png")

    # ------------------------------------------------------------------
    # 6. Distribuição por faixa de score (pizza) + Top autores (barras)
    # ------------------------------------------------------------------
    def plot_tiers_and_authors(self) -> Path:
        """Painel duplo: distribuição por tier de score + top autores."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # — Subplot A: pizza por tier —
        ax1 = axes[0]
        tier_counts = self.df["score_tier"].value_counts()
        colors_pie = [TIER_COLORS.get(str(t), C["subtext"]) for t in tier_counts.index]
        wedges, texts, autotexts = ax1.pie(
            tier_counts.values,
            labels=tier_counts.index.astype(str),
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": C["text"], "fontsize": 8.5},
            wedgeprops={"edgecolor": C["bg"], "linewidth": 2},
        )
        for at in autotexts:
            at.set_color(C["bg"])
            at.set_fontweight("bold")
        ax1.set_facecolor(C["surface"])
        _title(ax1, "Stories por Faixa de Score")

        # — Subplot B: top autores —
        ax2 = axes[1]
        authors = self.report.top_authors.head(10)
        if not authors.empty:
            bar_colors = [C["primary"] if i == 0 else C["secondary"]
                          for i in range(len(authors))]
            ax2.barh(authors.index[::-1], authors.values[::-1],
                     color=bar_colors[::-1], edgecolor=C["bg"],
                     linewidth=0.5, height=0.65)
            for i, (idx, val) in enumerate(zip(authors.index[::-1], authors.values[::-1])):
                ax2.text(val + 0.1,
                         i, str(val), va="center", fontsize=9,
                         color=C["text"], fontweight="bold")
            ax2.set_xlabel("Número de stories")
            ax2.tick_params(axis="y", labelsize=9)
        _title(ax2, "Top 10 Autores Mais Ativos")

        fig.suptitle("Hacker News — Visão Geral das Stories",
                     fontsize=13, fontweight="bold", color=C["text"], y=1.01)
        _source_note(fig)
        fig.tight_layout()
        return _save(fig, self.output_dir, "06_tiers_and_authors.png")

    # ------------------------------------------------------------------
    # Conveniência
    # ------------------------------------------------------------------
    def generate_all(self) -> list[Path]:
        """Gera todos os gráficos e retorna lista de Paths."""
        logger.info("Gerando visualizações...")
        plots = [
            self.plot_top_stories(),
            self.plot_score_distribution(),
            self.plot_score_by_hour(),
            self.plot_top_domains(),
            self.plot_score_vs_comments(),
            self.plot_tiers_and_authors(),
        ]
        logger.info("%d gráficos gerados.", len(plots))
        return plots