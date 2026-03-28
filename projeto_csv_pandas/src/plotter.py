"""
src/plotter.py
--------------
Geração de visualizações desacoplada da lógica de análise.
Recebe um AnalysisReport já calculado e produz PNGs de alta qualidade.

Design:
  - Cada método público gera exatamente um arquivo e retorna seu Path
  - generate_all() é o ponto de entrada conveniente
  - Tema visual coerente aplicado globalmente via rcParams
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from analyzer import AnalysisReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paleta e tema global
# ---------------------------------------------------------------------------
COLORS = {
    "safe":     "#2ecc71",
    "danger":   "#e74c3c",
    "primary":  "#3498db",
    "accent":   "#9b59b6",
    "gold":     "#f39c12",
    "dark":     "#2c3e50",
    "bg":       "#0d1117",   # fundo escuro — tema espacial
    "surface":  "#161b22",
    "grid":     "#21262d",
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
}

plt.rcParams.update({
    "figure.facecolor":    COLORS["bg"],
    "axes.facecolor":      COLORS["surface"],
    "axes.edgecolor":      COLORS["grid"],
    "axes.labelcolor":     COLORS["text"],
    "axes.titlecolor":     COLORS["text"],
    "text.color":          COLORS["text"],
    "xtick.color":         COLORS["subtext"],
    "ytick.color":         COLORS["subtext"],
    "grid.color":          COLORS["grid"],
    "grid.linestyle":      "--",
    "grid.alpha":          0.6,
    "axes.grid":           True,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.left":    False,
    "axes.spines.bottom":  False,
    "font.family":         "DejaVu Sans",
    "legend.facecolor":    COLORS["surface"],
    "legend.edgecolor":    COLORS["grid"],
    "legend.labelcolor":   COLORS["text"],
})


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    logger.info("Gráfico salvo: %s", path)
    return path


def _title(ax: plt.Axes, text: str, sub: str = "") -> None:
    ax.set_title(text, fontsize=13, fontweight="bold", pad=14, color=COLORS["text"])
    if sub:
        ax.text(
            0.5, 1.01, sub, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color=COLORS["subtext"],
        )


def _annotate_bar(ax: plt.Axes, fmt: str = "{:.0f}") -> None:
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h * 1.02,
                fmt.format(h), ha="center", va="bottom",
                fontsize=8, color=COLORS["text"], fontweight="bold",
            )


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------
class NasaPlotter:
    """
    Gera visualizações a partir de um AnalysisReport da NASA API.

    Cada método retorna o Path do arquivo salvo.
    """

    def __init__(self, report: AnalysisReport, output_dir: str | Path) -> None:
        self.report = report
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.neo = report.neo_df
        self.apod = report.apod_df

    # ------------------------------------------------------------------
    # 1. Asteroides por dia (barras empilhadas: seguro vs perigoso)
    # ------------------------------------------------------------------
    def plot_asteroids_per_day(self) -> Path:
        """Contagem diária de asteroides, empilhada por nível de perigo."""
        df = self.neo.copy()
        df["day"] = df["date"].dt.date

        grouped = (
            df.groupby(["day", "is_potentially_hazardous"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={False: "Seguro", True: "Potencialmente Perigoso"})
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(grouped))

        safe_vals = grouped.get("Seguro", pd.Series([0] * len(grouped))).values
        dang_vals = grouped.get("Potencialmente Perigoso", pd.Series([0] * len(grouped))).values

        ax.bar(x, safe_vals, color=COLORS["primary"], label="Seguro", alpha=0.85)
        ax.bar(x, dang_vals, bottom=safe_vals, color=COLORS["danger"],
               label="Potencialmente Perigoso", alpha=0.85)

        ax.set_xticks(list(x))
        ax.set_xticklabels([str(d) for d in grouped.index], rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend()
        ax.set_ylabel("Quantidade")
        _title(ax, "Asteroides Próximos à Terra por Dia",
               sub="Fonte: NASA NeoWs API — api.nasa.gov")

        fig.tight_layout()
        return _save(fig, self.output_dir, "01_asteroids_per_day.png")

    # ------------------------------------------------------------------
    # 2. Distribuição de diâmetro (histograma com escala log)
    # ------------------------------------------------------------------
    def plot_diameter_distribution(self) -> Path:
        """Histograma do diâmetro médio dos asteroides (escala logarítmica)."""
        diams = self.neo["diameter_avg_km"].dropna()
        diams = diams[diams > 0]

        fig, ax = plt.subplots(figsize=(9, 5))

        log_diams = np.log10(diams)
        counts, bins, patches = ax.hist(log_diams, bins=25,
                                         color=COLORS["accent"], edgecolor=COLORS["bg"],
                                         linewidth=0.5, alpha=0.85)

        # Colorir barras perigosas (diâmetro > 0.14 km = 140m, critério NASA)
        threshold_log = np.log10(0.14)
        for patch, left in zip(patches, bins[:-1]):
            if left >= threshold_log:
                patch.set_facecolor(COLORS["danger"])

        ax.set_xlabel("Diâmetro médio (log₁₀ km)")
        ax.set_ylabel("Quantidade de asteroides")
        _title(ax, "Distribuição de Tamanho dos Asteroides",
               sub="Barras vermelhas: diâmetro acima de 140m (critério NASA de risco)")

        # Adiciona legenda manual
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=COLORS["accent"], label="Abaixo de 140m"),
            Patch(color=COLORS["danger"], label="140m ou mais (potencial risco)"),
        ])

        fig.tight_layout()
        return _save(fig, self.output_dir, "02_diameter_distribution.png")

    # ------------------------------------------------------------------
    # 3. Velocidade vs distância (scatter)
    # ------------------------------------------------------------------
    def plot_velocity_vs_distance(self) -> Path:
        """Scatter: velocidade relativa × distância de passagem, colorido por perigo."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for hazardous, label, color, marker in [
            (False, "Seguro",                COLORS["primary"], "o"),
            (True,  "Potencialmente Perigoso", COLORS["danger"],  "^"),
        ]:
            subset = self.neo[self.neo["is_potentially_hazardous"] == hazardous]
            ax.scatter(
                subset["miss_distance_km"] / 1_000,       # → mil km
                subset["relative_velocity_kmh"] / 1_000,  # → mil km/h
                c=color, marker=marker, label=label,
                alpha=0.7, s=50, edgecolors="none",
            )

        ax.set_xlabel("Distância de passagem (× 1.000 km)")
        ax.set_ylabel("Velocidade relativa (× 1.000 km/h)")
        ax.legend()
        _title(ax, "Velocidade vs Distância de Passagem pela Terra",
               sub="Fonte: NASA NeoWs API")

        fig.tight_layout()
        return _save(fig, self.output_dir, "03_velocity_vs_distance.png")

    # ------------------------------------------------------------------
    # 4. Top 10 asteroides mais próximos (barras horizontais)
    # ------------------------------------------------------------------
    def plot_closest_asteroids(self) -> Path:
        """Top 10 asteroides com menor distância de passagem."""
        top10 = (
            self.neo.nsmallest(10, "miss_distance_km")[
                ["name", "miss_distance_km", "is_potentially_hazardous"]
            ]
            .sort_values("miss_distance_km")
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = [
            COLORS["danger"] if h else COLORS["primary"]
            for h in top10["is_potentially_hazardous"]
        ]
        bars = ax.barh(
            top10["name"],
            top10["miss_distance_km"] / 1_000,
            color=colors, alpha=0.85, edgecolor=COLORS["bg"],
        )

        for bar, val in zip(bars, top10["miss_distance_km"] / 1_000):
            ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,.0f}", va="center", fontsize=8, color=COLORS["text"])

        ax.set_xlabel("Distância de passagem (× 1.000 km)")
        _title(ax, "Top 10 Asteroides Mais Próximos da Terra",
               sub="Vermelho = potencialmente perigoso")

        fig.tight_layout()
        return _save(fig, self.output_dir, "04_closest_asteroids.png")

    # ------------------------------------------------------------------
    # 5. Magnitude absoluta vs diâmetro (scatter com regressão)
    # ------------------------------------------------------------------
    def plot_magnitude_vs_diameter(self) -> Path:
        """Relação entre magnitude absoluta e diâmetro estimado."""
        df = self.neo.dropna(subset=["absolute_magnitude", "diameter_avg_km"])
        df = df[df["diameter_avg_km"] > 0]

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.scatter(
            df["absolute_magnitude"],
            np.log10(df["diameter_avg_km"]),
            c=COLORS["gold"], alpha=0.6, s=40, edgecolors="none",
        )

        # Linha de tendência
        if len(df) >= 3:
            z = np.polyfit(df["absolute_magnitude"], np.log10(df["diameter_avg_km"]), 1)
            p = np.poly1d(z)
            x_line = np.linspace(df["absolute_magnitude"].min(),
                                  df["absolute_magnitude"].max(), 100)
            ax.plot(x_line, p(x_line), color=COLORS["danger"],
                    linewidth=1.5, linestyle="--", label="Tendência")
            ax.legend()

        ax.set_xlabel("Magnitude Absoluta (H)")
        ax.set_ylabel("Diâmetro médio (log₁₀ km)")
        _title(ax, "Magnitude Absoluta vs Diâmetro Estimado",
               sub="Quanto maior o H, menor e mais fraco o asteroide")

        fig.tight_layout()
        return _save(fig, self.output_dir, "05_magnitude_vs_diameter.png")

    # ------------------------------------------------------------------
    # 6. APOD — Tipos de mídia + comprimento das explicações
    # ------------------------------------------------------------------
    def plot_apod_overview(self) -> Path:
        """Visão geral do APOD: tipos de mídia e comprimento das explicações."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # — Subplot A: tipos de mídia —
        ax1 = axes[0]
        if not self.report.media_type_counts.empty:
            wedge_colors = [COLORS["primary"], COLORS["accent"],
                            COLORS["gold"], COLORS["safe"]]
            ax1.pie(
                self.report.media_type_counts.values,
                labels=self.report.media_type_counts.index,
                colors=wedge_colors[:len(self.report.media_type_counts)],
                autopct="%1.0f%%",
                startangle=90,
                textprops={"color": COLORS["text"]},
                wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2},
            )
            ax1.set_facecolor(COLORS["surface"])
        else:
            ax1.text(0.5, 0.5, "Sem dados APOD", ha="center", va="center",
                     transform=ax1.transAxes, color=COLORS["subtext"])
        _title(ax1, "Tipos de Mídia — APOD")

        # — Subplot B: comprimento das explicações por dia —
        ax2 = axes[1]
        if not self.apod.empty and "explanation_length" in self.apod.columns:
            apod_sorted = self.apod.sort_values("date")
            ax2.bar(
                range(len(apod_sorted)),
                apod_sorted["explanation_length"],
                color=COLORS["accent"], alpha=0.8, edgecolor=COLORS["bg"],
            )
            ax2.set_xticks(range(len(apod_sorted)))
            ax2.set_xticklabels(
                apod_sorted["date"].dt.strftime("%d/%m").tolist(),
                rotation=40, ha="right", fontsize=8,
            )
            ax2.set_ylabel("Caracteres na explicação")
        else:
            ax2.text(0.5, 0.5, "Sem dados APOD", ha="center", va="center",
                     transform=ax2.transAxes, color=COLORS["subtext"])
        _title(ax2, "Tamanho das Descrições por Dia — APOD")

        fig.suptitle("NASA APOD — Astronomy Picture of the Day",
                     fontsize=14, fontweight="bold", color=COLORS["text"], y=1.02)
        fig.tight_layout()
        return _save(fig, self.output_dir, "06_apod_overview.png")

    # ------------------------------------------------------------------
    # Conveniência
    # ------------------------------------------------------------------
    def generate_all(self) -> list[Path]:
        """Gera todos os gráficos e retorna lista de Paths."""
        logger.info("Gerando visualizações...")
        plots = [
            self.plot_asteroids_per_day(),
            self.plot_diameter_distribution(),
            self.plot_velocity_vs_distance(),
            self.plot_closest_asteroids(),
            self.plot_magnitude_vs_diameter(),
            self.plot_apod_overview(),
        ]
        logger.info("%d gráficos gerados.", len(plots))
        return plots