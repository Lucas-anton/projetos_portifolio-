"""
main.py
-------
Ponto de entrada do projeto. Orquestra o pipeline completo:
    1. Coleta de dados reais da NASA Open API
    2. Transformação e análise
    3. Geração de gráficos
    4. Impressão do relatório no terminal

Execução:
    python main.py                        # últimos 7 dias (padrão)
    python main.py --days 3               # últimos 3 dias
    python main.py --no-plots             # apenas análise textual
    python main.py --output resultados/   # diretório de saída customizado

Variáveis de ambiente:
    NASA_API_KEY    Chave pessoal da NASA API (opcional — DEMO_KEY funciona)
                    Cadastro gratuito: https://api.nasa.gov
                    Sem chave: limite de 30 req/hora por IP
                    Com chave:  limite de 1.000 req/hora
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Garante que src/ está no path sem precisar instalar como pacote
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analyzer import NasaAnalyzer   # noqa: E402
from plotter import NasaPlotter     # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisador de dados da NASA Open API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Janela de coleta em dias (1–7, padrão: 7)",
    )
    parser.add_argument(
        "--output", default="output",
        help="Diretório de saída para gráficos (padrão: output/)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Pula geração de gráficos (apenas análise textual)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (1 <= args.days <= 7):
        print("❌ --days deve ser entre 1 e 7 (limitação da API NeoWs).")
        sys.exit(1)

    # ── 1. Pipeline de análise ──────────────────────────────────────────
    print(f"\n🚀 Coletando dados da NASA API ({args.days} dias)...\n")
    analyzer = NasaAnalyzer(days=args.days)
    report = analyzer.run()

    # ── 2. Monta o relatório textual ────────────────────────────────────
    top5 = (
        report.neo_df
        .nsmallest(5, "miss_distance_km")[
            ["name", "miss_distance_km", "relative_velocity_kmh",
             "diameter_avg_km", "is_potentially_hazardous"]
        ]
        .rename(columns={
            "name": "Nome",
            "miss_distance_km": "Distância (km)",
            "relative_velocity_kmh": "Velocidade (km/h)",
            "diameter_avg_km": "Diâm. médio (km)",
            "is_potentially_hazardous": "Perigoso?",
        })
    )

    lines: list[str] = [report.summary()]

    lines += ["\n📋 TOP 5 ASTEROIDES MAIS PRÓXIMOS", top5.to_string(index=False)]

    if not report.apod_df.empty:
        apod_display = report.apod_df[["date", "title", "media_type"]].copy()
        apod_display["date"] = apod_display["date"].dt.strftime("%Y-%m-%d")
        lines += ["\n🌌 APOD — IMAGENS DO PERÍODO", apod_display.to_string(index=False)]

    if not report.top_keywords.empty:
        lines += ["\n🔑 TOP PALAVRAS NOS TÍTULOS DO APOD", report.top_keywords.to_string()]

    full_report = "\n".join(lines) + "\n"

    # ── 3. Exibe no terminal e salva em arquivo ─────────────────────────
    print(full_report)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"report_{timestamp}.txt"
    report_path.write_text(full_report, encoding="utf-8")
    print(f"📄 Relatório textual salvo em: {report_path}")

    # ── 4. Gráficos ─────────────────────────────────────────────────────
    if not args.no_plots:
        plots_dir = output_dir / "plots"
        plotter = NasaPlotter(report, plots_dir)
        saved = plotter.generate_all()
        print(f"\n✅ {len(saved)} gráficos salvos em '{plots_dir}/':")
        for p in saved:
            print(f"   • {p.name}")

    print()


if __name__ == "__main__":
    main()