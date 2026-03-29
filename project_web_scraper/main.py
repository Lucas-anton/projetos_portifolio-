"""
main.py
-------
Ponto de entrada do projeto. Pipeline completo:
    1. Coleta notícias do Hacker News (HackerNewsScraper)
    2. Limpeza, análise e persistência (HackerNewsAnalyzer)
    3. Geração de gráficos (HackerNewsPlotter)
    4. Relatório textual salvo em output/

Execução:
    python main.py                        # padrão: top 100 stories
    python main.py --limit 200            # coleta 200 stories
    python main.py --category newstories  # notícias mais recentes
    python main.py --no-plots             # apenas análise textual
    python main.py --output resultados/   # diretório de saída customizado

Categorias disponíveis:
    topstories   Mais votadas agora (padrão)
    newstories   Mais recentes
    beststories  Melhores de todos os tempos
    askstories   Ask HN
    showstories  Show HN
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scraper import HackerNewsScraper    # noqa: E402
from analyzer import HackerNewsAnalyzer  # noqa: E402
from plotter import HackerNewsPlotter    # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Web Scraper e Analisador do Hacker News",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Número de stories a coletar (1–500, padrão: 100)",
    )
    parser.add_argument(
        "--category", default="topstories",
        choices=["topstories", "newstories", "beststories", "askstories", "showstories"],
        help="Categoria do HN (padrão: topstories)",
    )
    parser.add_argument(
        "--output", default="output",
        help="Diretório de saída (padrão: output/)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Pula geração de gráficos",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (1 <= args.limit <= 500):
        print("❌ --limit deve estar entre 1 e 500.")
        sys.exit(1)

    output_dir = Path(args.output)

    # ── 1. Scraping ─────────────────────────────────────────────────────
    print(f"\n🔍 Coletando {args.limit} stories ({args.category})...\n")
    scraper = HackerNewsScraper(limit=args.limit, category=args.category)
    result = scraper.run()

    print(f"   ✓ {result.total_fetched} stories coletadas em {result.elapsed_seconds}s"
          f" (taxa de sucesso: {result.success_rate():.1%})\n")

    # ── 2. Análise + persistência ────────────────────────────────────────
    analyzer = HackerNewsAnalyzer(result, output_dir=output_dir)
    report = analyzer.run()

    # ── 3. Monta relatório textual ───────────────────────────────────────
    lines: list[str] = [report.summary()]

    lines += [
        "\n📋 TOP 10 STORIES",
        report.df.head(10)[["title", "score", "descendants", "domain", "by"]]
        .rename(columns={
            "title": "Título", "score": "Score",
            "descendants": "Comentários", "domain": "Domínio", "by": "Autor",
        })
        .to_string(index=False),
    ]

    if not report.top_domains.empty:
        lines += ["\n🌐 TOP 10 DOMÍNIOS", report.top_domains.to_string()]

    if not report.top_authors.empty:
        lines += ["\n✍️  TOP 10 AUTORES", report.top_authors.head(10).to_string()]

    if not report.score_by_hour.empty:
        lines += [
            "\n🕐 SCORE MÉDIO POR HORA (UTC)",
            report.score_by_hour.to_string(),
        ]

    lines += [
        "\n📊 ESTATÍSTICAS DE SCORE",
        report.score_distribution.to_string(),
    ]

    full_report = "\n".join(lines) + "\n"

    # ── 4. Exibe e salva relatório textual ───────────────────────────────
    print(full_report)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"report_{timestamp}.txt"
    report_path.write_text(full_report, encoding="utf-8")
    print(f"📄 Relatório salvo em: {report_path}")

    # ── 5. Gráficos ──────────────────────────────────────────────────────
    if not args.no_plots:
        plots_dir = output_dir / "plots"
        plotter = HackerNewsPlotter(report, plots_dir)
        saved = plotter.generate_all()
        print(f"\n✅ {len(saved)} gráficos salvos em '{plots_dir}/':")
        for p in saved:
            print(f"   • {p.name}")

    print()


if __name__ == "__main__":
    main()