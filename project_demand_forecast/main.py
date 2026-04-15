"""
main.py
-------
Ponto de entrada do projeto integrador.

Pipeline completo:
    1. Coleta/gera dados de produtos e histórico de vendas
    2. Treina modelo GBM com feature engineering de séries temporais
    3. Sobe API FastAPI com endpoints de previsão
    4. Serve dashboard HTML premium

Modos:
    python main.py           # pipeline completo
    python main.py --train   # só treina
    python main.py --serve   # só API (modelo já treinado)
    python main.py --once    # executa pipeline e imprime métricas

Dashboard: http://localhost:8000/app
API Docs:  http://localhost:8000/docs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="DemandIQ — Demand Forecast Pipeline")
    p.add_argument("--train", action="store_true", help="Treina o modelo")
    p.add_argument("--serve", action="store_true", help="Sobe a API")
    p.add_argument("--once",  action="store_true", help="Treina e imprime métricas")
    p.add_argument("--port",  type=int, default=8000)
    return p.parse_args()


def run_pipeline():
    from collector import SalesGenerator
    from model import DemandForecaster

    # 1. Coleta / gera dados
    print("\n📦 Gerando dados de vendas (365 dias × 15 produtos)...")
    gen = SalesGenerator(days=365)
    dataset = gen.run()

    Path("output").mkdir(exist_ok=True)
    dataset.sales.to_pickle("output/sales.pkl")
    dataset.products.to_pickle("output/products.pkl")
    print(f"   ✓ {len(dataset.sales):,} registros | {len(dataset.products)} produtos")
    print(f"   ✓ Período: {dataset.date_range[0]} → {dataset.date_range[1]}")

    # 2. Treino
    print("\n🤖 Treinando modelos de previsão...")
    forecaster = DemandForecaster()
    metrics = forecaster.train(dataset.sales)

    print(f"\n{'='*52}")
    print(f"  MÉTRICAS DO PIPELINE")
    print(f"{'='*52}")
    print(f"  Produtos treinados : {metrics['n_products']}")
    print(f"  MAE médio          : {metrics['avg_mae']} unidades")
    print(f"  MAPE médio         : {metrics['avg_mape']*100:.1f}%")
    print(f"  Treino até         : {metrics['train_end']}")
    print(f"{'='*52}")

    # Salva métricas
    Path("output/metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )
    print(f"\n✅ Modelo salvo em output/forecast_model.pkl")
    return metrics


def run_server(port: int):
    import uvicorn
    print(f"""
╔══════════════════════════════════════════════╗
║       DemandIQ — Demand Forecast Engine      ║
╠══════════════════════════════════════════════╣
║  Dashboard  →  http://localhost:{port}/app       ║
║  API Docs   →  http://localhost:{port}/docs      ║
╚══════════════════════════════════════════════╝
""")
    uvicorn.run("src.api:app", host="0.0.0.0", port=port,
                reload=False, log_level="warning")


def main():
    args = parse_args()

    if args.once:
        run_pipeline()
        return

    if not args.train and not args.serve:
        args.train = True
        args.serve = True

    if args.train:
        run_pipeline()

    if args.serve:
        run_server(args.port)


if __name__ == "__main__":
    main()