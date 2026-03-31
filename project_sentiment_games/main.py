"""
main.py
-------
Ponto de entrada do projeto.

Modos de uso:
    python main.py                  # Treina modelo e sobe dashboard
    python main.py --train          # Só treina
    python main.py --dashboard      # Só dashboard (requer modelo treinado)
    python main.py --api            # Sobe API FastAPI
    python main.py --train --api    # Treina e sobe API
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Games Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train",     action="store_true", help="Treina o modelo")
    parser.add_argument("--dashboard", action="store_true", help="Sobe o Streamlit")
    parser.add_argument("--api",       action="store_true", help="Sobe a FastAPI")
    parser.add_argument("--samples",   type=int, default=400, help="Amostras de treino")
    parser.add_argument("--port",      type=int, default=8000, help="Porta da API")
    return parser.parse_args()


def train(n_samples: int) -> None:
    from model import SentimentModel
    print("\n🤖 Treinando modelo de sentimentos...\n")
    model = SentimentModel()
    metrics = model.train(n_samples=n_samples)
    print(metrics.summary())

    metrics_dict = {
        "algorithm": "Logistic Regression + TF-IDF (1-2 grams)",
        "accuracy":  round(metrics.accuracy, 4),
        "n_train":   metrics.n_train,
        "n_test":    metrics.n_test,
        "classes":   ["negative", "neutral", "positive"],
        "report":    metrics.report,
    }
    Path("output/metrics.json").write_text(
        json.dumps(metrics_dict, ensure_ascii=False, indent=2)
    )
    print(f"\n✅ Modelo salvo em output/sentiment_model.joblib")
    print(f"📄 Métricas salvas em output/metrics.json\n")


def run_dashboard() -> None:
    import subprocess
    print("\n🚀 Subindo dashboard em http://localhost:8501\n")
    subprocess.run(["streamlit", "run", "app.py"])


def run_api(port: int) -> None:
    import uvicorn
    print(f"\n🚀 API rodando em http://localhost:{port}")
    print(f"📖 Docs: http://localhost:{port}/docs\n")
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=True)


def main():
    args = parse_args()

    if not any([args.train, args.dashboard, args.api]):
        args.train = True
        args.dashboard = True

    if args.train:
        train(args.samples)
    if args.api:
        run_api(args.port)
    elif args.dashboard:
        run_dashboard()


if __name__ == "__main__":
    main()