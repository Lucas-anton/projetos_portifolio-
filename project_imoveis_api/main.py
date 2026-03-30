"""
main.py
-------
Ponto de entrada do projeto.

Modos de uso:
    python main.py --train        # Treina o modelo e salva em output/
    python main.py --serve        # Sobe a API (requer modelo treinado)
    python main.py                # Treina e sobe a API em sequência
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

from trainer import ImovelTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="API de Previsão de Aluguel — Imóveis Brasil",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train", action="store_true",
                        help="Treina o modelo")
    parser.add_argument("--serve", action="store_true",
                        help="Sobe a API FastAPI")
    parser.add_argument("--samples", type=int, default=5000,
                        help="Amostras para treino (padrão: 5000)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host da API (padrão: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Porta da API (padrão: 8000)")
    return parser.parse_args()


def train(n_samples: int) -> None:
    trainer = ImovelTrainer(n_samples=n_samples)
    metrics = trainer.run()
    print(metrics.summary())

    # Salva métricas em JSON para o endpoint /model/info
    metrics_dict = {
        "algorithm":   "Random Forest Regressor",
        "n_estimators": 200,
        "n_train":     metrics.n_train,
        "n_test":      metrics.n_test,
        "mae":         round(metrics.mae, 2),
        "rmse":        round(metrics.rmse, 2),
        "r2":          round(metrics.r2, 4),
        "feature_importances": {
            k: round(v, 4)
            for k, v in sorted(
                metrics.feature_importances.items(),
                key=lambda x: x[1], reverse=True
            )
        },
    }
    metrics_path = Path("output/metrics.json")
    metrics_path.write_text(json.dumps(metrics_dict, ensure_ascii=False, indent=2))
    print(f"\n📄 Métricas salvas em: {metrics_path}")
    print(f"🤖 Modelo salvo em:    output/model.joblib\n")


def serve(host: str, port: int) -> None:
    import uvicorn
    print(f"\n🚀 API rodando em: http://{host}:{port}")
    print(f"📖 Documentação:   http://localhost:{port}/docs\n")
    uvicorn.run("src.api:app", host=host, port=port, reload=True)


def main() -> None:
    args = parse_args()

    # Sem flags = treina e serve
    if not args.train and not args.serve:
        args.train = True
        args.serve = True

    if args.train:
        print("\n🏋️  Treinando modelo...\n")
        train(args.samples)

    if args.serve:
        serve(args.host, args.port)


if __name__ == "__main__":
    main()