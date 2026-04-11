"""
main.py
-------
Ponto de entrada do projeto.

Modos:
    python main.py           # treina modelo + sobe API + dashboard
    python main.py --train   # só treina
    python main.py --serve   # só API (modelo já treinado)

Dashboard: http://localhost:8000/app
API docs:  http://localhost:8000/docs
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
    p = argparse.ArgumentParser(description="FraudScan — Anomaly Detector")
    p.add_argument("--train", action="store_true")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port",  type=int, default=8000)
    return p.parse_args()


def train():
    from model import HybridAnomalyDetector
    print("\n🤖 Treinando modelo híbrido (Isolation Forest + Autoencoder)...\n")
    det = HybridAnomalyDetector()
    metrics = det.train()

    print("=" * 52)
    print("  MÉTRICAS DO DETECTOR")
    print("=" * 52)
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    print("=" * 52)

    Path("output/metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    print(f"\n✅ Modelo salvo em output/model.pkl")
    print(f"📊 Dataset salvo em output/transactions.pkl\n")


def serve(port: int):
    import uvicorn
    print(f"""
╔══════════════════════════════════════════╗
║        FraudScan — Anomaly Detector      ║
╠══════════════════════════════════════════╣
║  Dashboard → http://localhost:{port}/app      ║
║  API Docs  → http://localhost:{port}/docs     ║
╚══════════════════════════════════════════╝
""")
    uvicorn.run("src.api:app", host="0.0.0.0", port=port,
                reload=False, log_level="warning")


def main():
    args = parse_args()

    if not args.train and not args.serve:
        args.train = True
        args.serve = True

    if args.train:
        train()
    if args.serve:
        serve(args.port)


if __name__ == "__main__":
    main()