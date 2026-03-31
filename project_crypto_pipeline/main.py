"""
main.py
-------
Ponto de entrada do projeto.

Modos:
    python main.py              # roda pipeline + server web
    python main.py --once       # coleta uma vez e exporta JSON
    python main.py --pipeline   # só o pipeline (sem server)
    python main.py --server     # só o server web
    python main.py --interval 60  # intervalo customizado (segundos)

Acesse o dashboard em: http://localhost:8080
"""

import argparse
import http.server
import logging
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="CryptoFlow Pipeline", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--once",     action="store_true", help="Coleta uma vez e sai")
    p.add_argument("--pipeline", action="store_true", help="Só o pipeline (loop)")
    p.add_argument("--server",   action="store_true", help="Só o servidor web")
    p.add_argument("--interval", type=int, default=300, help="Intervalo em segundos (padrão: 300)")
    p.add_argument("--port",     type=int, default=8080, help="Porta do servidor (padrão: 8080)")
    return p.parse_args()


def run_server(port: int, web_dir: str = "web") -> None:
    """Servidor HTTP simples para o dashboard."""
    os.chdir(web_dir)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # silencia logs de acesso

    server = http.server.HTTPServer(("", port), Handler)
    logger.info("🌐 Dashboard em: http://localhost:%d", port)
    server.serve_forever()


def main():
    args = parse_args()

    from pipeline import CryptoPipeline
    pipeline = CryptoPipeline()

    # Modo: coleta única
    if args.once:
        print("\n⚡ Executando coleta única...\n")
        result = pipeline.run_once()
        if result["status"] == "success":
            print(f"✅ {result['saved']} moedas coletadas às {result['ran_at']}")
            print(f"📁 Banco: output/crypto.db")
            print(f"🌐 JSON:  web/data.json")
        else:
            print(f"❌ Erro: {result.get('error')}")
        return

    # Modo: só pipeline
    if args.pipeline:
        pipeline.run_loop(interval=args.interval)
        return

    # Modo: só server
    if args.server:
        run_server(args.port)
        return

    # Modo padrão: pipeline + server em threads separadas
    print(f"""
╔══════════════════════════════════════════╗
║        CryptoFlow Pipeline               ║
╠══════════════════════════════════════════╣
║  Dashboard → http://localhost:{args.port:<5}      ║
║  Intervalo → {args.interval}s                     ║
║  Banco     → output/crypto.db            ║
║  Ctrl+C    → encerrar                    ║
╚══════════════════════════════════════════╝
""")

    # Coleta inicial antes de subir o server
    print("⚡ Coleta inicial...")
    pipeline.run_once()

    # Server em thread daemon
    t = threading.Thread(target=run_server, args=(args.port,), daemon=True)
    t.start()

    # Pipeline em loop na thread principal
    pipeline.run_loop(interval=args.interval)


if __name__ == "__main__":
    main()