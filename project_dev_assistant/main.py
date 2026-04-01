"""
main.py
-------
Sobe o backend FastAPI + serve o frontend estático.

Execução:
    python main.py

Variáveis de ambiente:
    GROQ_API_KEY    → https://console.groq.com (gratuito, recomendado)
    OPENAI_API_KEY  → https://platform.openai.com (pago)

Sem nenhuma key: roda em modo demo com instruções de configuração.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
    datefmt="%H:%M:%S",
)

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api import app as api_app


# Monta o frontend como rota /
api_app.mount("/static", StaticFiles(directory="web"), name="static")

@api_app.get("/app", include_in_schema=False)
def serve_frontend():
    return FileResponse("web/index.html")


if __name__ == "__main__":
    import os

    groq_key  = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    print("""
╔══════════════════════════════════════════════╗
║           DevMentor — Career Assistant       ║
╠══════════════════════════════════════════════╣
║  Chat  →  http://localhost:8000/app          ║
║  API   →  http://localhost:8000              ║
║  Docs  →  http://localhost:8000/docs         ║
╚══════════════════════════════════════════════╝
""")

    if groq_key:
        print("✅ Groq API key detectada — respostas reais ativas\n")
    elif openai_key:
        print("✅ OpenAI API key detectada — respostas reais ativas\n")
    else:
        print("⚠️  Nenhuma API key encontrada — modo demo ativo")
        print("   Para ativar: export GROQ_API_KEY=sua_chave")
        print("   Chave gratuita: https://console.groq.com\n")

    uvicorn.run(
        "main:api_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="warning",
    )