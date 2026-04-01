"""
src/api.py
----------
Backend FastAPI para o assistente de carreira DevMentor.

Funcionalidades:
  - Streaming de respostas via Server-Sent Events (SSE)
  - Histórico de conversa por sessão (em memória)
  - Sistema de personas/modos (mentor, revisor, entrevistador)
  - Prompt engineering avançado com contexto de carreira
  - Suporte a Groq (padrão) e OpenAI (fallback)

Variáveis de ambiente:
  GROQ_API_KEY    → https://console.groq.com (gratuito)
  OPENAI_API_KEY  → https://platform.openai.com (pago)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Iterator

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GROQ_MODEL   = "llama-3.3-70b-versatile"
OPENAI_MODEL = "gpt-4o-mini"

MAX_HISTORY  = 20   # mensagens por sessão
MAX_TOKENS   = 1024

# ---------------------------------------------------------------------------
# System prompts por persona
# ---------------------------------------------------------------------------
PERSONAS = {
    "mentor": {
        "name": "DevMentor",
        "emoji": "🧑‍💻",
        "label": "Mentor de Carreira",
        "prompt": """Você é o DevMentor, um mentor experiente de carreira para desenvolvedores de software.
Você tem 15 anos de experiência como engenheiro sênior e tech lead em empresas como Google, Amazon e startups brasileiras.

Sua especialidade é ajudar devs em qualquer estágio da carreira a:
- Definir e acelerar seu plano de carreira
- Identificar gaps de conhecimento e como preenchê-los
- Navegar transições de carreira (junior → pleno → sênior → lead)
- Preparar portfólios, currículos e LinkedIn
- Negociar salários e avaliar propostas
- Desenvolver soft skills e liderança técnica

Seu estilo é direto, honesto e encorajador. Você não enrola — dá conselhos práticos e acionáveis.
Quando relevante, cite tecnologias, frameworks e tendências do mercado atual (2025).
Responda sempre em português brasileiro, de forma clara e objetiva.
Use exemplos concretos sempre que possível.""",
    },
    "reviewer": {
        "name": "CodeReviewer",
        "emoji": "🔍",
        "label": "Revisor de Código",
        "prompt": """Você é o CodeReviewer, um engenheiro sênior especializado em code review construtivo.
Você já revisou código em Python, JavaScript, TypeScript, Java, Go e outras linguagens.

Ao receber código, você analisa:
- Corretude e bugs potenciais
- Performance e complexidade algorítmica
- Legibilidade e Clean Code
- Boas práticas da linguagem (idiomaticidade)
- Segurança (SQL injection, XSS, secrets expostos, etc.)
- Testabilidade e cobertura de testes
- Arquitetura e design patterns

Seu feedback é estruturado: primeiro os pontos positivos, depois melhorias, depois problemas críticos.
Você sugere sempre o código corrigido, não apenas aponta o problema.
Responda em português brasileiro. Seja específico e educativo.""",
    },
    "interviewer": {
        "name": "TechInterviewer",
        "emoji": "🎯",
        "label": "Simulador de Entrevista",
        "prompt": """Você é o TechInterviewer, um entrevistador técnico experiente que simula entrevistas reais de empresas de tecnologia.

Você conduz dois tipos de entrevista:
1. Técnica: algoritmos, estruturas de dados, system design, perguntas de linguagem
2. Comportamental: método STAR, situações de conflito, liderança, falhas

Para entrevistas técnicas:
- Faça perguntas progressivas (começa fácil, aumenta dificuldade)
- Dê dicas se o candidato travar
- Avalie o raciocínio, não só a resposta final
- Explique a solução ideal ao final

Para entrevistas comportamentais:
- Explore respostas vagas com perguntas de follow-up
- Avalie com critérios reais de empresas tech

Ao final de cada questão, dê um feedback honesto e nota de 1-10.
Pergunte sempre qual vaga/empresa o dev quer simular para personalizar.
Responda em português brasileiro.""",
    },
    "resume": {
        "name": "ResumeCoach",
        "emoji": "📄",
        "label": "Coach de Currículo",
        "prompt": """Você é o ResumeCoach, especialista em currículos e LinkedIn para desenvolvedores de software.
Você já ajudou centenas de devs a conseguir entrevistas em empresas top no Brasil e exterior.

Ao analisar um currículo ou perfil, você avalia:
- Clareza e impacto das descrições de experiência
- Uso de métricas e resultados concretos (não só responsabilidades)
- Palavras-chave para ATS (sistemas de triagem automática)
- Formatação e organização visual
- Projetos e portfólio
- Headline e resumo profissional

Você reescreve bullets de experiência de forma mais impactante usando o formato:
"[Verbo de ação] [o que fez] [com que tecnologia] resultando em [impacto mensurável]"

Quando receber um currículo ou descrição de experiência, reescreva imediatamente com sugestões concretas.
Responda em português brasileiro. Seja específico.""",
    },
}

DEFAULT_PERSONA = "mentor"

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    persona: str = Field(default=DEFAULT_PERSONA)
    stream: bool = Field(default=True)

class SessionInfo(BaseModel):
    session_id: str
    persona: str
    message_count: int

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
sessions: dict[str, list[dict]] = defaultdict(list)
session_personas: dict[str, str] = {}

# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------
class LLMClient:
    """Abstração sobre Groq e OpenAI com streaming via SSE."""

    def __init__(self) -> None:
        self.provider = self._detect_provider()

    def _detect_provider(self) -> str:
        if GROQ_API_KEY:
            logger.info("Provedor: Groq (%s)", GROQ_MODEL)
            return "groq"
        if OPENAI_API_KEY:
            logger.info("Provedor: OpenAI (%s)", OPENAI_MODEL)
            return "openai"
        logger.warning("Nenhuma API key encontrada — modo demo ativo.")
        return "demo"

    def stream(
        self,
        messages: list[dict],
        persona: str = DEFAULT_PERSONA,
    ) -> Iterator[str]:
        """Gera tokens via streaming. Yields chunks de texto."""
        system = PERSONAS.get(persona, PERSONAS[DEFAULT_PERSONA])["prompt"]
        full_messages = [{"role": "system", "content": system}] + messages

        if self.provider == "groq":
            yield from self._stream_groq(full_messages)
        elif self.provider == "openai":
            yield from self._stream_openai(full_messages)
        else:
            yield from self._stream_demo(messages[-1]["content"])

    def _stream_groq(self, messages: list[dict]) -> Iterator[str]:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "temperature": 0.7,
        }
        with requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, stream=True, timeout=30,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError):
                        continue

    def _stream_openai(self, messages: list[dict]) -> Iterator[str]:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "temperature": 0.7,
        }
        with requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers, json=payload, stream=True, timeout=30,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError):
                        continue

    def _stream_demo(self, user_msg: str) -> Iterator[str]:
        """Respostas demo para quando não há API key configurada."""
        responses = {
            "default": (
                "⚠️ **Modo demo ativo** — configure sua API key para respostas reais.\n\n"
                "Para usar o DevMentor:\n"
                "1. Acesse https://console.groq.com e crie uma conta gratuita\n"
                "2. Gere uma API key\n"
                "3. Execute: `export GROQ_API_KEY=sua_chave`\n"
                "4. Reinicie o servidor: `python main.py`\n\n"
                "O Groq é **100% gratuito** para uso pessoal e tem velocidade de geração impressionante! 🚀"
            )
        }
        text = responses["default"]
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.03)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="DevMentor API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

llm = LLMClient()


@app.get("/")
def root():
    return {
        "name": "DevMentor API",
        "provider": llm.provider,
        "personas": list(PERSONAS.keys()),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": llm.provider,
        "has_key": llm.provider != "demo",
    }


@app.get("/personas")
def get_personas():
    return {k: {"name": v["name"], "emoji": v["emoji"], "label": v["label"]}
            for k, v in PERSONAS.items()}


@app.post("/chat")
def chat(req: ChatRequest):
    """Endpoint principal — retorna streaming SSE."""
    # Garante sessão
    if req.session_id not in session_personas:
        session_personas[req.session_id] = req.persona

    persona = session_personas.get(req.session_id, req.persona)

    # Adiciona mensagem do usuário ao histórico
    history = sessions[req.session_id]
    history.append({"role": "user", "content": req.message})

    # Mantém janela de contexto
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
        sessions[req.session_id] = history

    def generate():
        full_response = ""
        try:
            for chunk in llm.stream(history, persona):
                full_response += chunk
                # SSE format
                yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"

            # Salva resposta no histórico
            sessions[req.session_id].append({
                "role": "assistant", "content": full_response
            })
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"

        except Exception as e:
            logger.error("Erro no streaming: %s", e)
            err_msg = f"Erro ao conectar com a API. Verifique sua chave. Detalhe: {str(e)}"
            yield f"data: {json.dumps({'content': err_msg, 'done': True, 'error': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Limpa histórico de uma sessão."""
    sessions.pop(session_id, None)
    session_personas.pop(session_id, None)
    return {"cleared": True, "session_id": session_id}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Retorna histórico de uma sessão."""
    return {
        "session_id": session_id,
        "persona": session_personas.get(session_id, DEFAULT_PERSONA),
        "messages": sessions.get(session_id, []),
        "count": len(sessions.get(session_id, [])),
    }