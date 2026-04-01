# 🧑‍💻 DevMentor — Assistente de Carreira para Devs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio-orange)

> Chatbot com LLM especializado em carreira para desenvolvedores — revisa código, simula entrevistas, otimiza currículos e orienta crescimento profissional. Interface premium com **streaming em tempo real**.

---

## 📁 Estrutura

```
dev_assistant/
├── src/
│   └── api.py          # FastAPI + streaming SSE + gerenciamento de sessões
├── web/
│   └── index.html      # Interface chat premium (HTML/CSS/JS puro)
├── main.py             # Servidor + serve frontend estático
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart

```bash
# 1. Entre na pasta e crie o venv
cd dev_assistant
python -m venv venv
source venv/bin/activate

# 2. Instale dependências
pip install -r requirements.txt

# 3. Configure sua API key (gratuita)
export GROQ_API_KEY=sua_chave_aqui   # Linux/macOS
set GROQ_API_KEY=sua_chave_aqui      # Windows

# 4. Rode
python main.py
```

Acesse **http://localhost:8000/app**

---

## 🔑 Obtendo a API Key (gratuita)

1. Acesse [console.groq.com](https://console.groq.com)
2. Crie uma conta gratuita (sem cartão)
3. Vá em **API Keys → Create API Key**
4. Copie e exporte: `export GROQ_API_KEY=gsk_...`

**Sem key**: o app roda em modo demo com instruções na tela.

---

## 🤖 Personas / Modos

| Persona | Descrição |
|---------|-----------|
| 🧑‍💻 **DevMentor** | Mentor de carreira — plano de crescimento, salário, tecnologias |
| 🔍 **CodeReviewer** | Revisa código com feedback construtivo e sugestões de melhoria |
| 🎯 **TechInterviewer** | Simula entrevistas técnicas e comportamentais reais |
| 📄 **ResumeCoach** | Otimiza currículo e LinkedIn com foco em impacto e ATS |

---

## 🏗️ Arquitetura

```
Browser (HTML/JS)
    │  fetch POST /chat
    ▼
FastAPI (src/api.py)
    │  Server-Sent Events (SSE streaming)
    │  Histórico de conversa por sessão (in-memory)
    │  System prompt por persona
    ▼
Groq API / OpenAI API
    │  Llama 3.3 70B / GPT-4o-mini
    ▼
Streaming de tokens → SSE → JavaScript → renderização Markdown
```

---

## 🔧 Prompt Engineering

Cada persona tem um system prompt cuidadosamente construído com:
- **Persona rica**: experiência, especialidade, estilo de comunicação
- **Escopo definido**: o que o assistente faz e como estrutura respostas
- **Tom calibrado**: direto, honesto, encorajador, com exemplos concretos
- **Idioma**: português brasileiro natural

---

## 📦 Dependências

```
fastapi>=0.110         # API + SSE
uvicorn[standard]      # Servidor ASGI
requests>=2.31         # Chamadas para Groq/OpenAI
pydantic>=2.0          # Validação de entrada
```