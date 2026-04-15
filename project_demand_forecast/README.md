# 📈 DemandIQ — Previsão de Demanda E-commerce

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GBM-f7931e?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20%E2%98%85-gold)

> **Projeto Integrador** — Pipeline completo de ML End-to-End: coleta dados de produtos, gera histórico de vendas com padrões realistas, treina modelo de **Gradient Boosting** com feature engineering de séries temporais, expõe via **FastAPI** e exibe em **dashboard premium** com previsões interativas.

---

## 📁 Estrutura

```
demand_forecast/
├── src/
│   ├── collector.py   # Gerador de dados + Open Food Facts API
│   ├── model.py       # GBM + feature engineering + previsão
│   └── api.py         # FastAPI: histórico, previsão, KPIs
├── web/
│   └── index.html     # Dashboard analytics (Instrument Serif + JetBrains Mono)
├── output/            # Modelos e dados (gerado automaticamente)
├── main.py            # CLI: pipeline completo
└── requirements.txt
```

---

## ⚡ Quickstart

```bash
cd demand_forecast
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Dashboard:** http://localhost:8000/app

---

## 🔄 Pipeline End-to-End

```
SalesGenerator
     │  365 dias × 15 produtos × padrões reais
     ▼
Feature Engineering
     │  lags, médias móveis, Fourier, tendência
     ▼
GradientBoostingRegressor (por produto)
     │  80% treino / 20% validação
     ▼
output/forecast_model.pkl + sales.pkl
     │
     ▼
FastAPI (/products, /forecast, /dashboard)
     │
     ▼
Dashboard HTML (3 abas: Overview · Previsão · Produtos)
```

---

## 🤖 Modelo de ML

### Features Engineered
| Feature | Descrição |
|---------|-----------|
| `lag_1..28` | Demanda dos últimos 1, 7, 14, 21, 28 dias |
| `ma_7..30` | Média móvel 7, 14 e 30 dias |
| `std_7` | Desvio padrão 7 dias (volatilidade) |
| `sin/cos_week` | Sazonalidade semanal via Fourier |
| `sin/cos_month` | Sazonalidade mensal via Fourier |
| `trend` | Tendência global normalizada |
| `is_promo` | Flag de promoção |

### Padrões Simulados
- Sazonalidade semanal (pico fim de semana)
- Sazonalidade mensal (dezembro +35%)
- Tendência individual por produto
- Promoções aleatórias (+40-80% demanda)
- Ruptura de estoque (<1% dos dias)

---

## 📊 Dashboard — 3 Abas

| Aba | Conteúdo |
|-----|----------|
| **Overview** | KPIs + série temporal + categorias + padrão semanal/mensal |
| **Previsão** | Histórico + forecast 30 dias + intervalo de confiança por produto |
| **Produtos** | Tabela completa com métricas de modelo por SKU |

---

## 📦 Dependências

```
scikit-learn>=1.4    fastapi>=0.110
pandas>=2.0          uvicorn
numpy>=1.24          requests>=2.31
```