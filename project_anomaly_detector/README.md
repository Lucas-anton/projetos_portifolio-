# ⚡ FraudScan — Detector de Anomalias Financeiras

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f7931e?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio-orange)

> Detector híbrido de anomalias em transações de cartão de crédito usando **Isolation Forest + Autoencoder (NumPy puro)**, com dashboard fintech premium e API REST.

---

## 📁 Estrutura

```
anomaly_detector/
├── src/
│   ├── model.py    # Geração de dados + IF + Autoencoder + Ensemble
│   └── api.py      # FastAPI: /transactions, /anomalies, /predict, /stats
├── web/
│   └── index.html  # Dashboard HTML premium (DM Mono + Clash Display)
├── output/         # Modelo e dataset (gerados automaticamente)
├── main.py         # CLI: --train, --serve
└── requirements.txt
```

---


## 📊 Dashboards e Resultados

Os resultados gerados pelo modelo, incluindo visualizações e análises das transações, estão disponíveis na pasta:

Nela você encontrará:

- 📈 Visualizações de anomalias detectadas  
- 📊 Estatísticas agregadas do modelo  
- 🕒 Análise temporal das transações  
- 🚨 Exemplos de fraudes identificadas  

> Esses dashboards complementam a API e permitem uma análise mais visual e exploratória dos dados.

## ⚡ Quickstart

```bash
cd anomaly_detector
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py          # treina + sobe dashboard
```

**Dashboard:** http://localhost:8000/app  
**API Docs:** http://localhost:8000/docs

---

## 🤖 Modelo Híbrido

### Isolation Forest
- Detecta anomalias globais por isolamento de pontos raros em árvores aleatórias
- Rápido, sem suposição sobre distribuição dos dados
- 200 estimadores, contamination=2%

### Autoencoder (NumPy puro — zero TensorFlow)
- Aprende a reconstruir padrões de transações normais
- Alto erro de reconstrução = transação atípica
- Arquitetura: 13→8→4→8→13, ReLU + Sigmoid

### Ensemble
```
score = α × IF_score + (1-α) × AE_score
threshold = percentil 98 dos dados de treino
```

---

## 📊 Padrões de Fraude Simulados

| Padrão | Descrição |
|--------|-----------|
| Valor muito alto | Compra acima do histórico do usuário |
| Múltiplas transações | Várias compras em < 1 hora |
| Horário atípico | Madrugada (1h–5h) |
| Localização inconsistente | Distância > 300km de casa |
| Valor fracionado | Structuring próximo a R$ 5.000 |

---

## 📡 API Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/health` | Status + métricas do modelo |
| `GET` | `/anomalies` | Lista anomalias detectadas |
| `GET` | `/stats` | Estatísticas agregadas |
| `GET` | `/timeline` | Volume por dia/hora |
| `POST` | `/predict` | Analisa nova transação |

---

## 📦 Dependências

```
scikit-learn>=1.4    fastapi>=0.110
pandas>=2.0          uvicorn
numpy>=1.24          pydantic>=2.0
```

> **Autoencoder implementado em NumPy puro** — sem TensorFlow, sem PyTorch.