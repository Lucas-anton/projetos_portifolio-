# 🎮 Games Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-ff4b4b?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f7931e?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio-orange)

> Análise de sentimentos em comentários reais de subreddits de games — classifica textos como **positivo**, **negativo** ou **neutro** usando NLP com TF-IDF + Logistic Regression. Entregue via **Dashboard Streamlit** e **API REST FastAPI**.

---

## 📁 Estrutura

```
sentiment_games/
├── src/
│   ├── collector.py     # Reddit API (PRAW) + dataset sintético
│   ├── model.py         # TF-IDF + Logistic Regression + predição
│   └── api.py           # Endpoints FastAPI
├── output/              # Modelo e métricas (gerados automaticamente)
├── app.py               # Dashboard Streamlit
├── main.py              # CLI: --train, --dashboard, --api
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart

```bash
# 1. Entre na pasta e crie o venv
cd sentiment_games
python -m venv venv
source venv/bin/activate

# 2. Instale dependências
pip install -r requirements.txt

# 3. Treina modelo e sobe dashboard
python main.py
```

---

## 🖥️ Modos de execução

```bash
python main.py                   # Treina + Dashboard (padrão)
python main.py --train           # Só treina o modelo
python main.py --dashboard       # Só dashboard (modelo já treinado)
python main.py --api             # Sobe API FastAPI em :8000
python main.py --train --api     # Treina e sobe API
```

---

## 🔑 Reddit API (opcional — aumenta dados reais)

```bash
# 1. Acesse https://www.reddit.com/prefs/apps → Create App → script
# 2. Exporte as credenciais:
export REDDIT_CLIENT_ID=seu_client_id
export REDDIT_CLIENT_SECRET=seu_client_secret

python main.py
```

Sem credenciais: usa dataset sintético realista (400 comentários categorizados).

---

## 🤖 Modelo NLP

| Etapa | Detalhe |
|-------|---------|
| Pré-processamento | Remove URLs, @mentions, normaliza texto |
| Vetorização | TF-IDF (uni+bigrams, 5.000 features) |
| Classificador | Logistic Regression (multinomial, balanced) |
| Classes | positive · negative · neutral |
| Léxico customizado | Palavras-chave domain-specific de games |

---

## 📡 API REST

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/predict` | Classifica um texto |
| `POST` | `/predict/batch` | Classifica até 100 textos |
| `GET`  | `/model/info` | Métricas do modelo |
| `GET`  | `/health` | Status da API |

### Exemplo

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This game is absolutely broken after the patch!"}'
```

```json
{
  "sentiment": "negative",
  "confidence": 0.89,
  "sentiment_emoji": "😤",
  "probabilities": {"negative": 0.89, "neutral": 0.08, "positive": 0.03},
  "lexicon_signals": ["❌ broken"]
}
```

---

## 📦 Dependências

```
streamlit>=1.32     scikit-learn>=1.4
fastapi>=0.110      pandas>=2.0
uvicorn             plotly>=5.18
praw>=7.7           joblib>=1.3
```