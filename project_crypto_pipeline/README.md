# ₿ CryptoFlow — Pipeline de Dados Automatizado

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)
![API](https://img.shields.io/badge/CoinGecko-API%20Gratuita-8DC63F)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio-orange)

> Pipeline automatizado que coleta preços reais de criptomoedas da **CoinGecko API** a cada 5 minutos, persiste no **SQLite** e serve um **dashboard HTML premium** com gráficos em tempo real, alertas e histórico completo.

---

## 📁 Estrutura

```
crypto_pipeline/
├── src/
│   └── pipeline.py      # Collector → Processor → SQLite → JSON export
├── web/
│   └── index.html       # Dashboard HTML/CSS/JS (zero dependências frontend)
├── output/              # Gerado automaticamente
│   └── crypto.db        # Banco SQLite com histórico completo
├── main.py              # CLI: pipeline + servidor web
├── requirements.txt     # Só `requests` — stdlib para o resto
└── README.md
```

---

## ⚡ Quickstart

```bash
# 1. Entre na pasta e crie o venv
cd crypto_pipeline
python -m venv venv
source venv/bin/activate

# 2. Instale (só requests!)
pip install -r requirements.txt

# 3. Rode pipeline + dashboard
python main.py
```

Acesse **http://localhost:8080** — o dashboard atualiza automaticamente a cada 5 minutos.

---

## 🖥️ Modos de execução

```bash
python main.py                    # Pipeline + Dashboard (padrão)
python main.py --once             # Coleta uma vez e exporta JSON
python main.py --pipeline         # Só o loop de coleta
python main.py --server           # Só o servidor web
python main.py --interval 60      # Coleta a cada 60s (dev/teste)
python main.py --port 9000        # Porta customizada
```

---

## 🔄 Fluxo do Pipeline

```
CoinGecko API
     │
     ▼
  Collector          → busca preços, volume, market cap, variações
     │
     ▼
  Processor          → limpa, enriquece, adiciona timestamp
     │
     ▼
  SQLite (crypto.db) → persiste histórico completo
     │
     ▼
  JSON Export        → web/data.json (consumido pelo dashboard)
     │
     ▼
  Dashboard          → gráficos em tempo real + alertas + tabela
```

---

## 📊 Dashboard

| Componente | Descrição |
|------------|-----------|
| KPI Cards | Preço atual + variação 24h + sparkline por moeda |
| Alertas | Banner automático para variações > 5% / 10% |
| Gráfico principal | Histórico de preço da moeda selecionada |
| Volume 24h | Barras comparativas entre moedas |
| Variação 24h | Comparativo colorido (verde/vermelho) |
| Tabela | Histórico completo do banco com range do dia |
| Log | Registro de execuções do pipeline |
| Countdown | Timer visual para próxima coleta |

---

## 🗄️ Banco de Dados (SQLite)

```sql
-- Tabela principal
CREATE TABLE prices (
    id           INTEGER PRIMARY KEY,
    coin_id      TEXT,
    symbol       TEXT,
    price_usd    REAL,
    market_cap   REAL,
    volume_24h   REAL,
    change_1h    REAL,
    change_24h   REAL,
    change_7d    REAL,
    high_24h     REAL,
    low_24h      REAL,
    collected_at TEXT
);

-- Log de execuções
CREATE TABLE pipeline_runs (
    id          INTEGER PRIMARY KEY,
    ran_at      TEXT,
    coins_saved INTEGER,
    status      TEXT,
    error       TEXT
);
```

---

## 🏗️ Destaques de Engenharia

- **Zero dependências frontend** — HTML/CSS/JS puro, sem npm, sem build
- **SQLite nativo** — stdlib Python, sem ORM, sem servidor
- **Servidor embutido** — `http.server` da stdlib, sem Flask/FastAPI
- **Threading** — pipeline e servidor rodam em threads separadas
- **Retry com backoff** — resistente a falhas de rede
- **Rate limit handling** — detecta HTTP 429 e aguarda automaticamente
- **`requirements.txt` mínimo** — só `requests`, todo o resto é stdlib

---

## 📦 Dependências

```
requests>=2.31    # Único pacote externo — todo o resto é stdlib Python
```