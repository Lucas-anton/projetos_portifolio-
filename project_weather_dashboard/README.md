# 🌤️ Weather Dashboard Brasil

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-ff4b4b?logo=streamlit&logoColor=white)
![API](https://img.shields.io/badge/Open%20Meteo-API%20Gratuita-green)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio-orange)

> Dashboard interativo de dados climáticos de cidades brasileiras, com dados reais da **Open Meteo API** — 100% gratuita, sem autenticação.

---

## 📁 Estrutura do Projeto

```
weather_dashboard/
├── src/
│   └── fetcher.py       # Cliente da Open Meteo API + consolidação de dados
├── app.py               # Dashboard Streamlit (ponto de entrada)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart

```bash
# 1. Entre na pasta
cd weather_dashboard

# 2. Ative o ambiente virtual
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o dashboard
streamlit run app.py
```

O dashboard abre automaticamente em `http://localhost:8501`.

---

## 🖥️ Funcionalidades

### Filtros interativos (sidebar)
- **Cidades**: selecione múltiplas cidades brasileiras
- **Período**: 1 a 14 dias de histórico + 3 dias de previsão
- **Variável principal**: temperatura, sensação térmica, umidade, vento ou chuva
- **Tipo de dado**: histórico, previsão ou ambos

### KPIs dinâmicos
- Temperatura média, máxima e mínima
- Umidade média
- Chuva total acumulada
- Velocidade média do vento
- Condição climática mais frequente

### Visualizações
| Gráfico | Descrição |
|---------|-----------|
| Série temporal | Variável selecionada ao longo do tempo por cidade |
| Barras horizontais | Comparativo de temperatura média entre cidades |
| Treemap | Distribuição de condições climáticas por cidade |
| Heatmap | Temperatura média por hora do dia × cidade |

### Tabela de dados
- Filtrável por cidade, temperatura mínima e condição climática
- Exportável como CSV com um clique

---

## 🌐 API Utilizada

**Open Meteo** — [open-meteo.com](https://open-meteo.com)
- Gratuita e sem autenticação
- Dados horários: temperatura, umidade, precipitação, vento, código de tempo WMO
- Histórico de até 92 dias + previsão de até 16 dias

### Cidades disponíveis
São Paulo · Rio de Janeiro · Brasília · Fortaleza · Manaus · Porto Alegre · Salvador · Recife · Petrolina · Curitiba

---

## 🏗️ Decisões de Arquitetura

- **`@st.cache_data(ttl=3600)`**: dados cacheados por 1 hora — evita rebuscar a cada interação
- **`WeatherFetcher` desacoplado do `app.py`**: lógica de coleta separada, fácil de testar
- **Plotly** para gráficos interativos (zoom, hover, export de imagem)
- **CSS customizado** para identidade visual coerente com tema escuro

---

## 📦 Dependências

```
streamlit>=1.32   # Framework do dashboard
pandas>=2.0       # Manipulação de dados
requests>=2.31    # HTTP com retry
plotly>=5.18      # Gráficos interativos
```