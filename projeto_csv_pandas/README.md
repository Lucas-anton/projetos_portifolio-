# 🚀 NASA Open Data Analyzer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![API](https://img.shields.io/badge/NASA-Open%20API-red?logo=nasa&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio-orange)

> Análise exploratória de dados reais da **NASA Open API** — asteroides próximos à Terra (NeoWs) e imagens astronômicas (APOD) — com visualizações profissionais em Python.

---

## 📁 Estrutura do Projeto

```
nasa_analyzer/
├── src/
│   ├── analyzer.py      # Coleta, limpeza e análise dos dados
│   └── plotter.py       # Geração de gráficos desacoplada
├── output/
│   └── plots/           # PNGs gerados automaticamente (gitignored)
├── main.py              # Ponto de entrada com CLI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart

```bash
# 1. Clone e entre na pasta
git clone https://github.com/seu-usuario/nasa-analyzer.git
cd nasa-analyzer

# 2. Crie um ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute
python main.py
```

**Sem cadastro necessário.** A `DEMO_KEY` embutida funciona imediatamente (limite: 30 req/hora por IP).

---

## 🔑 API Key (opcional — aumenta o limite)

Para subir para **1.000 req/hora**:

1. Cadastre-se gratuitamente em [api.nasa.gov](https://api.nasa.gov)
2. Exporte a chave antes de rodar:

```bash
export NASA_API_KEY=sua_chave_aqui   # Linux/macOS
set NASA_API_KEY=sua_chave_aqui      # Windows CMD
```

> ⚠️ **Nunca commite sua chave.** O `.gitignore` já protege arquivos `.env`.

---

## 🖥️ Opções de linha de comando

```bash
python main.py                        # Padrão: últimos 7 dias
python main.py --days 3               # Últimos 3 dias
python main.py --no-plots             # Apenas análise textual (mais rápido)
python main.py --output resultados/   # Diretório de saída customizado
```

---

## 📊 APIs Utilizadas

| API | Dados | Endpoint |
|-----|-------|----------|
| **NeoWs** | Asteroides próximos à Terra | `/neo/rest/v1/feed` |
| **APOD** | Imagem astronômica do dia + metadados | `/planetary/apod` |

Documentação completa: [api.nasa.gov](https://api.nasa.gov)

---

## 📈 Visualizações Geradas

| Arquivo | Conteúdo |
|---------|----------|
| `01_asteroids_per_day.png` | Contagem diária empilhada por nível de perigo |
| `02_diameter_distribution.png` | Histograma de tamanho em escala logarítmica |
| `03_velocity_vs_distance.png` | Scatter: velocidade × distância de passagem |
| `04_closest_asteroids.png` | Top 10 mais próximos (barras horizontais) |
| `05_magnitude_vs_diameter.png` | Magnitude absoluta vs diâmetro estimado |
| `06_apod_overview.png` | Tipos de mídia e tamanho das descrições APOD |

---

## 🏗️ Decisões de Arquitetura

- **Fluent API**: `.fetch().transform().analyze()` — legível e encadeável
- **`AnalysisReport` como `@dataclass`**: resultado tipado, imutável, testável
- **`NasaPlotter` desacoplado**: recebe o report, não o analyzer — sem dependência circular
- **`NasaApiClient` separado**: retry com backoff exponencial, fácil de mockar em testes
- **Sem autenticação obrigatória**: `DEMO_KEY` funciona out-of-the-box
- **Type hints completos**: compatível com `mypy`
- **Logging estruturado**: pronto para produção

---

## 📦 Dependências

```
requests>=2.31    # HTTP com retry
pandas>=2.0       # Manipulação de dados
numpy>=1.24       # Cálculos numéricos
matplotlib>=3.7   # Visualizações base
seaborn>=0.12     # Heatmaps e estilo
```

---

## 📄 Licença

MIT — veja [LICENSE](LICENSE) para detalhes.