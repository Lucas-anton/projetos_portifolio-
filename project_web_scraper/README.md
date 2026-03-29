# 📰 Hacker News Scraper & Analyzer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![API](https://img.shields.io/badge/Hacker%20News-API%20Oficial-ff6600?logo=ycombinator&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio-orange)

> Coleta, analisa e visualiza notícias do **Hacker News** usando a API oficial — sem autenticação, sem bloqueios, dados reais.

---

## 📁 Estrutura do Projeto

```
hn_scraper/
├── src/
│   ├── scraper.py       # Coleta paralela via ThreadPoolExecutor
│   ├── analyzer.py      # Limpeza, métricas e persistência (CSV + JSON)
│   └── plotter.py       # 6 visualizações profissionais
├── output/              # Gerado automaticamente (gitignored)
│   ├── data/            # CSVs e JSONs com timestamp
│   ├── plots/           # PNGs dos gráficos
│   └── report_*.txt     # Relatórios textuais
├── main.py              # Ponto de entrada com CLI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart

```bash
# 1. Clone e entre na pasta
git clone https://github.com/seu-usuario/hn-scraper.git
cd hn-scraper

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute
python main.py
```

**Sem cadastro ou chave de API necessários.** A API do HN é 100% pública.

---

## 🖥️ Opções de linha de comando

```bash
python main.py                           # Top 100 stories (padrão)
python main.py --limit 300               # Coleta 300 stories
python main.py --category newstories     # Notícias mais recentes
python main.py --category beststories    # Melhores de todos os tempos
python main.py --no-plots                # Só análise textual
python main.py --output resultados/      # Diretório customizado
```

### Categorias disponíveis

| Categoria | Descrição |
|-----------|-----------|
| `topstories` | Mais votadas agora (padrão) |
| `newstories` | Mais recentes |
| `beststories` | Melhores históricos |
| `askstories` | Perguntas Ask HN |
| `showstories` | Projetos Show HN |

---

## 📊 Visualizações Geradas

| Arquivo | Conteúdo |
|---------|----------|
| `01_top_stories.png` | Top 15 stories por score (barras horizontais) |
| `02_score_distribution.png` | Distribuição de scores — linear e logarítmica |
| `03_score_by_hour.png` | Score médio e volume por hora do dia (UTC) |
| `04_top_domains.png` | Top 10 domínios mais compartilhados |
| `05_score_vs_comments.png` | Scatter: score × comentários por faixa |
| `06_tiers_and_authors.png` | Pizza por tier de score + top autores |

---

## 💾 Dados Salvos

A cada execução, três arquivos são gerados em `output/` com timestamp:

```
output/
├── data/
│   ├── hn_stories_20260328_120000.csv   # Abre no Excel / Google Sheets
│   └── hn_stories_20260328_120000.json  # Pronto para pipelines downstream
├── plots/
│   └── *.png
└── report_20260328_120000.txt           # Relatório textual completo
```

---

## 🏗️ Decisões de Arquitetura

- **`HackerNewsClient` separado**: toda lógica HTTP isolada — retry com backoff exponencial, fácil de mockar
- **Coleta paralela**: `ThreadPoolExecutor` com 10 workers — reduz tempo de ~100s para ~5s
- **`ScrapeResult` → `AnalysisReport`**: pipeline tipado com `@dataclass`
- **`HackerNewsAnalyzer` desacoplado do `Plotter`**: sem dependência circular
- **Dupla persistência**: CSV para análise manual, JSON para pipelines
- **Type hints completos** em todos os módulos

---

## 📦 Dependências

```
requests>=2.31    # HTTP com retry
pandas>=2.0       # Manipulação de dados
numpy>=1.24       # Cálculos numéricos
matplotlib>=3.7   # Visualizações
```

---

## 📄 Licença

MIT — dados públicos da [Hacker News API](https://github.com/HackerNews/API).