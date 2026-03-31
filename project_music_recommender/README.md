# 🎵 Sistema de Recomendação de Músicas

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-ff4b4b?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f7931e?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio-orange)

> Sistema híbrido de recomendação de músicas combinando **Content-Based Filtering** (TF-IDF + similaridade de cosseno) e **Collaborative Filtering** (User-Based), com dashboard interativo em Streamlit.

---

## 📁 Estrutura do Projeto

```
music_recommender/
├── src/
│   ├── fetcher.py        # Coleta dados da Last.fm API (ou catálogo sintético)
│   └── recommender.py    # Motor híbrido: Item-Based + User-Based
├── app.py                # Dashboard Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart

```bash
# 1. Entre na pasta
cd music_recommender

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o dashboard
streamlit run app.py
```

Acessa em `http://localhost:8501`. **Sem cadastro necessário** — usa catálogo embutido.

---

## 🔑 Last.fm API Key (opcional)

Para usar dados reais do Last.fm:

1. Cadastre-se em [last.fm/api](https://www.last.fm/api/account/create) (gratuito)
2. Exporte a chave antes de rodar:

```bash
export LASTFM_API_KEY=sua_chave_aqui
streamlit run app.py
```

---

## 🤖 Algoritmos

### Item-Based (Content-Based Filtering)
- Representa cada música como vetor TF-IDF de suas tags e gêneros
- Calcula similaridade de cosseno entre todos os pares de músicas
- Recomenda as músicas com maior similaridade à selecionada

### User-Based (Collaborative Filtering)
- Constrói matriz usuário × música com ratings (1–5)
- Calcula similaridade de cosseno entre perfis de usuários
- Recomenda músicas avaliadas positivamente por usuários similares

### Híbrido
```
score = α × item_score + (1 - α) × user_score
```
- `α = 1.0` → só Item-Based
- `α = 0.0` → só User-Based
- `α = 0.5` → peso igual (padrão)

---

## 🖥️ Funcionalidades do Dashboard

- **3 modos**: Item-Based, User-Based e Híbrido
- **Slider de α**: ajuste em tempo real do peso de cada abordagem
- **Cards de recomendação** com score visual e gênero colorido
- **Gráfico de scores** empilhado (Item vs User) no modo híbrido
- **Heatmap de similaridade** entre músicas do catálogo
- **Histórico do usuário** selecionado
- **Catálogo completo** expansível

---

## 📦 Dependências

```
streamlit>=1.32     # Dashboard interativo
scikit-learn>=1.4   # TF-IDF + cosine similarity
scipy>=1.12         # Sparse matrix para user-based
pandas>=2.0         # Manipulação de dados
plotly>=5.18        # Gráficos interativos
requests>=2.31      # Last.fm API
```