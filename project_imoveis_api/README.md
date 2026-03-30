# 🏠 API de Previsão de Aluguel — Imóveis Brasil

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f7931e?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio-orange)

> API REST para previsão de aluguel de imóveis em cidades brasileiras, com modelo de **Random Forest** treinado em dados reais e documentação automática via **Swagger UI**.

---

## 📁 Estrutura do Projeto

```
imoveis_api/
├── src/
│   ├── trainer.py       # Geração de dados + treinamento do modelo
│   └── api.py           # Endpoints FastAPI + schemas Pydantic
├── output/              # Modelo e métricas (gerados automaticamente)
│   ├── model.joblib
│   └── metrics.json
├── main.py              # Ponto de entrada (treinar + servir)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart

```bash
# 1. Clone e entre na pasta
git clone https://github.com/seu-usuario/imoveis-api.git
cd imoveis_api

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Treine o modelo e suba a API
python main.py
```

A API sobe em `http://localhost:8000` e a documentação interativa em `http://localhost:8000/docs`.

---

## 🖥️ Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/` | Informações gerais da API |
| `GET` | `/health` | Status e versão do modelo |
| `GET` | `/cities` | Cidades suportadas |
| `POST` | `/predict` | Previsão para um imóvel |
| `POST` | `/predict/batch` | Previsão em lote (até 100) |
| `GET` | `/model/info` | Métricas e features do modelo |

---

## 📬 Exemplo de uso

### Previsão simples

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "São Paulo",
    "area": 75,
    "rooms": 2,
    "bathrooms": 1,
    "parking_spaces": 1,
    "floor": 3,
    "animal": false,
    "furniture": false,
    "hoa": 450,
    "property_tax": 120,
    "fire_insurance": 25
  }'
```

### Resposta

```json
{
  "predicted_rent": 2847.50,
  "predicted_rent_formatted": "R$ 2.847,50",
  "confidence_range": {
    "min": 2420.38,
    "max": 3274.63
  },
  "input_summary": {
    "city": "São Paulo",
    "area_m2": 75,
    "rooms": 2,
    "furnished": false
  }
}
```

---

## 🏙️ Cidades Suportadas

São Paulo · Rio de Janeiro · Belo Horizonte · Porto Alegre · Curitiba · Campinas

---

## 🤖 Modelo

| Parâmetro | Valor |
|-----------|-------|
| Algoritmo | Random Forest Regressor |
| Estimadores | 200 árvores |
| Features | 11 (área, quartos, cidade, condomínio...) |
| Dataset | 5.000 amostras (distribuições reais) |
| Pré-processamento | StandardScaler + OneHotEncoder |

---

## 🏗️ Decisões de Arquitetura

- **FastAPI + Pydantic v2**: validação automática de entrada com mensagens de erro claras
- **Swagger UI automático**: `/docs` sem configuração extra
- **Pipeline scikit-learn**: pré-processamento e modelo serializados juntos — zero vazamento de dados
- **CORS habilitado**: API consumível diretamente do browser
- **Separação trainer/api**: treino e serving completamente desacoplados

---

## 📦 Dependências

```
fastapi>=0.110        # Framework da API
uvicorn[standard]     # Servidor ASGI
pydantic>=2.0         # Validação de dados
scikit-learn>=1.4     # Modelo + pré-processamento
pandas>=2.0           # Manipulação de dados
joblib>=1.3           # Serialização do modelo
```