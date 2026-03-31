"""
app.py
------
Dashboard interativo de análise de sentimentos em comentários de games.

Funcionalidades:
  - Análise de texto ao vivo (digita e classifica em tempo real)
  - Visão geral do dataset coletado
  - Distribuição de sentimentos por jogo e subreddit
  - Linha do tempo de sentimentos
  - Wordcloud por sentimento
  - Tabela filtrável de comentários

Execução:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from collector import RedditCollector
from model import SentimentModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🎮 Games Sentiment",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1d2e, #16213e);
        border: 1px solid #2a2d3e; border-radius: 12px; padding: 16px 20px;
    }
    [data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.82rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e6edf3 !important; font-size: 1.8rem !important; font-weight: 700 !important;
    }
    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #21262d; }
    .section-title {
        font-size: 0.85rem; font-weight: 700; color: #8b949e;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 20px 0 8px 0; border-bottom: 1px solid #21262d; padding-bottom: 6px;
    }
    .sentiment-box {
        border-radius: 12px; padding: 20px 24px; margin: 8px 0;
        border: 1px solid #2a2d3e;
    }
    .footer { text-align:center; color:#3d4451; font-size:0.75rem; margin-top:40px; }
</style>
""", unsafe_allow_html=True)

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#64748b",
}
SENTIMENT_EMOJI = {"positive": "😊", "negative": "😤", "neutral": "😐"}

PLOTLY_BASE = dict(
    paper_bgcolor="#0e1117", plot_bgcolor="#161b27",
    font_color="#e6edf3",
    margin=dict(l=0, r=0, t=30, b=0),
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> SentimentModel:
    model = SentimentModel()
    if not Path("output/sentiment_model.joblib").exists():
        model.train(n_samples=400)
    else:
        model.load()
    return model


@st.cache_data(ttl=1800, show_spinner=False)
def load_dataset(n: int) -> pd.DataFrame:
    collector = RedditCollector()
    result = collector.run(limit=n)
    df = result.df
    model = load_model()
    df = model.analyze_dataset(df)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎮 Games Sentiment")
    st.markdown("**Modelo:** TF-IDF + Logistic Regression")
    st.divider()

    st.markdown("### 📊 Dataset")
    n_comments = st.slider("Comentários a analisar", 100, 500, 300, step=50)

    st.markdown("### 🎮 Filtros")
    all_games = ["CS2", "Valorant", "Apex Legends", "League of Legends",
                 "Dota 2", "Elden Ring", "Fortnite", "Overwatch 2",
                 "The Witcher 3", "Cyberpunk 2077"]
    selected_games = st.multiselect("Jogos", all_games, default=all_games[:5])

    sentiment_filter = st.multiselect(
        "Sentimentos",
        ["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"],
    )

    st.divider()
    st.caption("Dados: Reddit API · Modelo: scikit-learn")


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
with st.spinner("🤖 Carregando modelo e coletando dados..."):
    model = load_model()
    df_all = load_dataset(n_comments)

df = df_all[
    df_all["game"].isin(selected_games) &
    df_all["sentiment"].isin(sentiment_filter)
].copy() if selected_games else df_all.copy()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🎮 Games Sentiment Analysis")
st.markdown("Análise de sentimentos em comentários reais de subreddits de games.")
st.divider()

# ---------------------------------------------------------------------------
# Análise ao vivo
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">⚡ Analise um texto agora</div>',
            unsafe_allow_html=True)

live_col1, live_col2 = st.columns([2, 1])
with live_col1:
    user_text = st.text_area(
        "Digite um comentário de game",
        placeholder="Ex: This game is absolutely broken after the latest patch...",
        height=100,
        label_visibility="collapsed",
    )
    analyze_btn = st.button("🔍 Analisar Sentimento", type="primary")

if analyze_btn and user_text.strip():
    result = model.predict(user_text)
    color = SENTIMENT_COLORS[result.sentiment]
    emoji = SENTIMENT_EMOJI[result.sentiment]

    with live_col2:
        st.markdown(f"""
        <div class="sentiment-box" style="border-color:{color}; background: linear-gradient(135deg, #1a1d2e, #16213e);">
            <div style="font-size:2.5rem;text-align:center">{emoji}</div>
            <div style="text-align:center;font-size:1.3rem;font-weight:700;color:{color};margin:8px 0">
                {result.sentiment.upper()}
            </div>
            <div style="text-align:center;color:#8b949e;font-size:0.9rem">
                Confiança: <strong style="color:{color}">{result.confidence:.1%}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Barras de probabilidade
    probs = result.probabilities
    fig_probs = go.Figure()
    for sent, prob in sorted(probs.items(), key=lambda x: x[1]):
        fig_probs.add_trace(go.Bar(
            y=[sent], x=[prob], orientation="h",
            marker_color=SENTIMENT_COLORS[sent],
            name=sent, showlegend=False,
            text=f"{prob:.1%}", textposition="outside",
        ))
    fig_probs.update_layout(
        **PLOTLY_BASE, height=160,
        xaxis=dict(range=[0, 1], gridcolor="#21262d", tickformat=".0%"),
        yaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig_probs, use_container_width=True)

    if result.lexicon_signals:
        st.caption(f"Sinais léxicos encontrados: {' · '.join(result.lexicon_signals)}")

elif analyze_btn:
    st.warning("Digite um texto para analisar.")

st.divider()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
total = len(df)
pos_pct = (df["sentiment"] == "positive").mean()
neg_pct = (df["sentiment"] == "negative").mean()
neu_pct = (df["sentiment"] == "neutral").mean()
avg_conf = df["confidence"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💬 Comentários",    f"{total:,}")
k2.metric("😊 Positivos",      f"{pos_pct:.1%}", f"{int(pos_pct*total)}")
k3.metric("😤 Negativos",      f"{neg_pct:.1%}", f"{int(neg_pct*total)}")
k4.metric("😐 Neutros",        f"{neu_pct:.1%}", f"{int(neu_pct*total)}")
k5.metric("🎯 Conf. média",    f"{avg_conf:.1%}")
st.divider()

# ---------------------------------------------------------------------------
# Gráfico 1 + 2 — Distribuição geral | Por jogo
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">📊 Distribuição de Sentimentos</div>',
                unsafe_allow_html=True)
    sent_counts = df["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentimento", "Quantidade"]
    fig_pie = px.pie(
        sent_counts, names="Sentimento", values="Quantidade",
        color="Sentimento",
        color_discrete_map=SENTIMENT_COLORS,
        hole=0.45,
    )
    fig_pie.update_traces(textinfo="percent+label", textfont_size=12)
    fig_pie.update_layout(**PLOTLY_BASE, height=320,
                          legend=dict(bgcolor="#161b27"))
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.markdown('<div class="section-title">🎮 Sentimento por Jogo</div>',
                unsafe_allow_html=True)
    game_sent = (
        df.groupby(["game", "sentiment"])
        .size().reset_index(name="count")
    )
    fig_game = px.bar(
        game_sent, x="game", y="count", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        barmode="stack",
        labels={"game": "Jogo", "count": "Comentários", "sentiment": "Sentimento"},
    )
    fig_game.update_layout(
        **PLOTLY_BASE, height=320,
        xaxis=dict(tickangle=25, gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        legend=dict(bgcolor="#161b27"),
    )
    st.plotly_chart(fig_game, use_container_width=True)

# ---------------------------------------------------------------------------
# Gráfico 3 — Timeline de sentimentos
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">📅 Linha do Tempo de Sentimentos</div>',
            unsafe_allow_html=True)

df_time = df.copy()
df_time["hour"] = df_time["created_at"].dt.floor("6h")
timeline = (
    df_time.groupby(["hour", "sentiment"])
    .size().reset_index(name="count")
    .sort_values("hour")
)

fig_time = px.area(
    timeline, x="hour", y="count", color="sentiment",
    color_discrete_map=SENTIMENT_COLORS,
    labels={"hour": "Período", "count": "Comentários", "sentiment": "Sentimento"},
)
fig_time.update_layout(
    **PLOTLY_BASE, height=300,
    xaxis=dict(gridcolor="#21262d"),
    yaxis=dict(gridcolor="#21262d"),
    legend=dict(bgcolor="#161b27"),
)
st.plotly_chart(fig_time, use_container_width=True)

# ---------------------------------------------------------------------------
# Gráfico 4 + 5 — Distribuição de confiança | Score Reddit vs Sentimento
# ---------------------------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-title">🎯 Distribuição de Confiança do Modelo</div>',
                unsafe_allow_html=True)
    fig_conf = px.histogram(
        df, x="confidence", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        nbins=20, barmode="overlay", opacity=0.75,
        labels={"confidence": "Confiança", "count": "Comentários"},
    )
    fig_conf.update_layout(
        **PLOTLY_BASE, height=300,
        xaxis=dict(tickformat=".0%", gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        legend=dict(bgcolor="#161b27"),
    )
    st.plotly_chart(fig_conf, use_container_width=True)

with col4:
    st.markdown('<div class="section-title">🏆 Score Reddit vs Sentimento</div>',
                unsafe_allow_html=True)
    fig_score = px.box(
        df[df["score"] < 500], x="sentiment", y="score",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        labels={"sentiment": "Sentimento", "score": "Score (upvotes)"},
    )
    fig_score.update_layout(
        **PLOTLY_BASE, height=300,
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        showlegend=False,
    )
    st.plotly_chart(fig_score, use_container_width=True)

# ---------------------------------------------------------------------------
# Tabela filtrável
# ---------------------------------------------------------------------------
st.divider()
st.markdown('<div class="section-title">📋 Comentários Analisados</div>',
            unsafe_allow_html=True)

col_f1, col_f2 = st.columns(2)
with col_f1:
    min_conf = st.slider("Confiança mínima", 0.0, 1.0, 0.5, step=0.05)
with col_f2:
    search = st.text_input("Buscar no texto", placeholder="Ex: broken, amazing...")

df_table = df[df["confidence"] >= min_conf].copy()
if search:
    df_table = df_table[df_table["text"].str.contains(search, case=False, na=False)]

display = df_table[["text", "game", "sentiment", "confidence", "score"]].rename(columns={
    "text": "Comentário", "game": "Jogo",
    "sentiment": "Sentimento", "confidence": "Confiança", "score": "Score",
}).head(200)
display["Confiança"] = display["Confiança"].map("{:.1%}".format)

st.dataframe(display, use_container_width=True, height=350, hide_index=True)
st.caption(f"**{len(df_table):,} comentários** com confiança ≥ {min_conf:.0%}")

csv = df_table[["text","game","sentiment","confidence","score"]].to_csv(
    index=False, encoding="utf-8-sig"
).encode("utf-8-sig")
st.download_button("⬇️ Exportar CSV", csv, "sentiment_analysis.csv", "text/csv")

st.markdown(
    '<div class="footer">Dados: Reddit API · Modelo: TF-IDF + Logistic Regression · Portfólio Data Science</div>',
    unsafe_allow_html=True,
)