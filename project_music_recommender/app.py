"""
app.py
------
Dashboard interativo do sistema de recomendação de músicas.

Execução:
    streamlit run app.py

Variável de ambiente (opcional):
    LASTFM_API_KEY   → Usa dados reais do Last.fm
                       Sem chave: usa catálogo sintético enriquecido (50 músicas)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fetcher import MusicFetcher
from recommender import HybridRecommender

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🎵 Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }

    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1d2e, #16213e);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.82rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.8rem !important; font-weight: 700 !important; }

    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #21262d; }

    .rec-card {
        background: linear-gradient(135deg, #1a1d2e, #16213e);
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .rec-title { font-size: 1rem; font-weight: 700; color: #e6edf3; }
    .rec-artist { font-size: 0.85rem; color: #8b949e; margin-top: 2px; }
    .rec-genre { font-size: 0.75rem; color: #3b82f6; margin-top: 4px; }
    .score-bar { height: 4px; border-radius: 2px; margin-top: 8px; }

    .section-title {
        font-size: 0.85rem; font-weight: 700; color: #8b949e;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 20px 0 8px 0; border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
    }
    .footer { text-align: center; color: #3d4451; font-size: 0.75rem; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0e1117", plot_bgcolor="#161b27",
    font_color="#e6edf3",
    xaxis=dict(gridcolor="#21262d"),
    yaxis=dict(gridcolor="#21262d"),
    margin=dict(l=0, r=0, t=30, b=0),
)

GENRE_COLORS = {
    "pop": "#3b82f6", "rock": "#ef4444", "hip-hop": "#f59e0b",
    "latin": "#22c55e", "r&b": "#a855f7", "indie": "#06b6d4",
    "electronic": "#f97316", "unknown": "#64748b",
}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_recommender() -> HybridRecommender:
    fetcher = MusicFetcher(n_tracks=50, n_users=200)
    dataset = fetcher.run()
    rec = HybridRecommender(dataset, alpha=0.5)
    rec.fit()
    return rec


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎵 Music Recommender")
    st.markdown("**Algoritmo:** Híbrido (Item + User Based)")
    st.divider()

    st.markdown("### ⚙️ Configurações")
    mode = st.radio(
        "Modo de recomendação",
        ["🎵 Por Música (Item-Based)",
         "👤 Por Usuário (User-Based)",
         "🔀 Híbrido"],
        index=2,
    )

    n_recs = st.slider("Número de recomendações", 3, 15, 8)

    alpha = st.slider(
        "Peso Item-Based (α)",
        0.0, 1.0, 0.5, step=0.1,
        help="α=1.0 → só Item-Based | α=0.0 → só User-Based",
        disabled="Híbrido" not in mode,
    )

    st.divider()
    st.caption("Dados: Last.fm · Modelo: TF-IDF + Cosine Similarity")


# ---------------------------------------------------------------------------
# Carrega modelo
# ---------------------------------------------------------------------------
with st.spinner("🎵 Carregando sistema de recomendação..."):
    rec = load_recommender()

tracks_df = rec.get_all_tracks()
all_users = rec.get_all_users()

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.markdown("# 🎵 Sistema de Recomendação de Músicas")
st.markdown("Recomendações personalizadas usando **Filtragem Colaborativa** e **Content-Based Filtering**.")
st.divider()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("🎵 Músicas no catálogo", f"{len(tracks_df)}")
k2.metric("👤 Usuários simulados", f"{len(all_users)}")
k3.metric("⭐ Ratings totais", f"{len(rec.ratings):,}")
k4.metric("🎸 Gêneros", f"{tracks_df['genre'].nunique()}")
st.divider()

# ---------------------------------------------------------------------------
# Seletores de entrada
# ---------------------------------------------------------------------------
col_input1, col_input2 = st.columns(2)

with col_input1:
    st.markdown('<div class="section-title">🎵 Selecione uma música</div>',
                unsafe_allow_html=True)
    track_options = [
        f"{r['title']} — {r['artist']}"
        for _, r in tracks_df.iterrows()
    ]
    selected_track_str = st.selectbox("Música de referência", track_options, label_visibility="collapsed")
    sel_title, sel_artist = selected_track_str.split(" — ", 1)

with col_input2:
    st.markdown('<div class="section-title">👤 Selecione um usuário</div>',
                unsafe_allow_html=True)
    selected_user = st.selectbox("Usuário", all_users, label_visibility="collapsed",
                                  disabled="Item-Based" in mode)

st.divider()

# ---------------------------------------------------------------------------
# Gera recomendações
# ---------------------------------------------------------------------------
recommendations = []
error_msg = None

try:
    if "Item-Based" in mode:
        recommendations = rec.recommend_by_track(sel_title, sel_artist, n=n_recs)
    elif "User-Based" in mode:
        recommendations = rec.recommend_by_user(selected_user, n=n_recs)
    else:
        recommendations = rec.recommend_hybrid(
            selected_user, sel_title, sel_artist, n=n_recs, alpha=alpha
        )
except ValueError as e:
    error_msg = str(e)

if error_msg:
    st.error(f"❌ {error_msg}")
    st.stop()

# ---------------------------------------------------------------------------
# Resultado — Cards + Gráfico lado a lado
# ---------------------------------------------------------------------------
col_cards, col_chart = st.columns([1, 1])

with col_cards:
    st.markdown('<div class="section-title">🎯 Recomendações</div>',
                unsafe_allow_html=True)

    for i, r in enumerate(recommendations, 1):
        color = GENRE_COLORS.get(r.genre, "#64748b")
        bar_width = int(r.score * 100)
        st.markdown(f"""
        <div class="rec-card">
            <span style="color:{color};font-weight:700;font-size:0.8rem">#{i}</span>
            <div class="rec-title">{r.title}</div>
            <div class="rec-artist">{r.artist}</div>
            <div class="rec-genre">● {r.genre.upper()}</div>
            <div style="display:flex;align-items:center;gap:8px;margin-top:8px;">
                <div style="flex:1;background:#21262d;border-radius:2px;height:4px;">
                    <div style="width:{bar_width}%;background:{color};height:4px;border-radius:2px;"></div>
                </div>
                <span style="color:#8b949e;font-size:0.75rem">{r.score:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_chart:
    st.markdown('<div class="section-title">📊 Score das Recomendações</div>',
                unsafe_allow_html=True)

    df_recs = pd.DataFrame([{
        "Título": f"{r.title[:25]}…" if len(r.title) > 25 else r.title,
        "Score": r.score,
        "Item-Based": r.item_score,
        "User-Based": r.user_score,
        "Gênero": r.genre,
    } for r in recommendations])

    if "Híbrido" in mode:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_recs["Título"][::-1], x=df_recs["Item-Based"][::-1],
            name="Item-Based", orientation="h",
            marker_color="#3b82f6", opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            y=df_recs["Título"][::-1], x=df_recs["User-Based"][::-1],
            name="User-Based", orientation="h",
            marker_color="#f59e0b", opacity=0.85,
        ))
        fig.update_layout(barmode="stack", **PLOTLY_LAYOUT,
                          height=420, legend=dict(bgcolor="#161b27"))
    else:
        colors = [GENRE_COLORS.get(g, "#64748b") for g in df_recs["Gênero"][::-1]]
        fig = go.Figure(go.Bar(
            y=df_recs["Título"][::-1], x=df_recs["Score"][::-1],
            orientation="h", marker_color=colors, opacity=0.85,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=420)

    fig.update_xaxes(range=[0, 1], title="Score")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Seção inferior — Exploração do catálogo
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-title">🗺️ Mapa de Similaridade entre Músicas</div>',
                unsafe_allow_html=True)

    sim_df = rec.get_similarity_heatmap(n=12)
    fig_heat = px.imshow(
        sim_df, color_continuous_scale="Blues",
        zmin=0, zmax=1,
        labels={"color": "Similaridade"},
    )
    fig_heat.update_layout(
    paper_bgcolor="#0e1117", plot_bgcolor="#161b27",
    font_color="#e6edf3",
    height=380,
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(tickangle=45, tickfont_size=9, gridcolor="#21262d"),
    yaxis=dict(tickfont_size=9, gridcolor="#21262d"),
    coloraxis_colorbar=dict(title="Sim."),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_right:
    st.markdown('<div class="section-title">🎸 Distribuição de Gêneros no Catálogo</div>',
                unsafe_allow_html=True)

    genre_counts = tracks_df["genre"].value_counts().reset_index()
    genre_counts.columns = ["Gênero", "Músicas"]
    fig_genre = px.bar(
        genre_counts, x="Gênero", y="Músicas",
        color="Gênero",
        color_discrete_map=GENRE_COLORS,
    )
    fig_genre.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
    st.plotly_chart(fig_genre, use_container_width=True)

# ---------------------------------------------------------------------------
# Histórico do usuário selecionado
# ---------------------------------------------------------------------------
if "User-Based" in mode or "Híbrido" in mode:
    st.divider()
    st.markdown(f'<div class="section-title">📋 Histórico de {selected_user}</div>',
                unsafe_allow_html=True)

    history = rec.get_user_history(selected_user)
    if not history.empty:
        display = history[["title", "artist", "genre", "rating"]].rename(columns={
            "title": "Música", "artist": "Artista",
            "genre": "Gênero", "rating": "⭐ Rating",
        })
        st.dataframe(display, use_container_width=True, height=250, hide_index=True)
    else:
        st.info("Usuário sem histórico de ratings.")

# ---------------------------------------------------------------------------
# Tabela completa do catálogo
# ---------------------------------------------------------------------------
with st.expander("📀 Ver catálogo completo"):
    cat_display = tracks_df[["title", "artist", "genre", "subgenre", "popularity"]].rename(columns={
        "title": "Música", "artist": "Artista", "genre": "Gênero",
        "subgenre": "Subgênero", "popularity": "🔥 Popularidade",
    })
    st.dataframe(cat_display, use_container_width=True, hide_index=True)

st.markdown(
    '<div class="footer">Algoritmo híbrido: TF-IDF Content-Based + Collaborative Filtering · '
    'Dados: Last.fm API · Portfólio Data Science</div>',
    unsafe_allow_html=True,
)