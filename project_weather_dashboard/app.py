"""
app.py
------
Dashboard interativo de dados climáticos do Brasil.
Fonte: Open Meteo API (https://open-meteo.com) — gratuita, sem autenticação.

Execução:
    streamlit run app.py

Funcionalidades:
  - Seleção de cidade e período
  - KPIs dinâmicos (temperatura, umidade, vento, chuva)
  - Gráfico de temperatura + sensação térmica ao longo do tempo
  - Comparativo de temperatura entre cidades
  - Distribuição de condições climáticas
  - Tabela de dados filtrável e exportável
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent / "src"))
from fetcher import WeatherFetcher, CITIES

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🌤️ Weather Dashboard Brasil",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Tema / CSS customizado
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Fundo principal */
    .stApp { background-color: #0e1117; }

    /* Cards de KPI */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1d2e, #16213e);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8b949e !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #21262d;
    }

    /* Títulos de seção */
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 24px 0 8px 0;
        border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
    }

    /* Rodapé */
    .footer {
        text-align: center;
        color: #3d4451;
        font-size: 0.75rem;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cache de dados — evita rebuscar a cada interação
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(cities: tuple[str, ...], past_days: int) -> pd.DataFrame:
    """Busca e cacheia dados por 1 hora."""
    fetcher = WeatherFetcher(cities=list(cities), past_days=past_days)
    return fetcher.run()


# ---------------------------------------------------------------------------
# Sidebar — filtros
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌤️ Weather Dashboard")
    st.markdown("**Fonte:** [Open Meteo API](https://open-meteo.com)")
    st.divider()

    st.markdown("### 🏙️ Cidades")
    selected_cities = st.multiselect(
        "Selecione as cidades",
        options=list(CITIES.keys()),
        default=["São Paulo", "Rio de Janeiro", "Recife", "Petrolina"],
    )

    st.markdown("### 📅 Período")
    past_days = st.slider(
        "Dias de histórico",
        min_value=1, max_value=14, value=7,
        help="Quantos dias de histórico carregar (além dos próximos 3 dias de previsão)",
    )

    st.markdown("### 🌡️ Variável principal")
    variable_options = {
        "Temperatura (°C)":       "temperatura",
        "Sensação Térmica (°C)":  "sensacao_termica",
        "Umidade (%)":            "umidade",
        "Vento (km/h)":           "vento_kmh",
        "Precipitação (mm)":      "precipitacao",
    }
    selected_var_label = st.selectbox(
        "Variável para o gráfico principal",
        options=list(variable_options.keys()),
    )
    selected_var = variable_options[selected_var_label]

    st.markdown("### 🌦️ Tipo de dado")
    data_type = st.radio(
        "Mostrar",
        options=["Histórico + Previsão", "Apenas Histórico", "Apenas Previsão"],
        index=0,
    )

    st.divider()
    st.caption("Dados atualizados a cada hora · Horário local")

# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------
if not selected_cities:
    st.warning("⚠️ Selecione pelo menos uma cidade na barra lateral.")
    st.stop()

# ---------------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------------
with st.spinner("🔄 Buscando dados climáticos..."):
    try:
        df_all = load_data(tuple(selected_cities), past_days)
    except RuntimeError as e:
        st.error(f"❌ Erro ao buscar dados: {e}")
        st.stop()

# Aplica filtro de tipo
type_map = {
    "Histórico + Previsão": ["Histórico", "Previsão"],
    "Apenas Histórico":     ["Histórico"],
    "Apenas Previsão":      ["Previsão"],
}
df = df_all[df_all["tipo"].isin(type_map[data_type])].copy()

if df.empty:
    st.warning("Sem dados para os filtros selecionados.")
    st.stop()

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.markdown("# 🌤️ Weather Dashboard Brasil")
st.markdown(
    f"**{len(selected_cities)} cidade(s)** · "
    f"**{df['datetime'].min().strftime('%d/%m/%Y')}** → "
    f"**{df['datetime'].max().strftime('%d/%m/%Y')}** · "
    f"**{len(df):,} registros horários**"
)
st.divider()

# ---------------------------------------------------------------------------
# KPIs — cidade principal (primeira selecionada)
# ---------------------------------------------------------------------------
main_city = selected_cities[0]
df_main = df[df["city"] == main_city]

st.markdown(f'<div class="section-title">📊 KPIs — {main_city}</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

avg_temp   = df_main["temperatura"].mean()
max_temp   = df_main["temperatura"].max()
min_temp   = df_main["temperatura"].min()
avg_hum    = df_main["umidade"].mean()
total_rain = df_main["precipitacao"].sum()
avg_wind   = df_main["vento_kmh"].mean()
top_cond   = df_main["condicao"].mode()[0] if not df_main.empty else "—"

k1.metric("🌡️ Temp. Média",   f"{avg_temp:.1f}°C",  f"↑{max_temp:.0f}° ↓{min_temp:.0f}°")
k2.metric("💧 Umidade Média", f"{avg_hum:.0f}%")
k3.metric("🌧️ Chuva Total",   f"{total_rain:.1f}mm")
k4.metric("💨 Vento Médio",   f"{avg_wind:.0f}km/h")
k5.metric("☁️ Condição freq.", top_cond)

st.divider()

# ---------------------------------------------------------------------------
# Gráfico 1 — Série temporal da variável selecionada
# ---------------------------------------------------------------------------
st.markdown(f'<div class="section-title">📈 {selected_var_label} ao longo do tempo</div>',
            unsafe_allow_html=True)

fig_line = px.line(
    df.sort_values("datetime"),
    x="datetime", y=selected_var,
    color="city",
    labels={"datetime": "Data/Hora", selected_var: selected_var_label, "city": "Cidade"},
    color_discrete_sequence=px.colors.qualitative.Vivid,
)

# Linha vertical separando histórico de previsão
now_ms = int(pd.Timestamp.now().timestamp() * 1000)
fig_line.add_vline(
    x=now_ms, line_dash="dash", line_color="#64748b", line_width=1.5,
    annotation_text="agora", annotation_font_color="#64748b",
)

fig_line.update_layout(
    paper_bgcolor="#0e1117", plot_bgcolor="#161b27",
    font_color="#e6edf3", legend_bgcolor="#161b27",
    xaxis=dict(gridcolor="#21262d", showgrid=True),
    yaxis=dict(gridcolor="#21262d", showgrid=True),
    hovermode="x unified",
    height=400,
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_line, use_container_width=True)

# ---------------------------------------------------------------------------
# Gráfico 2 + 3 — Comparativo entre cidades | Distribuição de condições
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-title">🏙️ Comparativo de Temperatura Média por Cidade</div>',
                unsafe_allow_html=True)

    city_avg = (
        df.groupby("city")["temperatura"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=city_avg["city"],
        x=city_avg["mean"],
        orientation="h",
        name="Média",
        marker_color="#3b82f6",
        error_x=dict(
            type="data",
            symmetric=False,
            array=city_avg["max"] - city_avg["mean"],
            arrayminus=city_avg["mean"] - city_avg["min"],
            color="#64748b",
            thickness=2,
        ),
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#161b27",
        font_color="#e6edf3",
        xaxis=dict(gridcolor="#21262d", title="Temperatura (°C)"),
        yaxis=dict(gridcolor="#21262d"),
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown('<div class="section-title">☁️ Condições Climáticas Mais Frequentes</div>',
                unsafe_allow_html=True)

    cond_counts = (
        df.groupby(["city", "condicao"])
        .size()
        .reset_index(name="horas")
        .sort_values("horas", ascending=False)
        .head(40)
    )

    fig_pie = px.treemap(
        cond_counts,
        path=["city", "condicao"],
        values="horas",
        color="horas",
        color_continuous_scale="Blues",
        labels={"horas": "Horas"},
    )
    fig_pie.update_layout(
        paper_bgcolor="#0e1117",
        font_color="#e6edf3",
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------------------------
# Gráfico 4 — Temperatura por hora do dia (heatmap)
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">🕐 Temperatura Média por Hora do Dia</div>',
            unsafe_allow_html=True)

heat_df = (
    df.groupby(["city", "hour"])["temperatura"]
    .mean()
    .reset_index()
    .pivot(index="city", columns="hour", values="temperatura")
)

fig_heat = px.imshow(
    heat_df,
    color_continuous_scale="RdYlBu_r",
    labels={"x": "Hora do Dia", "y": "Cidade", "color": "°C"},
    aspect="auto",
    text_auto=".0f",
)
fig_heat.update_layout(
    paper_bgcolor="#0e1117", plot_bgcolor="#161b27",
    font_color="#e6edf3",
    height=280,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(tickmode="linear", dtick=2),
    coloraxis_colorbar=dict(title="°C"),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# Tabela de dados filtrável
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">📋 Tabela de Dados</div>', unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    city_filter = st.multiselect("Filtrar por cidade", options=selected_cities,
                                  default=selected_cities, key="table_city")
with col_f2:
    min_temp_filter = st.number_input("Temp. mínima (°C)", value=float(df["temperatura"].min()),
                                       step=1.0)
with col_f3:
    cond_filter = st.multiselect("Filtrar por condição",
                                  options=sorted(df["condicao"].unique()),
                                  default=sorted(df["condicao"].unique()), key="table_cond")

df_table = df[
    (df["city"].isin(city_filter)) &
    (df["temperatura"] >= min_temp_filter) &
    (df["condicao"].isin(cond_filter))
].copy()

display_cols = {
    "datetime":        "Data/Hora",
    "city":            "Cidade",
    "temperatura":     "Temp (°C)",
    "sensacao_termica":"Sensação (°C)",
    "umidade":         "Umidade (%)",
    "precipitacao":    "Chuva (mm)",
    "vento_kmh":       "Vento (km/h)",
    "condicao":        "Condição",
    "tipo":            "Tipo",
}

df_display = df_table[list(display_cols.keys())].rename(columns=display_cols)
df_display["Data/Hora"] = df_display["Data/Hora"].dt.strftime("%d/%m/%Y %H:%M")

st.dataframe(
    df_display,
    use_container_width=True,
    height=350,
    hide_index=True,
)

st.caption(f"**{len(df_display):,} registros** exibidos")

# Botão de export
csv_bytes = df_display.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    label="⬇️ Exportar CSV",
    data=csv_bytes,
    file_name="weather_data.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Rodapé
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer">'
    "Dados fornecidos por <a href='https://open-meteo.com' style='color:#3b82f6'>Open Meteo API</a> · "
    "Gratuito e sem autenticação · Atualizado a cada hora"
    "</div>",
    unsafe_allow_html=True,
)