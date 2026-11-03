"""Plataforma de Telemetria e Saúde de Baterias de VEs (Streamlit).

Dashboard 1: laboratório de distribuição e correlação.
Dashboard 2: KPIs e gestão de frota (público-alvo: gestores de frota e engenharia de BMS).
"""
import glob
import logging
import os

import kagglehub
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="EDA — Saúde de Baterias de VEs",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

COR_CONTEXTO = "#38bdf8"   # Saudável
COR_DESTAQUE = "#7c3aed"   # Degradada
COR_ALERTA = "#ef4444"
PALETA_ESTADO = {"Saudável": COR_CONTEXTO, "Degradada": COR_DESTAQUE}
ORDEM_ESTADO = {"estado_bateria": ["Saudável", "Degradada"]}
LIMITE_SOH = 80.0
MAX_HISTOGRAMAS = 12

KAGGLE_DATASET = "srisyra02/ev-battery-health-prediction-dataset-20k"
# Identificadores que nunca devem chegar à interface (privacidade).
COLS_SENSIVEIS = ["vehicle_id", "battery_serial"]


@st.cache_data(show_spinner="A descarregar e a preparar o dataset do Kaggle...")
def carregar_dados() -> pd.DataFrame:
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("Nenhum CSV encontrado no dataset descarregado.")

    df = pd.read_csv(csv_files[0])

    # Alvo derivado do SoH (limiar de fim de vida útil = 80%), como no notebook.
    soh_col = next((c for c in ("state_of_health", "battery_health_percent") if c in df.columns), None)
    if soh_col is not None:
        df["estado_bateria"] = np.where(df[soh_col] < LIMITE_SOH, "Degradada", "Saudável")
        df.loc[df[soh_col].isna(), "estado_bateria"] = np.nan
    elif "battery_failure" in df.columns:
        df["estado_bateria"] = np.where(df["battery_failure"] == 1, "Degradada", "Saudável")
    else:
        df["estado_bateria"] = "Saudável"

    # Remove identificadores antes de qualquer visualização.
    return df.drop(columns=[c for c in COLS_SENSIVEIS if c in df.columns])


try:
    df_raw = carregar_dados()
except Exception:  # não expõe traceback/caminhos ao público
    logger.exception("Falha ao carregar o dataset")
    st.error("Não foi possível carregar os dados neste momento. Tente novamente em instantes.")
    st.stop()

vars_quant = [
    c for c in df_raw.select_dtypes(include=[np.number]).columns
    if c not in ("battery_failure", "manufacturing_year")
]

# ───────────────────────── Painel lateral ─────────────────────────
st.sidebar.header("⚙️ Painel de Controlo")
st.sidebar.markdown("Filtros globais para os dois dashboards.")

vars_selecionadas = st.sidebar.multiselect(
    "1. Selecione as variáveis quantitativas:",
    options=vars_quant,
    default=vars_quant[:5],
    help="Alimenta o mapa de calor, a tabela descritiva e os histogramas do Dashboard 1.",
)

st.sidebar.markdown("---")

opcoes_filtro = [c for c in ("odometer_km", "cycle_count", "battery_stress_index", "vehicle_age_years")
                 if c in df_raw.columns]
var_filtro = st.sidebar.selectbox("Variável de referência para o filtro:", options=opcoes_filtro)

serie_filtro = df_raw[var_filtro]
min_val, max_val = float(serie_filtro.min()), float(serie_filtro.max())
intervalo = st.sidebar.slider(
    f"2. Manter apenas valores de {var_filtro} entre:",
    min_value=min_val,
    max_value=max_val,
    value=(min_val, max_val),
    help="Valores fora do intervalo são excluídos da avaliação (útil para tirar extremos).",
)

# Linhas sem medição da variável de filtro (NaN) são mantidas: o slider só exclui valores.
mascara = serie_filtro.between(*intervalo) | serie_filtro.isna()
df_filtrado = df_raw[mascara]

st.sidebar.markdown("---")
st.sidebar.caption(f"📊 **Registos ativos:** {len(df_filtrado):,} de {len(df_raw):,}")

# ───────────────────────── Cabeçalho e abas ─────────────────────────
st.title("🔋 Plataforma de Telemetria e Saúde de Baterias de VEs")
st.markdown("Análise exploratória avançada e monitoramento preditivo para frotas de veículos elétricos.")

tab1, tab2 = st.tabs([
    "🔬 Dashboard 1: Laboratório de Distribuição e Correlação",
    "🚛 Dashboard 2: KPIs & Gestão de Frota (Público-Alvo)",
])

# ───────────────────────── Dashboard 1 ─────────────────────────
with tab1:
    st.subheader("🔬 Laboratório de Análise de Distribuição e Correlação")
    st.markdown(
        "Explore a estrutura multivariada das variáveis operacionais. "
        "Utilize o painel lateral para ajustar o recorte de dados."
    )

    if df_filtrado.empty:
        st.warning("⚠️ O intervalo escolhido não contém registos. Amplie o slider no painel lateral.")
    elif len(vars_selecionadas) < 2:
        st.warning("⚠️ Selecione pelo menos **2 variáveis quantitativas** no painel lateral.")
    else:
        c_cor, c_desc = st.columns([1.1, 1])

        with c_cor:
            st.markdown("#### Mapa de Calor de Correlação")
            metodo = st.radio("Método:", ["pearson", "spearman"], horizontal=True, index=1)
            corr = df_filtrado[vars_selecionadas].corr(method=metodo)
            # Triângulo inferior sem a diagonal: remove a 1.ª linha e a última coluna vazias.
            tri = corr.where(np.tril(np.ones(corr.shape, dtype=bool), k=-1)).iloc[1:, :-1]
            fig_hm = px.imshow(
                tri, text_auto=".2f", zmin=-1, zmax=1,
                color_continuous_scale="RdBu_r", aspect="auto",
                title=f"Correlação ({metodo.capitalize()})",
            )
            fig_hm.update_xaxes(tickangle=-45)
            fig_hm.update_layout(height=470, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_hm, width="stretch")

        with c_desc:
            st.markdown("#### Estatística Descritiva & Assimetria")
            sub = df_filtrado[vars_selecionadas]
            resumo = pd.DataFrame({
                "média": sub.mean(),
                "mediana": sub.median(),
                "desvio-padrão": sub.std(),
                "assimetria": sub.skew(),
                "curtose": sub.kurt(),
            }).round(2)
            st.dataframe(resumo, width="stretch", height=330)
            st.caption("Assimetria ≠ 0 indica cauda longa; por isso Spearman é o método padrão.")

        st.markdown("---")
        st.markdown("#### 📈 Histograma por Variável Quantitativa")
        if len(vars_selecionadas) > MAX_HISTOGRAMAS:
            st.caption(f"A mostrar as primeiras {MAX_HISTOGRAMAS} das {len(vars_selecionadas)} variáveis selecionadas.")
        dados_hist = df_filtrado.dropna(subset=["estado_bateria"])
        colunas = st.columns(2)
        for i, col in enumerate(vars_selecionadas[:MAX_HISTOGRAMAS]):
            fig_h = px.histogram(
                dados_hist, x=col, color="estado_bateria", nbins=40, barmode="overlay",
                opacity=0.7, color_discrete_map=PALETA_ESTADO, category_orders=ORDEM_ESTADO,
                title=col, labels={"estado_bateria": "Estado"},
            )
            fig_h.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Frequência")
            colunas[i % 2].plotly_chart(fig_h, width="stretch")

# ───────────────────────── Dashboard 2 ─────────────────────────
with tab2:
    st.subheader("🚛 Painel Executivo: Gestores de Frota & Engenheiros de BMS")
    st.markdown(
        "**Público-alvo:** equipa de operações de frotas elétricas e engenharia de fiabilidade.  \n"
        "**Foco:** risco de degradação prematura, falhas críticas e planeamento de substituição de packs."
    )
    st.markdown("---")

    if df_filtrado.empty:
        st.warning("⚠️ O intervalo escolhido não contém registos. Amplie o slider no painel lateral.")
    else:
        st.markdown("#### 📌 KPIs Estratégicos")
        d = df_filtrado

        def media(col, escala=1.0):
            return d[col].mean() * escala if col in d.columns and d[col].notna().any() else float("nan")

        def fmt(valor, formato):
            return "—" if pd.isna(valor) else formato.format(valor)

        p_deg = (d.loc[d["estado_bateria"].notna(), "estado_bateria"] == "Degradada").mean() * 100
        p_falha = media("battery_failure", 100)
        stress = media("battery_stress_index")
        vida = d["predicted_remaining_life_cycles"].median() if "predicted_remaining_life_cycles" in d else float("nan")
        p_bms = (d["BMS_warning_count"].dropna() >= 2).mean() * 100 if "BMS_warning_count" in d else float("nan")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Taxa de Degradação", fmt(p_deg, "{:.1f}%"), help="% de baterias com SoH < 80%.")
        k2.metric("Falhas Críticas", fmt(p_falha, "{:.2f}%"), help="% de baterias com falha registada.")
        k3.metric("Estresse Médio", fmt(stress, "{:.1f} pts"), help="Média do Battery Stress Index.")
        k4.metric("Mediana de Vida Restante", fmt(vida, "{:,.0f} ciclos"), help="Ciclos previstos até o fim de vida.")
        k5.metric("Alertas BMS Elevados", fmt(p_bms, "{:.1f}%"), help="Veículos com 2 ou mais avisos no BMS.")

        st.markdown("---")
        st.markdown("#### 📊 Visuais de Suporte Operacional")
        dd = d.dropna(subset=["estado_bateria"])
        v1, v2 = st.columns(2)

        with v1:
            st.markdown("##### 1. Taxa de degradação por fabricante")
            if "battery_manufacturer" in dd.columns:
                tab_fab = (
                    pd.crosstab(dd["battery_manufacturer"], dd["estado_bateria"], normalize="index") * 100
                ).reset_index().melt(id_vars="battery_manufacturer", var_name="Estado", value_name="%")
                fig1 = px.bar(
                    tab_fab, x="battery_manufacturer", y="%", color="Estado",
                    color_discrete_map=PALETA_ESTADO, category_orders={"Estado": ["Saudável", "Degradada"]},
                    labels={"battery_manufacturer": "Fabricante da bateria", "%": "Percentagem (%)"},
                )
                fig1.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig1, width="stretch")

        with v2:
            st.markdown("##### 2. Estresse da bateria por tipo de tração")
            col_drive = "drive_type" if "drive_type" in dd.columns else "vehicle_type"
            if col_drive in dd.columns and "battery_stress_index" in dd.columns:
                fig2 = px.box(
                    dd.dropna(subset=[col_drive]), x=col_drive, y="battery_stress_index", color="estado_bateria",
                    color_discrete_map=PALETA_ESTADO, category_orders=ORDEM_ESTADO,
                    labels={col_drive: "Tipo de tração", "battery_stress_index": "Battery Stress Index",
                            "estado_bateria": "Estado"},
                )
                fig2.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig2, width="stretch")

        st.markdown("##### 3. Quilometragem vs. ciclos de carga")
        if {"odometer_km", "cycle_count"} <= set(dd.columns):
            amostra = dd.dropna(subset=["odometer_km", "cycle_count"])
            amostra = amostra.sample(min(2000, len(amostra)), random_state=42)
            fig3 = px.scatter(
                amostra, x="odometer_km", y="cycle_count", color="estado_bateria", opacity=0.6,
                color_discrete_map=PALETA_ESTADO, category_orders=ORDEM_ESTADO,
                labels={"odometer_km": "Quilometragem (km)", "cycle_count": "Ciclos de carga",
                        "estado_bateria": "Estado"},
            )
            fig3.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig3, width="stretch")
            st.caption("Amostra aleatória de até 2 000 pontos para manter a página fluida.")

st.divider()
st.caption("Dados: EV Battery Health Prediction Dataset (20k) · identificadores de veículo e bateria removidos.")