import glob
import os

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="EDA — Saúde de Baterias de VEs",
    page_icon="🔋",
    layout="wide",
)

COR_NEUTRA = "lightgray"
COR_CONTEXTO = "#38bdf8"
COR_DESTAQUE = "#7c3aed"
COR_ALERTA = "#ef4444"
PALETA_ESTADO = {"Saudável": COR_CONTEXTO, "Degradada": COR_DESTAQUE}

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 4.5)

KAGGLE_DATASET = "srisyra02/ev-battery-health-prediction-dataset-20k"
VARS_QUANT_PADRAO = [
    "battery_capacity_kwh",
    "odometer_km",
    "cycle_count",
    "cell_voltage_avg",
    "temperature_variance",
    "battery_stress_index",
    "predicted_remaining_life_cycles",
    "battery_health_percent",
]


@st.cache_data(show_spinner="A descarregar o dataset do Kaggle...")
def carregar_dados() -> pd.DataFrame:
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {path}")

    df = pd.read_csv(csv_files[0])

    soh_col = None
    for candidato in ("state_of_health", "battery_health_percent"):
        if candidato in df.columns:
            soh_col = candidato
            break

    if soh_col is not None:
        df["estado_bateria"] = np.where(
            df[soh_col] < 80.0, "Degradada", "Saudável"
        )
        df.loc[df[soh_col].isna(), "estado_bateria"] = np.nan
    elif "battery_failure" in df.columns:
        df["estado_bateria"] = np.where(
            df["battery_failure"] == 1, "Degradada", "Saudável"
        )
    else:
        df["estado_bateria"] = "Saudável"

    return df


def variaveis_numericas(df: pd.DataFrame) -> list[str]:
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    prioridade = [c for c in VARS_QUANT_PADRAO if c in numericas]
    restantes = [c for c in numericas if c not in prioridade]
    return prioridade + restantes


def aplicar_filtro(df: pd.DataFrame, coluna: str, intervalo: tuple[float, float]) -> pd.DataFrame:
    minimo, maximo = intervalo
    return df[df[coluna].between(minimo, maximo)]