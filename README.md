# EDA e dashboards Streamlit — saúde de baterias de VEs

**Aplicação online:** *(URL do Streamlit Cloud a preencher após o deploy)*

Dashboard interativo da análise exploratória de dados (EDA) sobre o conjunto [EV Battery Health Prediction Dataset (20k)](https://www.kaggle.com/datasets/srisyra02/ev-battery-health-prediction-dataset-20k).

O notebook completo está em `notebooks/`. Os rascunhos dos dois ecrãs estão em `entregas/`. Os prompts de IA usados para montar a aplicação estão em `prompts/documentacao_ia.md`.

## O que a aplicação mostra

Dois separadores partilham a mesma barra lateral (seletor de variável + slider de intervalo):

1. **Distribuições** — histogramas e boxplots de Saudável vs. Degradada, tabela com média/mediana/IQR e teste de Shapiro-Wilk.
2. **Relações** — dispersão entre duas variáveis, Spearman vs. Pearson, Mann-Whitney U e heatmap fabricante × estado da bateria.

## Como correr localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

O servidor abre em `http://localhost:8501`. O `app.py` descarrega o CSV automaticamente com `kagglehub` (`srisyra02/ev-battery-health-prediction-dataset-20k`).

## Conclusões da EDA

- **Amostra:** cerca de 20 000 veículos/packs, dezenas de colunas de telemetria (capacidade, hodómetro, ciclos, tensão, temperatura, índices de stress, fabricante, etc.). Não há linhas duplicadas; cada linha é uma medição única.
- **Alvo:** `estado_bateria` deriva do limiar automotivo de fim de vida útil. SoH (`state_of_health` / `battery_health_percent`) abaixo de 80% é **Degradada**; caso contrário **Saudável**.
- **Ausentes:** várias colunas de telemetria têm valores em falta no CSV original. A análise **não removeu linhas** para imputar ou descartar nulos: manteve-se N = 20 000 e os gráficos usam os valores disponíveis de cada variável.
- **Outliers:** picos em `temperature_variance` e `battery_stress_index` foram tratados como eventos reais de sobrecarga, não como erros. Não houve exclusão por IQR.
- **Normalidade:** Shapiro-Wilk (subamostra n = 500) rejeita normalidade nos sinais operacionais. Por isso a EDA privilegia mediana, IQR, Spearman e Mann-Whitney.
- **Relações:** a correlação principal reportada é Spearman. Não se eliminaram variáveis por colinearidade nesta fase.
- **Fairness industrial:** a taxa de degradação entre fabricantes/montadoras não mostrou disparidade severa (regra dos 80%) que justificasse rebalancear ou excluir grupos.

## Estrutura do repositório

```
app.py
requirements.txt
README.md
notebooks/          # notebook da EDA
entregas/           # rascunhos dos dois dashboards
prompts/            # documentação dos prompts de IA
```
