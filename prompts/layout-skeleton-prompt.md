# Template: rascunho de layout → esqueleto do app

Preencha cada bloco e envie à ferramenta de IA.

## Contexto
Estou desenvolvendo um app Streamlit com dois dashboards a partir da minha EDA
do dataset "EV Battery Health Prediction Dataset (20k)" (Kaggle:
srisyra02/ev-battery-health-prediction-dataset-20k). O app se chama
"Plataforma de Telemetria e Saúde de Baterias de VEs".

O alvo `estado_bateria` é derivado do SoH: "Degradada" quando
`battery_health_percent` < 80% e "Saudável" caso contrário.

O público-alvo do Dashboard 2 são gestores de frota e engenheiros de BMS, que
precisam identificar risco de degradação, falhas críticas e planejar a
substituição de baterias.

## Entrada
Rascunhos de layout anexados:
- entregas/rascunho_dashboard1.png (Laboratório de Distribuição e Correlação)
- entregas/rascunho_dashboard2.png (KPIs & Gestão de Frota)

Blocos descritos nos rascunhos:
- Painel lateral (único, comum aos dois dashboards): seletor de variáveis
  (multisseleção), variável do filtro, slider de intervalo e o texto
  "Registros ativos: N de 20.000".
- Dashboard 1: mapa de calor de correlação (Pearson/Spearman), tabela
  descritiva (média, mediana, desvio, assimetria, curtose) e um histograma
  por variável selecionada.
- Dashboard 2: 5 KPIs (taxa de degradação, falhas críticas, estresse médio,
  vida restante mediana, alertas BMS), visual 1 (degradação por fabricante,
  barras), visual 2 (estresse por tipo de tração, boxplot) e visual 3
  (quilometragem x ciclos de carga, dispersão).

## Restrições
- Bibliotecas: Streamlit, pandas, numpy, plotly e kagglehub. Sem matplotlib.
- Os dois dashboards em abas (`st.tabs`) e um único painel lateral (`st.sidebar`).
- Carregar os dados com kagglehub dentro de `st.cache_data`.
- Remover colunas identificadoras (`vehicle_id`, `battery_serial`) antes de
  qualquer visualização.
- Não exibir traceback nem caminhos internos ao usuário: mensagem de erro genérica.
- Tratar sem quebrar: filtro vazio, menos de 2 variáveis selecionadas e
  colunas ausentes.
- Usar cores consistentes por estado (Saudável e Degradada) em todos os gráficos.
- Entregar somente o esqueleto do layout descrito nos rascunhos, sem análises
  ou gráficos extras.

## Formato de saída
Um único arquivo `app.py` completo, comentado por seção (painel lateral,
Dashboard 1, Dashboard 2) e executável com `streamlit run app.py`. Ao final,
uma lista curta com as colunas do dataset que o código assume existirem.