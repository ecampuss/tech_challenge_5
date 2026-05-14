import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

from utils import (
    DataCleaning,
    MinMax,
    OneHotEncodingNames
)


st.set_page_config(
    page_title='Tech Challenge 5',
    layout='wide'
)


st.title('Tech Challenge 5 - Passos Mágicos')
st.subheader('Modelo Preditivo de Risco Educacional')


@st.cache_data
def load_data():

    file_path = 'base_de_dados_passos_magicos.xlsx'

    df_2022 = pd.read_excel(file_path, sheet_name='PEDE2022')
    df_2023 = pd.read_excel(file_path, sheet_name='PEDE2023')
    df_2024 = pd.read_excel(file_path, sheet_name='PEDE2024')

    limpeza = DataCleaning()

    df = limpeza.transform({
        '2022': df_2022,
        '2023': df_2023,
        '2024': df_2024
    })

    return df


df = load_data()


# =========================================================
# VISÃO GERAL
# =========================================================

st.header('Visão Geral da Base')

col1, col2, col3, col4 = st.columns(4)

col1.metric('Quantidade de alunos', len(df))
col2.metric('Média INDE', round(df['INDE'].mean(), 2))
col3.metric('Média IDA', round(df['IDA'].mean(), 2))
col4.metric('Média IEG', round(df['IEG'].mean(), 2))


# =========================================================
# 1. ADEQUAÇÃO DO NÍVEL (IAN)
# =========================================================

st.header('1. Adequação do nível (IAN)')

fig = px.histogram(
    df,
    x='IAN',
    title='Distribuição do IAN'
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('''
### Insight

O gráfico mostra a distribuição do nível de adequação dos alunos.
É possível identificar quantos alunos apresentam maior risco de defasagem.
''')


# =========================================================
# 2. DESEMPENHO ACADÊMICO (IDA)
# =========================================================

st.header('2. Desempenho acadêmico (IDA)')

ida_ano = df.groupby('Ano')['IDA'].mean().reset_index()

fig = px.line(
    ida_ano,
    x='Ano',
    y='IDA',
    markers=True,
    title='Evolução do IDA ao longo dos anos'
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 3. ENGAJAMENTO (IEG)
# =========================================================

st.header('3. Relação entre IEG, IDA e IPV')

fig = px.scatter(
    df,
    x='IEG',
    y='IDA',
    color='IPV',
    title='Relação entre Engajamento e Desempenho'
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 4. AUTOAVALIAÇÃO (IAA)
# =========================================================

st.header('4. Autoavaliação dos alunos (IAA)')

fig = px.scatter(
    df,
    x='IAA',
    y='IDA',
    color='IEG',
    title='IAA vs IDA'
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 5. ASPECTOS PSICOSSOCIAIS (IPS)
# =========================================================

st.header('5. Aspectos psicossociais (IPS)')

fig = px.box(
    df,
    x='IAN',
    y='IPS',
    title='IPS por categoria de risco'
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 6. ASPECTOS PSICOPEDAGÓGICOS (IPP)
# =========================================================

st.header('6. Aspectos psicopedagógicos (IPP)')

fig = px.scatter(
    df,
    x='IPP',
    y='IAN',
    color='IAN',
    title='IPP vs IAN'
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 7. PONTO DE VIRADA (IPV)
# =========================================================

st.header('7. Ponto de virada (IPV)')

corr_ipv = df[[
    'IPV',
    'IDA',
    'IEG',
    'IPS',
    'IPP',
    'IAA'
]].corr()['IPV'].sort_values(ascending=False)

st.bar_chart(corr_ipv)


# =========================================================
# 8. MULTIDIMENSIONALIDADE
# =========================================================

st.header('8. Multidimensionalidade dos indicadores')

corr_inde = df[[
    'INDE',
    'IDA',
    'IEG',
    'IPS',
    'IPP'
]].corr()['INDE'].sort_values(ascending=False)

st.bar_chart(corr_inde)


# =========================================================
# 9. MACHINE LEARNING
# =========================================================

st.header('9. Previsão de risco com Machine Learning')


features = [
    'IEG',
    'INDE',
    'IDA',
    'IPV',
    'IAA',
    'IPS',
    'IPP',
    'Matem',
    'Portug',
    'Pedra',
    'Fase',
    'Fase ideal',
    'Instituição de ensino',
    'Turma',
    'Ano',
    'idade_calc'
]


X = df[features]
y = df['risco']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


preprocess_pipeline = Pipeline([
    ('min_max_scaler', MinMax()),
    ('one_hot_encoding', OneHotEncodingNames())
])


X_train_prep = preprocess_pipeline.fit_transform(X_train)
X_test_prep = preprocess_pipeline.transform(X_test)


modelo = RandomForestClassifier(
    random_state=42,
    n_estimators=200
)

modelo.fit(X_train_prep, y_train)


y_pred = modelo.predict(X_test_prep)


accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(
    y_test,
    modelo.predict_proba(X_test_prep),
    multi_class='ovr'
)


col1, col2 = st.columns(2)

col1.metric('Accuracy', round(accuracy, 4))
col2.metric('AUC', round(auc, 4))


st.subheader('Classification Report')

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)


# MATRIZ CONFUSÃO

cm = confusion_matrix(y_test, y_pred)

fig = px.imshow(
    cm,
    text_auto=True,
    title='Matriz de Confusão'
)

st.plotly_chart(fig, use_container_width=True)


# FEATURE IMPORTANCE

st.subheader('Variáveis mais importantes')

importancias = pd.Series(
    modelo.feature_importances_,
    index=X_train_prep.columns
).sort_values(ascending=False).head(15)

st.bar_chart(importancias)


# =========================================================
# 10. EFETIVIDADE DO PROGRAMA
# =========================================================

st.header('10. Efetividade do programa')

pedra_inde = df.groupby('Pedra')['INDE'].mean().reset_index()

fig = px.bar(
    pedra_inde,
    x='Pedra',
    y='INDE',
    title='INDE médio por Pedra'
)

st.plotly_chart(fig, use_container_width=True)


st.success('Análise finalizada com sucesso.')
