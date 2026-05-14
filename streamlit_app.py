import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from utils import DataCleaning, MinMax, OneHotEncodingNames


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


# VISÃO GERAL

st.header('Visão Geral da Base')

col1, col2, col3, col4 = st.columns(4)

col1.metric('Quantidade de alunos', len(df))
col2.metric('Média INDE', round(df['INDE'].mean(), 2))
col3.metric('Média IDA', round(df['IDA'].mean(), 2))
col4.metric('Média IEG', round(df['IEG'].mean(), 2))



# 1. ADEQUAÇÃO DO NÍVEL (IAN)

st.header('1. Adequação do nível (IAN)')

st.markdown('''
Qual é o perfil geral de defasagem dos alunos (IAN) e como ele evolui ao 
longo do ano?
''')

fig = px.histogram(
    df,
    x='IAN',
    title='Distribuição do IAN'
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('''
### Como interpretar este gráfico

Cada barra representa a quantidade de alunos em cada nível de IAN.
Quanto maior a barra, maior é a concentração de alunos naquele nível de adequação.
Valores menores de IAN indicam maior risco de defasagem, enquanto valores mais altos
representam alunos com melhor adequação acadêmica.
Esse gráfico ajuda a entender o perfil geral da base e identificar se existe uma concentração
significativa de alunos em situação de risco.
''')



# 2. DESEMPENHO ACADÊMICO (IDA)

st.header('2. Desempenho acadêmico (IDA)')

st.markdown('''
O desempenho acadêmico médio (IDA) está melhorando, estagnado ou caindo ao longo
das fases e anos?
''')

ida_ano = df.groupby('Ano')['IDA'].mean().reset_index()

fig = px.line(
    ida_ano,
    x='Ano',
    y='IDA',
    markers=True,
    title='Evolução do IDA ao longo dos anos'
)

fig.update_xaxes(
    tickmode='linear',
    dtick=1
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('''
### Como interpretar este gráfico

A linha mostra a média do desempenho acadêmico (IDA) ao longo dos anos.
Quando a linha sobe, significa que o desempenho médio dos alunos está melhorando.
Quando a linha permanece estável, indica manutenção do desempenho.
Já quedas na linha podem indicar perda de desempenho acadêmico ao longo do tempo.
Esse gráfico ajuda a identificar tendências de evolução educacional dos alunos
participantes do programa.
''')



# 3. ENGAJAMENTO (IEG)

st.header('3. Relação entre IEG, IDA e IPV')

st.markdown('''
O grau de engajamento dos alunos (IEG) tem relação direta com seus indicadores 
de desempenho (IDA) e do ponto de virada (IPV)?
''')

fig = px.scatter(
    df,
    x='IEG',
    y='IDA',
    color='IPV',
    title='Relação entre Engajamento e Desempenho'
)

st.plotly_chart(fig, use_container_width=True)



# 4. AUTOAVALIAÇÃO (IAA)

st.header('4. Autoavaliação dos alunos (IAA)')

st.markdown('''
As percepções dos alunos sobre si mesmos (IAA) são coerentes com seu desempenho
real (IDA) e engajamento (IEG)?
''')

fig = px.scatter(
    df,
    x='IAA',
    y='IDA',
    color='IEG',
    title='IAA vs IDA'
)

st.plotly_chart(fig, use_container_width=True)



# 5. ASPECTOS PSICOSSOCIAIS (IPS)

st.header('5. Aspectos psicossociais (IPS)')

st.markdown('''
Há padrões psicossociais (IPS) que antecedem quedas de desempenho acadêmico
ou de engajamento?
''')

fig = px.box(
    df,
    x='IAN',
    y='IPS',
    title='IPS por categoria de risco'
)

st.plotly_chart(fig, use_container_width=True)



# 6. ASPECTOS PSICOPEDAGÓGICOS (IPP)

st.header('6. Aspectos psicopedagógicos (IPP)')

st.markdown('''
As avaliações psicopedagógicas (IPP) confirmam ou contradizem a defasagem
identificada pelo IAN?
''')

fig = px.scatter(
    df,
    x='IPP',
    y='IAN',
    color='IAN',
    title='IPP vs IAN'
)

st.plotly_chart(fig, use_container_width=True)



# 7. PONTO DE VIRADA (IPV)

st.header('7. Ponto de virada (IPV)')

st.markdown('''
Quais comportamentos - acadêmicos, emocionais ou de engajamento - mais
influenciam o IPV ao longo do tempo?
''')

corr_ipv = df[[
    'IPV',
    'IDA',
    'IEG',
    'IPS',
    'IPP',
    'IAA'
]].corr()['IPV'].sort_values(ascending=False)

st.bar_chart(corr_ipv)



# 8. MULTIDIMENSIONALIDADE

st.header('8. Multidimensionalidade dos indicadores')

st.markdown('''
Quais combinações de indicadores (IDA + IEG + IPS + IPP) elevam mais a 
nota global do aluno (INDE)?
''')

corr_inde = df[[
    'INDE',
    'IDA',
    'IEG',
    'IPS',
    'IPP'
]].corr()['INDE'].sort_values(ascending=False)

st.bar_chart(corr_inde)



# 9. MACHINE LEARNING

st.header('9. Previsão de risco com Machine Learning')

st.markdown('''
Quais padrões nos indicadores permitem identificar alunos em risco antes de queda
no desempenho ou aumento da defasagem? Construa um modelo preditivo que mostre uma
probabilidade do aluno ou aluna entrar em risco de defasagem.
''')

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

# FEATURE IMPORTANCE

st.subheader('Variáveis mais importantes')

importancias = pd.Series(
    modelo.feature_importances_,
    index=X_train_prep.columns
).sort_values(ascending=False).head(15)

st.bar_chart(importancias)



# 10. EFETIVIDADE DO PROGRAMA

st.header('10. Efetividade do programa')

st.markdown('''
Os indicadores mostram melhora consistente ao longo do ciclo nas diferentes
fases (Quartzo, Ágata, Ametista e Topázio), confirmando o impacto real
do programa?
''')

pedra_inde = df.groupby('Pedra')['INDE'].mean().reset_index()

fig = px.bar(
    pedra_inde,
    x='Pedra',
    y='INDE',
    title='INDE médio por Pedra'
)

st.plotly_chart(fig, use_container_width=True)


st.success('Análise finalizada com sucesso.')
