import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score
)

from utils import preprocess_pipeline

# Configuração inicial da página
st.set_page_config(
    page_title='Tech Challenge 5',
    layout='wide'
)

st.title('Tech Challenge 5 - Passos Mágicos')
st.subheader('Modelo Preditivo de Risco Educacional')

# Link do arquivo hospedado no GitHub
URL = 'https://raw.githubusercontent.com/ecampuss/tech_challenge_5/main/base_final.xlsx'

# Cache para evitar recarregar o arquivo toda vez
@st.cache_data
def load_data():

    df = pd.read_excel(URL)

    # Garantindo que algumas colunas estejam como texto
    df['Fase'] = df['Fase'].astype(str)
    df['Turma'] = df['Turma'].astype(str)

    return df

df = load_data()

# Criação da variável target
map_risco = {
    2.5: 0,
    5.0: 1,
    10.0: 2
}

df['risco'] = df['IAN'].map(map_risco)

# Menu lateral da aplicação
menu = st.sidebar.selectbox(
    'Menu',
    [
        'Visão Geral',
        'Quick Insights',
        'Modelo Preditivo',
        'Predição Individual'
    ]
)

# =========================================================
# VISÃO GERAL
# =========================================================

if menu == 'Visão Geral':

    st.header('Visão Geral da Base')

    col1, col2, col3, col4 = st.columns(4)

    col1.metric('Quantidade de alunos', df.shape[0])
    col2.metric('Média INDE', round(df['INDE'].mean(), 2))
    col3.metric('Média IDA', round(df['IDA'].mean(), 2))
    col4.metric('Média IEG', round(df['IEG'].mean(), 2))

    st.subheader('Prévia dos dados')

    st.dataframe(df.head())

# =========================================================
# QUICK INSIGHTS
# =========================================================

elif menu == 'Quick Insights':

    st.header('Quick Insights')

    # Distribuição do IAN
    st.subheader('1. Perfil geral de defasagem (IAN)')

    fig, ax = plt.subplots(figsize=(7,4))

    df['IAN'].value_counts().sort_index().plot(
        kind='bar',
        ax=ax
    )

    ax.set_title('Distribuição do IAN')

    st.pyplot(fig)

    # Evolução do desempenho acadêmico
    st.subheader('2. Evolução do desempenho acadêmico (IDA)')

    fig, ax = plt.subplots(figsize=(8,4))

    df.groupby('Ano')['IDA'].mean().plot(
        marker='o',
        ax=ax
    )

    ax.set_title('Média do IDA por ano')

    st.pyplot(fig)

    # Relação entre engajamento e desempenho
    st.subheader('3. Relação entre IEG e IDA')

    fig, ax = plt.subplots(figsize=(6,4))

    ax.scatter(df['IEG'], df['IDA'])

    ax.set_xlabel('IEG')
    ax.set_ylabel('IDA')

    st.pyplot(fig)

    # Relação entre autoavaliação e desempenho
    st.subheader('4. Relação entre IAA e IDA')

    fig, ax = plt.subplots(figsize=(6,4))

    ax.scatter(df['IAA'], df['IDA'])

    ax.set_xlabel('IAA')
    ax.set_ylabel('IDA')

    st.pyplot(fig)

    # Correlação envolvendo IPS
    st.subheader('5. Correlação dos aspectos psicossociais (IPS)')

    corr_ips = df[['IPS', 'IDA', 'IEG', 'IPV']].corr()

    fig, ax = plt.subplots(figsize=(6,4))

    im = ax.imshow(corr_ips)

    ax.set_xticks(range(len(corr_ips.columns)))
    ax.set_yticks(range(len(corr_ips.columns)))

    ax.set_xticklabels(corr_ips.columns)
    ax.set_yticklabels(corr_ips.columns)

    plt.colorbar(im)

    st.pyplot(fig)

    # Relação entre IPP e IAN
    st.subheader('6. Relação entre IPP e IAN')

    fig, ax = plt.subplots(figsize=(7,4))

    df.boxplot(
        column='IPP',
        by='IAN',
        ax=ax
    )

    st.pyplot(fig)

    # Correlação com IPV
    st.subheader('7. Variáveis mais relacionadas ao IPV')

    corr_ipv = (
        df.select_dtypes(include='number')
        .corr()['IPV']
        .sort_values(ascending=False)
    )

    st.dataframe(corr_ipv)

    # Correlação multidimensional
    st.subheader('8. Correlação entre os principais indicadores')

    corr_multi = df[
        ['IDA', 'IEG', 'IPS', 'IPP', 'INDE']
    ].corr()

    fig, ax = plt.subplots(figsize=(6,4))

    im = ax.imshow(corr_multi)

    ax.set_xticks(range(len(corr_multi.columns)))
    ax.set_yticks(range(len(corr_multi.columns)))

    ax.set_xticklabels(corr_multi.columns)
    ax.set_yticklabels(corr_multi.columns)

    plt.colorbar(im)

    st.pyplot(fig)

    # Efetividade do programa
    st.subheader('10. Efetividade do programa por pedra')

    fig, ax = plt.subplots(figsize=(8,4))

    df.groupby('Pedra')['INDE'].mean().plot(
        kind='bar',
        ax=ax
    )

    ax.set_title('Média do INDE por Pedra')

    st.pyplot(fig)

# =========================================================
# MODELO PREDITIVO
# =========================================================

elif menu == 'Modelo Preditivo':

    st.header('Modelo Preditivo')

    # Separação das variáveis
    X = df.drop(columns=['IAN', 'risco'])
    y = df['risco']

    # Divisão treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Aplicando pipeline
    X_train_prep = preprocess_pipeline.fit_transform(X_train)
    X_test_prep = preprocess_pipeline.transform(X_test)

    # Criação do modelo
    modelo = RandomForestClassifier(
        random_state=42
    )

    modelo.fit(X_train_prep, y_train)

    # Predições
    y_pred = modelo.predict(X_test_prep)

    # Classification report
    st.subheader('Classification Report')

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    # AUC multiclasse
    auc = roc_auc_score(
        y_test,
        modelo.predict_proba(X_test_prep),
        multi_class='ovr'
    )

    st.metric(
        'AUC Multiclasse',
        round(auc, 4)
    )

    # Matriz de confusão
    st.subheader('Matriz de Confusão')

    fig, ax = plt.subplots(figsize=(7,7))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        normalize='true',
        cmap='Blues',
        ax=ax
    )

    st.pyplot(fig)

    # Feature importance
    st.subheader('Top 15 variáveis mais importantes')

    importancias = pd.Series(
        modelo.feature_importances_,
        index=X_train_prep.columns
    ).sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(10,6))

    importancias.head(15).sort_values().plot(
        kind='barh',
        ax=ax2
    )

    st.pyplot(fig2)

# =========================================================
# PREDIÇÃO INDIVIDUAL
# =========================================================

elif menu == 'Predição Individual':

    st.header('Predição Individual')

    st.write('Preencha os dados do aluno para simular o risco.')

    idade = st.slider('Idade', 10, 25, 17)

    matematica = st.slider('Nota de Matemática', 0.0, 10.0, 5.0)

    portugues = st.slider('Nota de Português', 0.0, 10.0, 5.0)

    ieg = st.slider('IEG', 0.0, 10.0, 5.0)

    ida = st.slider('IDA', 0.0, 10.0, 5.0)

    if st.button('Prever risco'):

        st.success(
            'Estrutura pronta para expansão com previsão individual utilizando o pipeline completo.'
        )

