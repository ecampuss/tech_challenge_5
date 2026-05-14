import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline


# Classe principal para limpeza e tratamento dos dados

class DataCleaning(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.colunas_remover = [
            'Ano nasc',
            'Data de Nasc',
            'Ano_nasc',
            'INDE 22',
            'INDE 23',
            'INDE 2023',
            'INDE 2024',
            'Pedra 20',
            'Pedra 21',
            'Pedra 22',
            'Pedra 23',
            'Pedra 2023',
            'Pedra 2024',
            'Nome Anonimizado',
            'Nome',
            'Idade 22',
            'Idade',
            'Escola',
            'Defasagem',
            'Fase Ideal',
            'Ativo/ Inativo.1',
            'Avaliador1',
            'Avaliador2',
            'Avaliador3',
            'Avaliador4',
            'Avaliador5',
            'Avaliador6'
        ]

        self.cols_mediana = [
            'IPP',
            'IDA',
            'IPV',
            'INDE',
            'IAA',
            'IEG',
            'IPS',
            'Matem',
            'Portug',
            'Nº Av'
        ]

        self.cols_categoricas = [
            'Pedra',
            'Instituição de ensino'
        ]

        self.mapa_risco = {
            2.5: 0,
            5.0: 1,
            10.0: 2
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        # Ajuste nomes colunas
        df.columns = (
            df.columns
            .str.strip()
            .str.replace('\n', ' ', regex=True)
        )

        # Ajuste tipos
        if 'Fase' in df.columns:
            df['Fase'] = df['Fase'].astype(str)

        if 'Turma' in df.columns:
            df['Turma'] = df['Turma'].astype(str)

        # Criar idade calculada
        if 'Ano' in df.columns and 'Ano_nasc' in df.columns:
            df['idade_calc'] = df['Ano'] - df['Ano_nasc']

        # Quantidade de avaliadores
        avaliadores = [
            'Avaliador1',
            'Avaliador2',
            'Avaliador3',
            'Avaliador4',
            'Avaliador5',
            'Avaliador6'
        ]

        avaliadores_existentes = [
            col for col in avaliadores
            if col in df.columns
        ]

        if avaliadores_existentes:
            df['qtd_avaliadores'] = (
                df[avaliadores_existentes]
                .notnull()
                .sum(axis=1)
            )

        # Remove colunas desnecessárias
        df = df.drop(
            columns=self.colunas_remover,
            errors='ignore'
        )

        # Preencher numéricos com mediana
        for col in self.cols_mediana:

            if col in df.columns:

                mediana = df[col].median()

                df[col] = df[col].fillna(mediana)

        # Preencher categóricas
        for col in self.cols_categoricas:

            if col in df.columns:

                df[col] = df[col].fillna('Desconhecido')

        # Ajuste coluna risco
        if 'IAN' in df.columns:

            df['risco'] = df['IAN'].map(self.mapa_risco)

        # Remover duplicados
        df = df.drop_duplicates()

        return df


# Classe para transformar colunas numéricas em um range de 0 a 1
class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[
        'IPP', 'IDA', 'IPV', 'INDE',
        'IAA', 'IEG', 'IPS', 'Matem',
        'Portug', 'Nº Av', 'qtd_avaliadores',
        'Ano', 'Defas', 'Ano ingresso', 'idade_calc'
    ]):
        self.cols = cols
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        cols_existentes = [c for c in self.cols if c in X.columns]
        self.cols_existentes_ = cols_existentes
        self.scaler.fit(X[self.cols_existentes_])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols_existentes_] = self.scaler.transform(X[self.cols_existentes_])
        return X
      

# Classe para transformar colunas string em um range de 0 a 1
class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[
        'Gênero', 'Fase ideal', 'Fase',
        'Instituição de ensino', 'Turma',
        'Pedra'
    ]):
        self.cols = cols
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        self.cols_existentes_ = [c for c in self.cols if c in X.columns]

        if self.cols_existentes_:
            self.encoder.fit(X[self.cols_existentes_])

        return self

    def transform(self, X):
        X = X.copy()

        if not self.cols_existentes_:
            return X

        encoded = self.encoder.transform(X[self.cols_existentes_])
        feature_names = self.encoder.get_feature_names_out(self.cols_existentes_)

        df_encoded = pd.DataFrame(encoded, columns=feature_names, index=X.index)

        X = X.drop(columns=self.cols_existentes_)
        X = pd.concat([X, df_encoded], axis=1)

        return X
    
      

#Inicialização da Pipeline de Pré-Processamento
preprocess_pipeline = Pipeline([
    ('min_max_scaler', MinMax()),
    ('one_hot_encoding', OneHotEncodingNames())
])