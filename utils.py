import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline


import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


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
            'Avaliador6',
            'Ct',
            'Cf',
            'Cg',
            'Destaque IEG',
            'Destaque IDA',
            'Destaque IPV',
            'Rec Av1',
            'Rec Av2',
            'Rec Av3',
            'Rec Av4',
            'Rec Psicologia',
            'Indicado',
            'Atingiu PV',
            'Inglês',
            'Ativo/ Inativo',
            'RA',
            'Defas'
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

    def fit(self, X, y=None):
        return self

    def extrair_ano(self, valor):

        if pd.isna(valor):
            return np.nan

        valor_str = str(valor).strip()

        if valor_str.isdigit() and len(valor_str) == 4:
            return int(valor_str)

        try:
            return pd.to_datetime(valor, errors='coerce').year

        except:
            return np.nan

    def padronizar_colunas(self, df):

        return df.rename(columns={
            'Nome Anonimizado': 'Nome',
            'Mat': 'Matem',
            'Por': 'Portug',
            'Ing': 'Inglês',
            'Fase Ideal': 'Fase ideal',
            'Defasagem': 'Defas'
        })

    def remover_colunas(self, df):

        return df.drop(
            columns=[
                c for c in self.colunas_remover
                if c in df.columns
            ],
            errors='ignore'
        )

    def alinhar_colunas(self, df, colunas):

        for col in colunas:

            if col not in df.columns:
                df[col] = None

        return df[colunas]

    def transform(self, dataframes):

        df_2022 = dataframes['2022']
        df_2023 = dataframes['2023']
        df_2024 = dataframes['2024']

        # adicionar ano
        df_2022['Ano'] = 2022
        df_2023['Ano'] = 2023
        df_2024['Ano'] = 2024

        # padronizar nomes
        df_2022 = self.padronizar_colunas(df_2022)
        df_2023 = self.padronizar_colunas(df_2023)
        df_2024 = self.padronizar_colunas(df_2024)

        # criar INDE
        df_2022['INDE'] = df_2022['INDE 22']
        df_2023['INDE'] = df_2023['INDE 2023']
        df_2024['INDE'] = df_2024['INDE 2024']

        # criar Pedra
        df_2022['Pedra'] = df_2022.get('Pedra 22')
        df_2023['Pedra'] = df_2023.get('Pedra 23')
        df_2024['Pedra'] = df_2024.get('Pedra 2024')

        # ano nascimento
        df_2022['Ano_nasc'] = df_2022['Ano nasc'].apply(self.extrair_ano)
        df_2023['Ano_nasc'] = df_2023['Data de Nasc'].apply(self.extrair_ano)
        df_2024['Ano_nasc'] = df_2024['Data de Nasc'].apply(self.extrair_ano)

        # idade
        df_2022['idade_calc'] = df_2022['Ano'] - df_2022['Ano_nasc']
        df_2023['idade_calc'] = df_2023['Ano'] - df_2023['Ano_nasc']
        df_2024['idade_calc'] = df_2024['Ano'] - df_2024['Ano_nasc']

        # remover colunas
        df_2022 = self.remover_colunas(df_2022)
        df_2023 = self.remover_colunas(df_2023)
        df_2024 = self.remover_colunas(df_2024)

        # alinhar colunas
        todas_colunas = list(
            set(df_2022.columns)
            | set(df_2023.columns)
            | set(df_2024.columns)
        )

        df_2022 = self.alinhar_colunas(df_2022, todas_colunas)
        df_2023 = self.alinhar_colunas(df_2023, todas_colunas)
        df_2024 = self.alinhar_colunas(df_2024, todas_colunas)

        # concatenar
        df = pd.concat(
            [df_2022, df_2023, df_2024],
            ignore_index=True
        )

        # remover colunas .1
        df = df.loc[:, ~df.columns.str.contains(r'\.1')]

        # converter numéricos
        cols_numericas = [
            'IDA',
            'IEG',
            'IAA',
            'IPS',
            'IPP',
            'IPV',
            'INDE'
        ]

        for col in cols_numericas:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors='coerce'
                )

        # preencher medianas
        for col in self.cols_mediana:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors='coerce'
                )

                df[col] = df[col].fillna(
                    df[col].median()
                )

        # preencher categóricas
        for col in self.cols_categoricas:

            if col in df.columns:

                df[col] = (
                    df[col]
                    .fillna('Desconhecido')
                    .astype(str)
                )

        # ajustes finais
        if 'Turma' in df.columns:
            df['Turma'] = df['Turma'].astype(str)

        if 'Fase' in df.columns:
            df['Fase'] = df['Fase'].astype(str)

        # criar target
        df['risco'] = df['IAN'].map({
            2.5: 0,
            5.0: 1,
            10.0: 2
        })

        # remover duplicados
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