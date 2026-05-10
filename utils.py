import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

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