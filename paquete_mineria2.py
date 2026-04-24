"""
Paquete mineria avanzada- Jorge Chacon, Stacy Quesada
Clases: EDA, Supervisado (hereda EDA), NoSupervisado (hereda EDA), WebScraping
"""
from abc import ABCMeta, abstractmethod
import statistics
from numpy import corrcoef
from scipy.stats import boxcox
from scipy import signal
from scipy.cluster.hierarchy import dendrogram, ward, single, complete, average, fcluster
from math import ceil, pi
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import inspect
import pandas as pd
import numpy as np
import math
from functools import wraps
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     TimeSeriesSplit, cross_val_score, cross_validate)
from sklearn.metrics import (confusion_matrix, accuracy_score, get_scorer,
                             precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.decomposition import PCA
import matplotlib
try:
    get_ipython()  # type: ignore[name-defined]  # solo existe en Jupyter/IPython
except NameError:
    matplotlib.use('Agg')  # backend no-interactivo para scripts y Streamlit

# Imports opcionales
try:
    import umap as um
except ImportError:
    um = None
try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
try:
    from statsmodels.tsa.api import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
try:
    from sklearn.preprocessing import MinMaxScaler
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
try:
    from sklearn_extra.cluster import KMedoids
except (ImportError, ValueError):
    KMedoids = KMeans
try:
    from prince import PCA as PCA_Prince
except ImportError:
    PCA_Prince = None
try:
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Integer, Continuous, Categorical
    GENETIC_AVAILABLE = True
except ImportError:
    GENETIC_AVAILABLE = False
try:
    from selenium import webdriver as _selenium_webdriver
    from selenium.webdriver.common.by import By as _By
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
try:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    MLP_SKLEARN_AVAILABLE = True
except ImportError:
    MLP_SKLEARN_AVAILABLE = False
try:
    import importlib.util as _iutil
    KERAS_NN_AVAILABLE = _iutil.find_spec('keras') is not None
except Exception:
    KERAS_NN_AVAILABLE = False
# Keras se importa de forma lazy dentro de cada método que lo necesita
_KerasSequential = _Dense = _Conv1D = _MaxPooling1D = _Flatten = _KerasLSTM = None

pd.options.display.max_rows = 10
warnings.filterwarnings('ignore')

# ============================================================================
# UTILIDADES
# ============================================================================


class ErrorHandler:
    @staticmethod
    def handle_error(msg="Error", raise_exception=False):
        print(f"ERROR: {msg}")
        if raise_exception:
            raise Exception(msg)

    @staticmethod
    def validate_dataframe(df, min_rows=1, min_cols=1):
        if not isinstance(df, pd.DataFrame) or df.empty or df.shape[0] < min_rows or df.shape[1] < min_cols:
            ErrorHandler.handle_error(
                "DataFrame inválido", raise_exception=True)
        return True


def error_handler_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\nError en {func.__name__}: {str(e)}")
            return None
    return wrapper


def _tiene_parametro(clase, nombre_param):
    """Verifica de forma segura si un modelo sklearn acepta un parámetro dado.

    FIX: reemplaza el uso frágil de __code__.co_varnames que fallaba con
    clases que usan herencia o *args/**kwargs.
    """
    try:
        sig = inspect.signature(clase.__init__)
        return nombre_param in sig.parameters
    except (ValueError, TypeError):
        return False


class Utilidades:
    @staticmethod
    def cargar_datos(path, sep=",", decimal=".", index_col=None):
        try:
            df = pd.read_csv(path, sep=sep, decimal=decimal,
                             index_col=index_col)
            print(f"Datos cargados: {df.shape}")
            return df
        except Exception as e:
            ErrorHandler.handle_error(
                f"Error al cargar: {str(e)}", raise_exception=True)

    @staticmethod
    def centroide(num_cluster, datos, clusters):
        return pd.DataFrame(datos[clusters == num_cluster].mean()).T

# ============================================================================
# CLASE EDA
# ============================================================================


class EDA:
    def __init__(self, path=None, df=None, sep=",", decimal=".", index_col=None):
        if path:
            self.df = Utilidades.cargar_datos(path, sep, decimal, index_col)
        elif df is not None:
            ErrorHandler.validate_dataframe(df)
            self.df = df.copy()
        else:
            ErrorHandler.handle_error(
                "Debe proporcionar 'path' o 'df'", raise_exception=True)
        self.df_original = self.df.copy()

    def analisis_numerico(self):
        self.df = self.df.select_dtypes(include=["number"])
        print(f"Análisis numérico: {self.df.shape[1]} columnas")
        return self

    def analisis_completo(self):
        self.df = pd.get_dummies(self.df)
        print(f"Análisis completo: {self.df.shape[1]} columnas")
        return self

    def resumen_estadistico(self):
        print("\n" + "="*70)
        print("RESUMEN ESTADISTICO")
        print("="*70)
        print(f"\nDimensiones: {self.df.shape}")
        print(f"\nPrimeras filas:\n{self.df.head()}")
        print(f"\nDescripción:\n{self.df.describe()}")
        print(f"\nNulos:\n{self.df.isnull().sum()}")
        return self

    def _plot(self, kind, title, figsize=(12, 8), show=True, **kwargs):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        if kind == 'box':
            self.df.boxplot(ax=ax)
        elif kind in ['density', 'hist']:
            self.df.plot(kind=kind, ax=ax, **kwargs)
        plt.title(title)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def grafico_boxplot(self, figsize=(15, 8), show=True):
        return self._plot('box', 'Boxplot - Outliers', figsize, show=show)

    def grafico_densidad(self, figsize=(12, 8), show=True):
        return self._plot('density', 'Función de Densidad', figsize, show=show)

    def grafico_histograma(self, figsize=(10, 6), show=True):
        return self._plot('hist', 'Histograma', figsize, show=show, alpha=0.7)

    def matriz_correlacion(self, figsize=(12, 8), mostrar_valores=True, show=True):
        corr = self.df.corr(numeric_only=True)
        print(f"\nMatriz de Correlación:\n{corr}")
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=sns.diverging_palette(220, 10, as_cmap=True).reversed(),
                    square=True, annot=mostrar_valores, fmt='.2f', ax=ax)
        plt.title("Mapa de Calor - Correlaciones")
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def analisis_completo_visual(self, show=True):
        print("\nEjecutando análisis visual completo...")
        figs = {
            'boxplot':     self.grafico_boxplot(show=show),
            'densidad':    self.grafico_densidad(show=show),
            'histograma':  self.grafico_histograma(show=show),
            'correlacion': self.matriz_correlacion(show=show),
        }
        print("Análisis completado")
        return figs

    def reset(self):
        self.df = self.df_original.copy()
        print("DataFrame restaurado")
        return self

# ============================================================================
# CLASE SUPERVISADO
# ============================================================================


class Supervisado(EDA):
    def __init__(self, df, target_col='target'):
        super().__init__(df=df)
        self.target_col = target_col
        self.X_train = self.X_test = self.y_train = self.y_test = self.y = None
        if target_col not in df.columns:
            ErrorHandler.handle_error(
                f"Columna '{target_col}' no encontrada", raise_exception=True)
        print(
            f"Supervisado inicializado - Target: {target_col}, Shape: {self.df.shape}")

    def _asegurar_df_encoded(self):
        if not hasattr(self, 'df_encoded'):
            df_encoded = self.df.copy()
            # Convertir TODAS las columnas a numéricas - GARANTIZADO
            for col in df_encoded.columns:
                col_str = df_encoded[col].astype(str)

                # Intenta conversión numérica (reemplaza comas)
                col_numeric = pd.to_numeric(
                    col_str.str.replace(',', '.', regex=False),
                    errors='coerce'
                )

                # Calcula tasa de éxito
                success_rate = col_numeric.notna().sum(
                ) / len(col_numeric) if len(col_numeric) > 0 else 0

                if success_rate > 0.5:
                    # Mayoría se convirtió: usar versión numérica
                    df_encoded[col] = col_numeric
                    if df_encoded[col].isna().any():
                        med = df_encoded[col].median()
                        if pd.notna(med):
                            df_encoded[col].fillna(med, inplace=True)
                        else:
                            df_encoded[col].fillna(0, inplace=True)
                else:
                    # No se pudo convertir: usar códigos categóricos
                    df_encoded[col] = pd.Categorical(df_encoded[col]).codes
            self.df_encoded = df_encoded

    def preparar_datos(self, test_size=0.25, random_state=42):
        """Prepara datos normalizando decimales con coma y convirtiendo tipos.
        GARANTIZA que todas las columnas sean numéricas al final."""
        df_prep = self.df.copy()

        # Convertir TODAS las columnas a numéricas - GARANTIZADO
        for col in df_prep.columns:
            col_str = df_prep[col].astype(str)

            # Intenta primero conversión numérica (reemplaza comas)
            col_numeric = pd.to_numeric(
                col_str.str.replace(',', '.', regex=False),
                errors='coerce'
            )

            # Calcula tasa de éxito
            success_rate = col_numeric.notna().sum(
            ) / len(col_numeric) if len(col_numeric) > 0 else 0

            if success_rate > 0.5:
                # Mayoría de valores se convirtieron: usar la versión numérica
                df_prep[col] = col_numeric
                # Rellenar NaN restantes con la mediana
                if df_prep[col].isna().any():
                    med = df_prep[col].median()
                    if pd.notna(med):
                        df_prep[col].fillna(med, inplace=True)
                    else:
                        # Si mediana es NaN (todos NaN), llenar con 0
                        df_prep[col].fillna(0, inplace=True)
            else:
                # No se pudo convertir a numérica: convertir a códigos categóricos
                # Esto mapea cada valor único a un entero (0, 1, 2, ...)
                df_prep[col] = pd.Categorical(df_prep[col]).codes

        self.df_encoded = df_prep

        X = df_prep.drop(columns=[self.target_col])
        y = df_prep[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        print(
            f"Datos preparados - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self

    @staticmethod
    def _calcular_metricas_clasificacion(y_test, y_pred, y_unique):
        MC = confusion_matrix(y_test, y_pred)
        precision = np.sum(MC.diagonal()) / np.sum(MC)
        return {
            "Matriz de Confusión": MC,
            "Precisión Global": precision,
            "Error Global": 1 - precision,
            "Precisión por categoría": pd.DataFrame(MC.diagonal()/np.sum(MC, axis=1)).T
        }

    def _balance_data(self, X, y, method=None, random_state=42):
        """Balancea clases en el conjunto de entrenamiento.

        Métodos soportados:
          - None / 'none': no cambia
          - 'oversample': sobre-muestreo aleatorio de la clase minoritaria
          - 'undersample': sub-muestreo aleatorio de la clase mayoritaria
          - 'smote': generación de muestras sintéticas (solo features numéricas)
        """
        if method is None or method == 'none':
            return X, y

        X = pd.DataFrame(X).copy()
        y = pd.Series(y).copy()

        counts = y.value_counts()
        if len(counts) <= 1:
            return X, y

        if method == 'oversample':
            max_count = counts.max()
            resampled_idx = []
            for cls, n in counts.items():
                cls_idx = y[y == cls].index.to_numpy()
                if n < max_count:
                    extra = np.random.choice(
                        cls_idx, size=(max_count - n), replace=True)
                    idx = np.concatenate([cls_idx, extra])
                else:
                    idx = cls_idx
                resampled_idx.append(idx)
            idx = np.concatenate(resampled_idx)
            np.random.shuffle(idx)
            idx = pd.Index(idx)
            return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)

        if method == 'undersample':
            min_count = counts.min()
            resampled_idx = []
            for cls in counts.index:
                cls_idx = y[y == cls].index.to_numpy()
                idx = np.random.choice(cls_idx, size=min_count, replace=False)
                resampled_idx.append(idx)
            idx = np.concatenate(resampled_idx)
            np.random.shuffle(idx)
            idx = pd.Index(idx)
            return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)

        if method == 'smote':
            try:
                from sklearn.neighbors import NearestNeighbors
            except ImportError:
                print("SMOTE requiere scikit-learn; no está disponible")
                return X, y

            X_num = X.select_dtypes(include=[np.number]).copy()
            if X_num.shape[1] == 0:
                print(
                    "ADVERTENCIA: SMOTE requiere columnas numéricas. Se devuelven datos originales.")
                return X, y

            if X_num.isna().any().any():
                X_num = X_num.fillna(X_num.median())

            max_count = counts.max()
            resampled_X = [X_num]
            resampled_y = [y]

            for cls, n in counts.items():
                if n == max_count:
                    continue
                cls_idx = y[y == cls].index
                X_cls = X_num.loc[cls_idx].to_numpy()
                if len(X_cls) < 2:
                    continue
                neigh = NearestNeighbors(n_neighbors=min(
                    5, len(X_cls)), metric='euclidean')
                neigh.fit(X_cls)
                n_samples = max_count - n
                synthetic = []
                for _ in range(n_samples):
                    idx = np.random.randint(0, len(X_cls))
                    nn = neigh.kneighbors(
                        [X_cls[idx]], return_distance=False)[0]
                    nn = nn[nn != idx]
                    if len(nn) == 0:
                        neighbor = X_cls[idx]
                    else:
                        neighbor = X_cls[np.random.choice(nn)]
                    diff = neighbor - X_cls[idx]
                    gap = np.random.rand()
                    synthetic.append(X_cls[idx] + gap * diff)
                if synthetic:
                    synthetic = pd.DataFrame(synthetic, columns=X_num.columns)
                    resampled_X.append(synthetic)
                    resampled_y.append(pd.Series([cls] * len(synthetic)))

            X_bal = pd.concat(resampled_X, ignore_index=True)
            y_bal = pd.concat(resampled_y, ignore_index=True)
            return X_bal, y_bal

        print(f"Método de balanceo desconocido: {method}")
        return X, y

    def _entrenar_clasificador(self, modelo, nombre, scale=True, balance_method=None,
                               class_weight=None, random_state=42, **params):
        print(f"\n{nombre}")
        X_train, y_train = self.X_train.copy(), self.y_train.copy()
        if balance_method in ['oversample', 'undersample', 'smote']:
            X_train, y_train = self._balance_data(X_train, y_train, method=balance_method,
                                                  random_state=random_state)

        if class_weight is not None and _tiene_parametro(modelo, 'class_weight'):
            params['class_weight'] = class_weight
        if _tiene_parametro(modelo, 'random_state') and 'random_state' not in params:
            params['random_state'] = 42

        estimator = modelo(**params)
        pipeline = (make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
                    if scale else make_pipeline(SimpleImputer(strategy='median'), estimator))
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(self.X_test)
        metricas = self._calcular_metricas_clasificacion(
            self.y_test, y_pred, list(np.unique(self.y_test)))
        for k, v in metricas.items():
            print(f"{k}:\n{v}")
        return pipeline, metricas

    def clasificacion_knn(self, n_neighbors=3, algorithm='auto', scale=True):
        return self._entrenar_clasificador(KNeighborsClassifier, "KNN", scale=scale,
                                           n_neighbors=n_neighbors, algorithm=algorithm)

    def clasificacion_decision_tree(self, min_samples_split=2, max_depth=None, scale=False):
        return self._entrenar_clasificador(DecisionTreeClassifier, "Decision Tree", scale=scale,
                                           min_samples_split=min_samples_split, max_depth=max_depth)

    def clasificacion_random_forest(self, n_estimators=100, min_samples_split=2, max_depth=None, scale=False):
        return self._entrenar_clasificador(RandomForestClassifier, "Random Forest", scale=scale,
                                           n_estimators=n_estimators, min_samples_split=min_samples_split,
                                           max_depth=max_depth)

    def clasificacion_xgboost(self, n_estimators=100, min_samples_split=2, max_depth=3, scale=False):
        return self._entrenar_clasificador(GradientBoostingClassifier, "XGBoost", scale=scale,
                                           n_estimators=n_estimators, min_samples_split=min_samples_split,
                                           max_depth=max_depth)

    def clasificacion_adaboost(self, n_estimators=50, estimator=None, scale=False):
        if estimator is None:
            estimator = DecisionTreeClassifier(max_depth=1)
        return self._entrenar_clasificador(AdaBoostClassifier, "AdaBoost", scale=scale,
                                           estimator=estimator, n_estimators=n_estimators)

    @staticmethod
    def _calcular_errores_regresion(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        MSE = np.sum(np.square(y_true - y_pred)) / len(y_true)
        MAE = np.sum(np.abs(y_true - y_pred)) / len(y_true)
        return pd.DataFrame({
            'Métrica': ['RMSE', 'MAE', 'ER'],
            'Valor': [math.sqrt(MSE), MAE, np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))]
        })

    def _entrenar_regresor(self, modelo, nombre, scale=True, **params):
        print(f"\n{nombre}")
        # FIX: usa inspect en lugar de __code__.co_varnames
        if _tiene_parametro(modelo, 'random_state') and 'random_state' not in params:
            params['random_state'] = 42
        estimator = modelo(**params)
        pipeline = (make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
                    if scale else make_pipeline(SimpleImputer(strategy='median'), estimator))
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        errores = self._calcular_errores_regresion(self.y_test, y_pred)
        print(errores)
        return pipeline, errores

    def regresion_lineal(self, scale=True):
        return self._entrenar_regresor(LinearRegression, "Regresión Lineal", scale=scale)

    def regresion_lasso(self, alpha=0.1, scale=True):
        return self._entrenar_regresor(Lasso, f"Lasso (alpha={alpha})", scale=scale, alpha=alpha)

    def regresion_ridge(self, alpha=1.0, scale=True):
        return self._entrenar_regresor(Ridge, f"Ridge (alpha={alpha})", scale=scale, alpha=alpha)

    def regresion_svm(self, kernel='rbf', C=100, epsilon=0.1, scale=True):
        return self._entrenar_regresor(SVR, f"SVM Regresión (kernel={kernel})", scale=scale,
                                       kernel=kernel, C=C, epsilon=epsilon)

    def regresion_decision_tree(self, max_depth=3, scale=False):
        return self._entrenar_regresor(DecisionTreeRegressor, f"Decision Tree (max_depth={max_depth})",
                                       scale=scale, max_depth=max_depth)

    def regresion_random_forest(self, n_estimators=100, max_depth=None, scale=False):
        return self._entrenar_regresor(RandomForestRegressor, f"Random Forest (n={n_estimators})",
                                       scale=scale, n_estimators=n_estimators, max_depth=max_depth)

    def regresion_xgboost(self, n_estimators=100, max_depth=4, scale=False):
        return self._entrenar_regresor(GradientBoostingRegressor, f"XGBoost (n={n_estimators})",
                                       scale=scale, n_estimators=n_estimators, max_depth=max_depth)

    def benchmark_regresion(self):
        print("\n" + "="*70 + "\nBENCHMARK DE REGRESION\n" + "="*70)
        modelos = {
            'Lineal':        self.regresion_lineal,
            'Lasso':         self.regresion_lasso,
            'Ridge':         self.regresion_ridge,
            'SVM (RBF)': lambda: self.regresion_svm(kernel='rbf'),
            'Decision Tree': self.regresion_decision_tree,
            'Random Forest': self.regresion_random_forest,
            'XGBoost':       self.regresion_xgboost,
        }
        resultados = []
        for nombre, func in modelos.items():
            _, errores = func()
            resultados.append({
                'Modelo': nombre,
                'RMSE': errores.loc[errores['Métrica'] == 'RMSE', 'Valor'].values[0],
                'MAE':  errores.loc[errores['Métrica'] == 'MAE',  'Valor'].values[0],
                'ER':   errores.loc[errores['Métrica'] == 'ER',   'Valor'].values[0],
            })
        df_res = pd.DataFrame(resultados).sort_values('RMSE')
        print(f"\nResultados:\n{df_res.to_string(index=False)}")
        return df_res

    def benchmark_clasificacion(self, cv_method='stratified', balance_method=None, n_folds=5):
        print("\n" + "="*70 + "\nBENCHMARK DE CLASIFICACION\n" + "="*70)
        modelos = {
            'KNN':          (KNeighborsClassifier, {'n_neighbors': 5}),
            'Decision Tree': (DecisionTreeClassifier,  {'random_state': 42}),
            'Random Forest': (RandomForestClassifier,  {'n_estimators': 100, 'random_state': 42}),
            'XGBoost':      (GradientBoostingClassifier, {'n_estimators': 100, 'random_state': 42}),
            'AdaBoost':     (AdaBoostClassifier,      {'n_estimators': 50,  'random_state': 42}),
        }
        resultados = []
        for nombre, (modelo, params) in modelos.items():
            df_metricas = self.validacion_cruzada_completa(
                modelo, n_folds=n_folds, cv_method=cv_method,
                balance_method=balance_method, **params
            )
            resultados.append({
                'Modelo':    nombre,
                'Accuracy':  df_metricas.loc[df_metricas['Métrica'] == 'accuracy',           'Test (promedio)'].values[0],
                'Precision': df_metricas.loc[df_metricas['Métrica'] == 'precision_weighted', 'Test (promedio)'].values[0],
                'Recall':    df_metricas.loc[df_metricas['Métrica'] == 'recall_weighted',    'Test (promedio)'].values[0],
                'F1':        df_metricas.loc[df_metricas['Métrica'] == 'f1_weighted',        'Test (promedio)'].values[0],
            })
        df_res = pd.DataFrame(resultados).sort_values('F1', ascending=False)
        print(f"\nResultados:\n{df_res.to_string(index=False)}")
        return df_res

    def benchmark_balanceo(self, modelo=None, n_folds=5, scoring='accuracy',
                           cv_method='stratified', scale=True, **params):
        """Benchmark de balanceo de clases usando distintos métodos.

        Compara el desempeño de un mismo modelo con:
        - Sin balanceo, Oversampling, Undersampling, SMOTE, class_weight='balanced'
        """
        if modelo is None:
            modelo = RandomForestClassifier

        metodos = ['none', 'oversample',
                   'undersample', 'smote', 'class_weight']
        resultados = []

        for metodo in metodos:
            print(f"\n{'='*70}\nBalanceo: {metodo}\n{'='*70}")

            if metodo == 'class_weight':
                if not _tiene_parametro(modelo, 'class_weight'):
                    print(
                        f"  ADVERTENCIA: {modelo.__name__} no acepta class_weight. Omitido.")
                    resultados.append(
                        {'Balance': metodo, 'Promedio': float('nan'), 'Std': float('nan')})
                    continue
                res = self.validacion_cruzada(modelo, n_folds=n_folds, scale=scale,
                                              scoring=scoring, cv_method=cv_method,
                                              class_weight='balanced', **params)
            else:
                res = self.validacion_cruzada(modelo, n_folds=n_folds, scale=scale,
                                              scoring=scoring, cv_method=cv_method,
                                              balance_method=None if metodo == 'none' else metodo,
                                              **params)

            resultados.append({
                'Balance':  metodo,
                'Promedio': res['promedio'],
                'Std':      res['std'],
            })

        df_res = pd.DataFrame(resultados).sort_values(
            'Promedio', ascending=False)
        print(f"\nBenchmark de balanceo:\n{df_res.to_string(index=False)}")
        return df_res

    def validacion_cruzada(self, modelo, n_folds=10, scale=True, scoring='accuracy',
                           cv_method='kfold', balance_method=None, balance_params=None,
                           class_weight=None, **params):
        """Validación cruzada con diferentes métodos de particionado.

        Parámetros
        ----------
        modelo        : clase del modelo (ej: DecisionTreeClassifier)
        n_folds       : número de folds (default=10)
        scale         : aplicar StandardScaler en pipeline (default=True)
        scoring       : métrica de evaluación (default='accuracy')
        cv_method     : 'kfold' | 'stratified' | 'timeseries'
        balance_method: 'oversample' | 'undersample' | 'smote' | None
        balance_params: dict con kwargs adicionales para _balance_data
        class_weight  : valor para el parámetro class_weight del modelo (ej: 'balanced')
        **params      : kwargs adicionales del modelo

        Retorna
        -------
        dict con resultados por fold y estadísticas globales
        """
        print(f"\n{'='*70}")
        print(f"VALIDACION CRUZADA - {modelo.__name__}")
        print(f"{'='*70}")
        print(
            f"Folds: {n_folds} | Métrica: {scoring} | CV method: {cv_method}")
        if balance_method:
            print(f"Balanceo: {balance_method}")

        self._asegurar_df_encoded()
        df = self.df_encoded
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        if _tiene_parametro(modelo, 'random_state') and 'random_state' not in params:
            params['random_state'] = 42
        if class_weight is not None and _tiene_parametro(modelo, 'class_weight'):
            params['class_weight'] = class_weight

        estimator = modelo(**params)
        pipeline = (make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
                    if scale else make_pipeline(SimpleImputer(strategy='median'), estimator))

        if cv_method == 'stratified':
            min_class = int(y.value_counts().min())
            if min_class < 2:
                print(f"ADVERTENCIA: alguna clase tiene solo {min_class} muestra(s). "
                      f"Se cambia a KFold para evitar error en StratifiedKFold.")
                cv_method = 'kfold'
            elif n_folds > min_class:
                print(f"ADVERTENCIA: n_folds={n_folds} excede el mínimo de muestras por clase ({min_class}). "
                      f"Se ajusta a {min_class}.")
                n_folds = min_class

        if cv_method == 'kfold':
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif cv_method == 'stratified':
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=42)
        elif cv_method == 'timeseries':
            cv = TimeSeriesSplit(n_splits=n_folds)
        else:
            raise ValueError(
                "cv_method debe ser 'kfold', 'stratified' o 'timeseries'.")

        scorer = get_scorer(scoring)
        resultados = []
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

            if balance_method:
                X_tr, y_tr = self._balance_data(
                    X_tr, y_tr, method=balance_method, **(balance_params or {}))

            pipeline.fit(X_tr, y_tr)
            score = scorer(pipeline, X_te, y_te)
            resultados.append(score)
            print(f"   Fold {i:2d}: {score:.4f}")

        resultados = np.array(resultados)
        print(f"\n{'='*70}")
        print(f"Promedio: {resultados.mean():.4f} | Std: {resultados.std():.4f} | "
              f"Min: {resultados.min():.4f} | Max: {resultados.max():.4f}")
        print(f"{'='*70}")

        return {
            'resultados': resultados,
            'promedio':   resultados.mean(),
            'std':        resultados.std(),
            'min':        resultados.min(),
            'max':        resultados.max(),
        }

    def validacion_cruzada_completa(self, modelo, n_folds=10, scale=True, cv_method='kfold',
                                    balance_method=None, balance_params=None,
                                    scoring=None, **params):
        """Validación cruzada con múltiples métricas (clasificación y regresión).

        Retorna
        -------
        DataFrame con columnas: Métrica, Test (promedio), Test (std)
        """
        print(f"\n{'='*70}")
        print(f"VALIDACION CRUZADA COMPLETA - {modelo.__name__}")
        print(f"{'='*70}")
        print(f"CV method: {cv_method}")
        if balance_method:
            print(f"Balanceo: {balance_method}")

        self._asegurar_df_encoded()
        df = self.df_encoded
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        if _tiene_parametro(modelo, 'random_state') and 'random_state' not in params:
            params['random_state'] = 42

        estimator = modelo(**params)
        pipeline = (make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
                    if scale else make_pipeline(SimpleImputer(strategy='median'), estimator))

        if cv_method == 'stratified':
            min_class = int(y.value_counts().min())
            if min_class < 2:
                print(f"ADVERTENCIA: alguna clase tiene solo {min_class} muestra(s). "
                      f"Se cambia a KFold para evitar error en StratifiedKFold.")
                cv_method = 'kfold'
            elif n_folds > min_class:
                print(f"ADVERTENCIA: n_folds={n_folds} excede el mínimo de muestras por clase ({min_class}). "
                      f"Se ajusta a {min_class}.")
                n_folds = min_class

        if cv_method == 'kfold':
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif cv_method == 'stratified':
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=42)
        elif cv_method == 'timeseries':
            cv = TimeSeriesSplit(n_splits=n_folds)
        else:
            raise ValueError(
                "cv_method debe ser 'kfold', 'stratified' o 'timeseries'.")

        es_clasificacion = y.dtype == 'object' or len(np.unique(y)) < 20

        if es_clasificacion:
            metric_funcs = {
                'accuracy':           accuracy_score,
                'precision_weighted': lambda yt, yp: precision_score(yt, yp, average='weighted', zero_division=0),
                'recall_weighted': lambda yt, yp: recall_score(yt, yp, average='weighted', zero_division=0),
                'f1_weighted': lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0),
            }
        else:
            metric_funcs = {
                'mse': lambda yt, yp: mean_squared_error(yt, yp),
                'mae': lambda yt, yp: mean_absolute_error(yt, yp),
                'r2': lambda yt, yp: r2_score(yt, yp),
            }

        scores = {m: [] for m in metric_funcs}
        for train_idx, test_idx in cv.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

            if balance_method:
                X_tr, y_tr = self._balance_data(
                    X_tr, y_tr, method=balance_method, **(balance_params or {}))

            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_te)

            for name, func in metric_funcs.items():
                scores[name].append(func(y_te, y_pred))

        df_resultados = pd.DataFrame([{
            'Métrica':        name,
            'Test (promedio)': np.mean(vals),
            'Test (std)':      np.std(vals),
        } for name, vals in scores.items()])

        print(f"\nResultados:\n{df_resultados.to_string(index=False)}")
        print(f"{'='*70}")
        return df_resultados

    def optimizar_con_ga(self, tipo='clasificacion', modelo='random_forest',
                         pop_size=8, generations=8):
        if not GENETIC_AVAILABLE:
            print(
                "ADVERTENCIA: sklearn-genetic-opt no instalado\nInstala: pip install sklearn-genetic-opt")
            return None, None
        print("\n" + "="*70 + "\nOPTIMIZACION CON ALGORITMOS GENETICOS\n" + "="*70)

        if modelo == 'random_forest':
            estimator = (RandomForestClassifier(random_state=42, n_jobs=-1) if tipo == 'clasificacion'
                         else RandomForestRegressor(random_state=42, n_jobs=-1))
            param_grid = {
                'n_estimators':     Integer(50, 200),
                'max_depth':        Integer(3, 20),
                'min_samples_split': Integer(2, 10),
                'min_samples_leaf': Integer(1, 5),
            }
        else:
            estimator = (GradientBoostingClassifier(random_state=42) if tipo == 'clasificacion'
                         else GradientBoostingRegressor(random_state=42))
            param_grid = {
                'n_estimators':     Integer(50, 200),
                'learning_rate':    Continuous(0.01, 0.3),
                'max_depth':        Integer(3, 10),
                'min_samples_split': Integer(2, 10),
            }

        ga_search = GASearchCV(
            estimator=estimator, cv=3,
            scoring='accuracy' if tipo == 'clasificacion' else 'neg_mean_squared_error',
            population_size=pop_size, generations=generations,
            n_jobs=-1, verbose=False, param_grid=param_grid)

        print(f"Ejecutando GA para {modelo} ({tipo})...")
        ga_search.fit(self.X_train, self.y_train)
        print(
            f"\nOptimización completada!\n   Mejor score (CV): {ga_search.best_score_:.4f}\n   Mejores parámetros:")
        for param, valor in ga_search.best_params_.items():
            print(f"      - {param}: {valor}")

        y_pred = ga_search.best_estimator_.predict(self.X_test)
        if tipo == 'clasificacion':
            print(
                f"   Accuracy en Test: {accuracy_score(self.y_test, y_pred):.4f}")
        else:
            print(
                f"\n   Errores en Test:\n{self._calcular_errores_regresion(self.y_test, y_pred).to_string(index=False)}")
        return ga_search.best_estimator_, ga_search

    def arima_model(self, order=(1, 1, 1)):
        if not STATSMODELS_AVAILABLE:
            print(
                "ADVERTENCIA: statsmodels no disponible\nInstala: pip install statsmodels")
            return None
        print(f"\nModelo ARIMA{order}")
        print("Para análisis completo usa: SeriesTiempo(ts).arima(...)")
        return None

    def sarima_model(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        if not STATSMODELS_AVAILABLE:
            print(
                "ADVERTENCIA: statsmodels no disponible\nInstala: pip install statsmodels")
            return None
        print(f"\nModelo SARIMA{order}x{seasonal_order}")
        print("Para análisis completo usa: SeriesTiempo(ts)")
        return None

    def prophet_model(self, seasonality_mode='additive', changepoint_prior_scale=0.05):
        print("\nModelo Prophet — requiere: pip install prophet")
        return None

    def exponential_smoothing(self, seasonal='add', seasonal_periods=12):
        if not STATSMODELS_AVAILABLE:
            print(
                "ADVERTENCIA: statsmodels no disponible\nInstala: pip install statsmodels")
            return None
        print("\nSuavizado Exponencial (Holt-Winters)")
        print("Usa SeriesTiempo(ts=tu_serie).holt_winters(...)")
        return None

# ============================================================================
# CLASE NO SUPERVISADO
# ============================================================================


class NoSupervisado(EDA):
    def __init__(self, df):
        super().__init__(df=df)
        self.df_scaled = None
        print(f"NoSupervisado inicializado - Shape: {self.df.shape}")

    def escalar_datos(self):
        self.df_scaled = pd.DataFrame(StandardScaler().fit_transform(self.df),
                                      columns=self.df.columns, index=self.df.index)
        print(f"Datos escalados: {self.df_scaled.shape}")
        return self

    def pca(self, n_componentes=2, plot=True, show=True):
        if PCA_Prince is None:
            print("ADVERTENCIA: Librería 'prince' no disponible, usando sklearn PCA")
            return self.pca_sklearn(n_componentes, plot, show=show)
        print(f"\nPCA con {n_componentes} componentes")
        datos = self.df_scaled if self.df_scaled is not None else self.df
        modelo = PCA_Prince(n_components=n_componentes).fit(datos)
        coordenadas = modelo.row_coordinates(datos)
        var_explicada = modelo.percentage_of_variance_
        print(f"\nVarianza explicada por componente:")
        for i, var in enumerate(var_explicada):
            print(f"   PC{i+1}: {var:.2f}%")
        print(f"   Total: {sum(var_explicada):.2f}%")
        fig = None
        if plot and n_componentes >= 2:
            fig = self._plot_pca(coordenadas, var_explicada,
                                 modelo.column_correlations, show=show)
        return modelo, coordenadas, var_explicada, fig

    def pca_sklearn(self, n_componentes=2, plot=True, scale=True, show=True):
        print(f"\nPCA (sklearn) con {n_componentes} componentes")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        if scale and self.df_scaled is None:
            pipeline = make_pipeline(
                StandardScaler(), PCA(n_components=n_componentes))
            componentes = pipeline.fit_transform(datos)
            pca = pipeline.named_steps['pca']
        else:
            pca = PCA(n_components=n_componentes)
            componentes = pca.fit_transform(datos)
            pipeline = pca

        var_explicada = pca.explained_variance_ratio_ * 100
        print(
            f"\nVarianza explicada: {[f'{v:.2f}%' for v in var_explicada]}, Total: {sum(var_explicada):.2f}%")

        fig = None
        if plot and n_componentes >= 2:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.6)
            ax.set_xlabel(f'PC1 ({var_explicada[0]:.2f}%)')
            ax.set_ylabel(f'PC2 ({var_explicada[1]:.2f}%)')
            ax.set_title('PCA - Plano Principal')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if show:
                plt.show()
        return pipeline, componentes, var_explicada, fig

    def _plot_pca(self, coordenadas, var_explicada, correlaciones, show=True):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
        x, y = coordenadas[0].values, coordenadas[1].values
        axes[0].scatter(x, y, alpha=0.6)
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_xlabel(f'PC1 ({var_explicada[0]:.2f}%)')
        axes[0].set_ylabel(f'PC2 ({var_explicada[1]:.2f}%)')
        axes[0].set_title('Plano Principal')
        axes[0].grid(True, alpha=0.3)

        cor = correlaciones.iloc[:, [0, 1]].values
        circle = plt.Circle((0, 0), 1, color='steelblue', fill=False)
        axes[1].add_patch(circle)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        for i in range(cor.shape[0]):
            axes[1].arrow(0, 0, cor[i, 0]*0.95, cor[i, 1]*0.95,
                          color='steelblue', alpha=0.6, head_width=0.05)
            axes[1].text(cor[i, 0]*1.1, cor[i, 1]*1.1,
                         correlaciones.index[i], fontsize=9, ha='center')
        axes[1].set_xlim(-1.2, 1.2)
        axes[1].set_ylim(-1.2, 1.2)
        axes[1].set_xlabel(f'PC1 ({var_explicada[0]:.2f}%)')
        axes[1].set_ylabel(f'PC2 ({var_explicada[1]:.2f}%)')
        axes[1].set_title('Círculo de Correlación')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def _ejecutar_clustering(self, modelo, nombre, n_clusters, plot, scale=True, show=True, **kwargs):
        print(f"\n{nombre} con {n_clusters} clusters")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        if scale and self.df_scaled is None:
            pipeline = make_pipeline(StandardScaler(), modelo)
            clusters = pipeline.fit_predict(datos)
            centros = pipeline.named_steps[list(pipeline.named_steps.keys())[
                1]].cluster_centers_
            datos_escalados = pipeline.named_steps['standardscaler'].transform(
                datos)
        else:
            pipeline = modelo
            clusters = modelo.fit_predict(datos)
            centros = modelo.cluster_centers_
            datos_escalados = datos

        print(f"\nDistribución:")
        for i in range(n_clusters):
            count = np.sum(clusters == i)
            print(f"   Cluster {i}: {count} ({count/len(clusters)*100:.1f}%)")

        fig = None
        if plot:
            fig = self._plot_clusters(
                datos_escalados, clusters, centros, nombre, show=show)
        return pipeline, clusters, centros, fig

    def kmeans(self, n_clusters=3, max_iter=500, n_init=150, plot=True, scale=True, show=True):
        return self._ejecutar_clustering(
            KMeans(n_clusters=n_clusters, max_iter=max_iter,
                   n_init=n_init, random_state=42),
            "K-Means", n_clusters, plot, scale=scale, show=show)

    def kmedoids(self, n_clusters=3, max_iter=500, plot=True, scale=True, show=True):
        return self._ejecutar_clustering(
            KMedoids(n_clusters=n_clusters, max_iter=max_iter,
                     metric='cityblock', random_state=42),
            "K-Medoids", n_clusters, plot, scale=scale, show=show)

    def _plot_clusters(self, datos, clusters, centros, titulo, show=True):
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(datos)
        centros_pca = pca.transform(centros)
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        colores = ['red', 'green', 'blue', 'orange',
                   'purple', 'brown', 'pink', 'gray']
        for i in range(len(centros)):
            mask = clusters == i
            ax.scatter(componentes[mask, 0], componentes[mask, 1], c=colores[i % len(colores)],
                       label=f'Cluster {i}', alpha=0.6, s=50)
        ax.scatter(centros_pca[:, 0], centros_pca[:, 1], c='black', marker='X', s=200,
                   label='Centroides', edgecolors='white', linewidths=2)
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_title(f'{titulo} - Visualización con PCA')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def hac(self, n_clusters=3, metodo='ward', plot=True, scale=True, show=True):
        print(f"\nClustering Jerárquico ({metodo}) con {n_clusters} clusters")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        if scale and self.df_scaled is None:
            scaler = StandardScaler()
            datos_para_hac = scaler.fit_transform(datos)
        else:
            datos_para_hac = datos

        Z = {'ward': ward, 'average': average, 'single': single,
             'complete': complete}[metodo](datos_para_hac)
        clusters = fcluster(Z, n_clusters, criterion='maxclust') - 1
        print(f"\nDistribución:")
        for i in range(n_clusters):
            count = np.sum(clusters == i)
            print(f"   Cluster {i}: {count} ({count/len(clusters)*100:.1f}%)")

        fig = None
        if plot:
            fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
            dendrogram(Z, labels=datos.index.tolist(), ax=ax)
            ax.set_title(f'Dendrograma - Método {metodo.capitalize()}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            if show:
                plt.show()
        return Z, clusters, fig

    def tsne(self, n_componentes=2, perplexity=30, plot=True, scale=True, show=True):
        if TSNE is None:
            print("ADVERTENCIA: t-SNE no disponible")
            return None, None, None
        print(f"\nt-SNE con {n_componentes} componentes")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        if scale and self.df_scaled is None:
            pipeline = make_pipeline(StandardScaler(), TSNE(
                n_components=n_componentes, perplexity=perplexity, random_state=42))
            componentes = pipeline.fit_transform(datos)
        else:
            componentes = TSNE(n_components=n_componentes,
                               perplexity=perplexity, random_state=42).fit_transform(datos)
            pipeline = None

        fig = None
        if plot and n_componentes >= 2:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.6)
            ax.set_xlabel('Componente t-SNE 1')
            ax.set_ylabel('Componente t-SNE 2')
            ax.set_title('t-SNE - Reducción Dimensional')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if show:
                plt.show()
        print("t-SNE completado")
        return pipeline, componentes, fig

    def umap_reduction(self, n_componentes=2, n_neighbors=15, plot=True, scale=True, show=True):
        if um is None:
            print("ADVERTENCIA: UMAP no disponible. Instala: pip install umap-learn")
            return None, None, None
        print(f"\nUMAP con {n_componentes} componentes")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        if scale and self.df_scaled is None:
            pipeline = make_pipeline(StandardScaler(), um.UMAP(
                n_components=n_componentes, n_neighbors=n_neighbors, random_state=42))
            componentes = pipeline.fit_transform(datos)
        else:
            modelo_umap = um.UMAP(n_components=n_componentes,
                                  n_neighbors=n_neighbors, random_state=42)
            componentes = modelo_umap.fit_transform(datos)
            pipeline = modelo_umap

        fig = None
        if plot and n_componentes >= 2:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.6)
            ax.set_xlabel('Componente UMAP 1')
            ax.set_ylabel('Componente UMAP 2')
            ax.set_title('UMAP - Reducción Dimensional')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if show:
                plt.show()
        print("UMAP completado")
        return pipeline, componentes, fig

    @staticmethod
    def bar_plot(centros, labels, scale=False, figsize=(15, 8), show=True):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        centros_plot = np.copy(centros)
        if scale:
            for col in range(centros_plot.shape[1]):
                max_val = np.max(np.abs(centros_plot[:, col]))
                if max_val > 0:
                    centros_plot[:, col] = centros_plot[:, col] / max_val
        n_clusters = centros_plot.shape[0]
        x = np.arange(len(labels))
        width = 0.8 / n_clusters
        for i in range(n_clusters):
            ax.bar(x + width * i - (width * (n_clusters - 1) / 2),
                   centros_plot[i], width, label=f'Cluster {i}', alpha=0.8)
        ax.set_xlabel('Variables')
        ax.set_ylabel('Valor')
        ax.set_title('Comparación de Centroides por Cluster')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    @staticmethod
    def radar_plot(centros, labels, show=True):
        centros_norm = np.array([((n - min(n)) / (max(n) - min(n)) * 100) if max(n) != min(n) else (n/n * 50)
                                 for n in centros.T])
        angulos = [n / float(len(labels)) * 2 *
                   pi for n in range(len(labels))] + [0]
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150,
                               subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angulos[:-1], labels)
        plt.yticks([25, 50, 75, 100], ["25%", "50%",
                   "75%", "100%"], color="grey", size=8)
        plt.ylim(0, 100)
        colores = ['blue', 'red', 'green', 'orange', 'purple']
        for i in range(centros_norm.shape[1]):
            valores = centros_norm[:, i].tolist(
            ) + [centros_norm[:, i].tolist()[0]]
            ax.plot(angulos, valores, linewidth=2,
                    label=f'Cluster {i}', color=colores[i % len(colores)])
            ax.fill(angulos, valores, alpha=0.25,
                    color=colores[i % len(colores)])
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Radar Plot - Comparación de Clusters', y=1.08)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

# ============================================================================
# CLASES DE SERIES DE TIEMPO
# ============================================================================


class BasePrediccion(metaclass=ABCMeta):
    @abstractmethod
    def forecast(self):
        pass


class Prediccion(BasePrediccion):
    def __init__(self, modelo):
        self.__modelo = modelo

    @property
    def modelo(self):
        return self.__modelo

    @modelo.setter
    def modelo(self, modelo):
        if isinstance(modelo, Modelo):
            self.__modelo = modelo
        else:
            warnings.warn('El objeto debe ser una instancia de Modelo.')


class BaseModelo(metaclass=ABCMeta):
    @abstractmethod
    def fit(self):
        pass


class Modelo(BaseModelo):
    def __init__(self, ts):
        self.__ts = ts
        self._coef = None

    @property
    def ts(self):
        return self.__ts

    @ts.setter
    def ts(self, ts):
        if isinstance(ts, pd.core.series.Series):
            if isinstance(ts.index, pd.DatetimeIndex) and ts.index.freqstr is not None:
                self.__ts = ts
            else:
                warnings.warn(
                    'ERROR: La serie debe tener un DatetimeIndex con frecuencia especificada.')
        else:
            warnings.warn(
                'ERROR: El parámetro ts no es una instancia de serie de tiempo.')

    @property
    def coef(self):
        return self._coef


def _safe_freq(ts):
    """Retorna la frecuencia de la serie o 'D' como fallback seguro.

    FIX: evita AttributeError en forecast cuando la frecuencia no pudo inferirse.
    """
    freq = getattr(ts.index, 'freq', None)
    if freq is not None:
        return freq
    freqstr = getattr(ts.index, 'freqstr', None)
    if freqstr:
        return freqstr
    # Fallback: inferir de los datos
    inferred = pd.infer_freq(ts.index)
    return inferred if inferred else 'D'


# Modelos Básicos de Predicción
class meanfPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = [self.modelo.coef for _ in range(steps)]
        freq = _safe_freq(self.modelo.ts)
        fechas = pd.date_range(
            start=self.modelo.ts.index[-1], periods=steps + 1, freq=freq)
        return pd.Series(res, index=fechas[1:])


class naivePrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = [self.modelo.coef for _ in range(steps)]
        freq = _safe_freq(self.modelo.ts)
        fechas = pd.date_range(
            start=self.modelo.ts.index[-1], periods=steps + 1, freq=freq)
        return pd.Series(res, index=fechas[1:])


class snaivePrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = []
        pos = 0
        for _ in range(steps):
            if pos >= len(self.modelo.coef):
                pos = 0
            res.append(self.modelo.coef[pos])
            pos += 1
        freq = _safe_freq(self.modelo.ts)
        fechas = pd.date_range(
            start=self.modelo.ts.index[-1], periods=steps + 1, freq=freq)
        return pd.Series(res, index=fechas[1:])


class driftPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = [self.modelo.ts[-1] + self.modelo.coef * i for i in range(steps)]
        freq = _safe_freq(self.modelo.ts)
        fechas = pd.date_range(
            start=self.modelo.ts.index[-1], periods=steps + 1, freq=freq)
        return pd.Series(res, index=fechas[1:])


class meanf(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self):
        self._coef = statistics.mean(self.ts)
        return meanfPrediccion(self)


class naive(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self):
        self._coef = self.ts[-1]
        return naivePrediccion(self)


class snaive(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self, h=1):
        self._coef = self.ts.values[-h:]
        return snaivePrediccion(self)


class drift(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self):
        self._coef = (self.ts[-1] - self.ts[0]) / len(self.ts)
        return driftPrediccion(self)


# Holt-Winters
class HW_Prediccion(Prediccion):
    def __init__(self, modelo, alpha, beta, gamma):
        super().__init__(modelo)
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma

    @property
    def alpha(self):
        return self.__alpha

    @property
    def beta(self):
        return self.__beta

    @property
    def gamma(self):
        return self.__gamma

    def forecast(self, steps=1):
        return self.modelo.forecast(steps)


class HW_calibrado(Modelo):
    def __init__(self, ts, test, trend=None, seasonal=None):
        super().__init__(ts)
        self.__test = test
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels no disponible. Instala: pip install statsmodels")
        if seasonal is not None:
            self.__modelo = ExponentialSmoothing(
                ts, trend=trend, seasonal=seasonal)
        else:
            self.__modelo = ExponentialSmoothing(ts, trend=trend)

    @property
    def test(self):
        return self.__test

    @test.setter
    def test(self, test):
        if isinstance(test, pd.core.series.Series):
            if test.index.freqstr is not None:
                self.__test = test
            else:
                warnings.warn(
                    'ERROR: No se indica la frecuencia de la serie de tiempo.')
        else:
            warnings.warn(
                'ERROR: El parámetro ts no es una instancia de serie de tiempo.')

    def fit(self, paso=0.1):
        if self.__modelo.trend is None and self.__modelo.seasonal is None:
            model_fit = self.__modelo.fit()
            alpha = getattr(model_fit.params, 'smoothing_level', None)
            beta = getattr(model_fit.params, 'smoothing_trend', None)
            gamma = getattr(model_fit.params, 'smoothing_seasonal', None)
            return HW_Prediccion(model_fit, alpha, beta, gamma)

        error = float("inf")
        best_model = None
        best_params = {'alpha': None, 'beta': None, 'gamma': None}
        n = np.append(np.arange(0, 1, paso), 1)
        has_trend = self.__modelo.trend is not None
        has_seasonal = self.__modelo.seasonal is not None

        for alpha in n:
            for beta in (n if has_trend else [None]):
                for gamma in (n if has_seasonal else [None]):
                    fit_kwargs = {'smoothing_level': alpha}
                    if beta is not None:
                        fit_kwargs['smoothing_trend'] = beta
                    if gamma is not None:
                        fit_kwargs['smoothing_seasonal'] = gamma
                    try:
                        model_fit = self.__modelo.fit(**fit_kwargs)
                        pred = np.array(model_fit.forecast(len(self.test)))
                        mse = np.mean((pred - self.test.values) ** 2)
                        if mse < error:
                            error = mse
                            best_model = model_fit
                            best_params = {'alpha': alpha,
                                           'beta': beta, 'gamma': gamma}
                    except Exception:
                        continue

        if best_model is None:
            model_fit = self.__modelo.fit()
            alpha = getattr(model_fit.params, 'smoothing_level', None)
            beta = getattr(model_fit.params, 'smoothing_trend', None)
            gamma = getattr(model_fit.params, 'smoothing_seasonal', None)
            return HW_Prediccion(model_fit, alpha, beta, gamma)

        return HW_Prediccion(best_model, best_params['alpha'], best_params['beta'], best_params['gamma'])


# LSTM para Series de Tiempo
class LSTM_TSPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)
        if not KERAS_AVAILABLE:
            raise ImportError(
                "Keras/TensorFlow no disponible. Instala: pip install tensorflow")
        self.__scaler = MinMaxScaler(feature_range=(0, 1))
        self.__X = self.__scaler.fit_transform(self.modelo.ts.to_frame())

    def __split_sequence(self, sequence, n_steps):
        X, y = [], []
        for i in range(n_steps, len(sequence)):
            X.append(self.__X[i-n_steps:i, 0])
            y.append(self.__X[i, 0])
        return np.array(X), np.array(y)

    def forecast(self, steps=1):
        res = []
        p = self.modelo.p
        for i in range(steps):
            y_pred = [self.__X[-p:].tolist()]
            X, y = self.__split_sequence(self.__X, p)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            self.modelo.m.fit(X, y, epochs=10, batch_size=1, verbose=0)
            pred = self.modelo.m.predict(y_pred, verbose=0)
            res.append(self.__scaler.inverse_transform(pred).tolist()[0][0])
            self.__X = np.append(self.__X, pred.tolist(), axis=0)

        freq = _safe_freq(self.modelo.ts)
        fechas = pd.date_range(
            start=self.modelo.ts.index[-1], periods=steps + 1, freq=freq)
        return pd.Series(res, index=fechas[1:])


class LSTM_TS(Modelo):
    def __init__(self, ts, p=1, lstm_units=50, dense_units=1, optimizer='rmsprop', loss='mse'):
        super().__init__(ts)
        if not KERAS_AVAILABLE:
            raise ImportError(
                "Keras/TensorFlow no disponible. Instala: pip install tensorflow")
        try:
            from keras.models import Sequential
            from keras.layers import LSTM, Dense
        except ImportError:
            raise ImportError("Keras no disponible")
        self.__p = p
        self.__m = Sequential()
        self.__m.add(LSTM(units=lstm_units, input_shape=(p, 1)))
        self.__m.add(Dense(units=dense_units))
        self.__m.compile(optimizer=optimizer, loss=loss)

    @property
    def m(self):
        return self.__m

    @property
    def p(self):
        return self.__p

    def fit(self):
        return LSTM_TSPrediccion(self)


# Clase de Errores para Series de Tiempo
class ts_error:
    def __init__(self, preds, real, nombres=None):
        self.__preds = preds if isinstance(preds, list) else [preds]
        self.__real = real
        self.__nombres = nombres

    @property
    def preds(self):
        return self.__preds

    @preds.setter
    def preds(self, preds):
        if isinstance(preds, (pd.core.series.Series, np.ndarray)):
            self.__preds = [preds]
        elif isinstance(preds, list):
            self.__preds = preds
        else:
            warnings.warn('ERROR: preds debe ser una serie o lista de series.')

    @property
    def real(self):
        return self.__real

    @real.setter
    def real(self, real):
        self.__real = real

    @property
    def nombres(self):
        return self.__nombres

    @nombres.setter
    def nombres(self, nombres):
        if isinstance(nombres, str):
            nombres = [nombres]
        if len(nombres) == len(self.__preds):
            self.__nombres = nombres
        else:
            warnings.warn(
                'ERROR: Los nombres no calzan con la cantidad de métodos.')

    def RSS(self):
        return [sum((pred - self.real)**2) for pred in self.preds]

    def MSE(self):
        return [rss / len(self.real) for rss in self.RSS()]

    def RMSE(self):
        return [math.sqrt(mse) for mse in self.MSE()]

    def RE(self):
        return [sum(abs(self.real - pred)) / sum(abs(self.real)) for pred in self.preds]

    def CORR(self):
        res = []
        for pred in self.preds:
            corr = corrcoef(self.real, pred)[0, 1]
            res.append(0 if math.isnan(corr) else corr)
        return res

    def df_errores(self):
        res = pd.DataFrame({'MSE': self.MSE(), 'RMSE': self.RMSE(),
                            'RE': self.RE(), 'CORR': self.CORR()})
        if self.nombres is not None:
            res.index = self.nombres
        return res

    def __escalar(self):
        res = self.df_errores()
        for nombre in res.columns.values:
            res[nombre] = res[nombre] - min(res[nombre])
            max_val = max(res[nombre])
            if max_val > 0:
                res[nombre] = res[nombre] / max_val * 100
        return res

    def plot_errores(self, show=True):
        fig = plt.figure(figsize=(8, 8))
        df = self.__escalar()
        if len(df) == 1:
            df.loc[0] = 100

        N = len(df.columns.values)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], df.columns.values)
        ax.set_rlabel_position(0)
        plt.yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"],
                   color="grey", size=10)
        plt.ylim(-10, 110)
        for i in df.index.values:
            p = df.loc[i].values.tolist() + df.loc[i].values.tolist()[:1]
            ax.plot(angles, p, linewidth=1, linestyle='solid', label=i)
            ax.fill(angles, p, alpha=0.1)
        plt.legend(loc='best')
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plotly_errores(self):
        if not PLOTLY_AVAILABLE:
            print("Plotly no disponible. Usando matplotlib...")
            return self.plot_errores()

        df = self.__escalar()
        etqs = df.columns.values.tolist() + df.columns.values.tolist()[:1]
        if len(df) == 1:
            df.loc[0] = 100

        fig = go.Figure()
        for i in df.index.values:
            p = df.loc[i].values.tolist() + df.loc[i].values.tolist()[:1]
            fig.add_trace(go.Scatterpolar(
                r=p, theta=etqs, fill='toself', name=i))
        fig.update_layout(polar=dict(
            radialaxis=dict(visible=True, range=[-10, 110])))
        return fig


# Clase Periodograma
class Periodograma:
    def __init__(self, ts):
        self.__ts = ts
        self.__freq, self.__spec = signal.periodogram(ts)

    @property
    def ts(self):
        return self.__ts

    @property
    def freq(self):
        return self.__freq

    @property
    def spec(self):
        return self.__spec

    def mejor_freq(self, best=3):
        res = np.argsort(-self.spec)
        res = res[res != 0][0:best]
        return self.freq[res]

    def mejor_periodos(self, best=3):
        return 1 / self.mejor_freq(best)

    def plot_periodograma(self, best=3, show=True):
        res = self.mejor_freq(best)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.freq, self.spec, color="darkgray")
        for i in range(best):
            ax.axvline(
                x=res[i], label=f"Mejor {i + 1}", ls='--', c=np.random.rand(3,))
        ax.set_xlabel('Frecuencia')
        ax.set_ylabel('Densidad Espectral')
        ax.set_title('Periodograma')
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plotly_periodograma(self, best=3):
        if not PLOTLY_AVAILABLE:
            print("Plotly no disponible. Usando matplotlib...")
            return self.plot_periodograma(best)

        res = self.mejor_freq(best)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.freq, y=self.spec,
                      mode='lines+markers', line_color='darkgray'))
        for i in range(best):
            v = np.random.rand(3)
            color = f"rgb({v[0]}, {v[1]}, {v[2]})"
            fig.add_vline(x=res[i], line_width=2, line_dash="dash",
                          annotation_text=f"Mejor {i + 1}", line_color=color)
        fig.update_layout(
            title='Periodograma', xaxis_title='Frecuencia', yaxis_title='Densidad Espectral')
        return fig


def _auto_d(ts, max_d=2, significance=0.05):
    """Detecta orden de diferenciación óptimo con ADF test."""
    if not STATSMODELS_AVAILABLE:
        return 1
    ts_series = pd.Series(ts) if not isinstance(ts, pd.Series) else ts
    for d in range(max_d + 1):
        try:
            serie_diff = ts_series.diff(
                d).dropna() if d > 0 else ts_series.dropna()
            if len(serie_diff) < 10:
                continue
            p_value = adfuller(serie_diff, autolag='AIC')[1]
            if p_value < significance:
                return d
        except Exception:
            continue
    return 1


def _walk_forward_arima(ts_train, ts_test, order, seasonal_order=(0, 0, 0, 0)):
    """Forecast ARIMA paso a paso refitando con valores reales."""
    if not STATSMODELS_AVAILABLE:
        return None
    history = list(ts_train.values)
    predictions = []
    p, d, q = order
    trend = 'c' if d == 0 else 'n'
    for i in range(len(ts_test)):
        try:
            modelo = SARIMAX(history, order=order, seasonal_order=seasonal_order,
                             trend=trend, enforce_stationarity=True, enforce_invertibility=True)
            resultado = modelo.fit(disp=False, maxiter=200)
            pred = float(resultado.forecast(steps=1)[0])
            predictions.append(pred)
        except Exception:
            predictions.append(history[-1])
        history.append(ts_test.values[i])
    return np.array(predictions)


# Clase Principal de Series de Tiempo
class SeriesTiempo:

    def __init__(self, ts=None, path=None, date_col='fecha', value_col=None, freq='D'):

        if ts is not None:
            if not isinstance(ts, pd.Series):
                raise ValueError("ts debe ser una pandas Series")
            if not isinstance(ts.index, pd.DatetimeIndex):
                raise ValueError(
                    "El índice de la serie debe ser DatetimeIndex. "
                    "Use pd.to_datetime() para convertir.")
            if ts.index.isna().any():
                raise ValueError(
                    "El índice contiene valores NaT. Verifique que las fechas sean válidas.")
            if not pd.api.types.is_numeric_dtype(ts):
                try:
                    ts = pd.to_numeric(ts, errors='coerce')
                    print("Los valores se convirtieron a numérico")
                except Exception:
                    raise ValueError(
                        "Los valores de la serie deben ser numéricos")
            if ts.isna().any():
                print(
                    f"Advertencia: {ts.isna().sum()} valores no numéricos → NaN. Se rellenarán.")
                ts = ts.ffill().bfill()
            self.ts = ts

            # Inferir frecuencia si no está definida
            if self.ts.index.freq is None:
                try:
                    inferred_freq = pd.infer_freq(self.ts.index)
                    if inferred_freq:
                        self.ts.index.freq = inferred_freq
                    else:
                        try:
                            time_diffs = self.ts.index.to_series().diff().dropna()
                            if len(time_diffs) > 0:
                                most_common_diff = time_diffs.value_counts(
                                ).index[0]
                                if most_common_diff.days == 1:
                                    self.ts.index.freq = 'D'
                                elif most_common_diff.days == 7:
                                    self.ts.index.freq = 'W'
                                elif 28 <= most_common_diff.days <= 31:
                                    self.ts.index.freq = 'ME'
                        except Exception:
                            pass
                except Exception:
                    pass

            # Segunda pasada de NaN tras conversión (por si bfill no alcanzó)
            if self.ts.isna().any():
                print(
                    f"Advertencia: {self.ts.isna().sum()} NaN tras conversión. Se rellenarán.")
                # FIX: mismo fix de ffill/bfill
                self.ts = self.ts.ffill().bfill()

        elif path is not None:
            df = pd.read_csv(path)
            if value_col is None:
                value_col = [c for c in df.columns if c != date_col][0]
            try:
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
            except ValueError as e:
                raise ValueError(f"Error al convertir fechas en {path}: {e}")
            df = df.set_index(date_col)
            self.ts = df[value_col]
            self.ts.index.freq = freq
        else:
            raise ValueError("Debe proporcionar 'ts' o 'path'")

        print(f"Serie de tiempo cargada: {len(self.ts)} observaciones")
        freq_str = getattr(self.ts.index, 'freqstr', None)
        if freq_str:
            print(f"Frecuencia: {freq_str}")

    def info(self):
        print("\n" + "="*70)
        print("INFORMACION DE SERIE DE TIEMPO")
        print("="*70)
        print(f"Observaciones: {len(self.ts)}")
        freq_str = getattr(self.ts.index, 'freqstr', None)
        print(f"Frecuencia: {freq_str if freq_str else 'No especificada'}")
        print(f"Rango: {self.ts.index[0]} a {self.ts.index[-1]}")
        print(f"\nEstadísticas:\n{self.ts.describe()}")
        return self

    def plot(self, title='Serie de Tiempo', figsize=(12, 6), show=True):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        self.ts.plot(ax=ax, marker='o', markersize=3)
        ax.set_title(title)
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Valor')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plotly_plot(self, title='Serie de Tiempo'):
        if not PLOTLY_AVAILABLE:
            print("Plotly no disponible. Usando matplotlib...")
            return self.plot(title)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.ts.index, y=self.ts.values,
                                 mode='lines+markers', name='Serie'))
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(title=title, xaxis_title='Fecha',
                          yaxis_title='Valor')
        fig.show()
        return self

    def meanf(self):
        return meanf(self.ts).fit()

    def naive(self):
        return naive(self.ts).fit()

    def snaive(self, h=1):
        return snaive(self.ts).fit(h)

    def drift(self):
        return drift(self.ts).fit()

    def holt_winters(self, trend='add', seasonal='add', seasonal_periods=None):
        if not STATSMODELS_AVAILABLE:
            print("statsmodels no disponible. Instala: pip install statsmodels")
            return None
        modelo = ExponentialSmoothing(self.ts, trend=trend, seasonal=seasonal,
                                      seasonal_periods=seasonal_periods)
        modelo_fit = modelo.fit()
        print(f"Holt-Winters ajustado (trend={trend}, seasonal={seasonal})")
        return modelo_fit

    def holt_winters_calibrado(self, test, paso=0.1, trend='add', seasonal='add'):
        modelo = HW_calibrado(self.ts, test, trend, seasonal)
        resultado = modelo.fit(paso)
        # FIX: alpha/beta/gamma pueden ser None → format seguro
        def fmt(v): return f"{v:.3f}" if v is not None else "N/A"
        print(f"HW Calibrado - alpha: {fmt(resultado.alpha)}, "
              f"beta: {fmt(resultado.beta)}, gamma: {fmt(resultado.gamma)}")
        return resultado

    def lstm(self, p=1, lstm_units=50, dense_units=1, optimizer='rmsprop', loss='mse'):
        modelo = LSTM_TS(self.ts, p, lstm_units, dense_units, optimizer, loss)
        return modelo.fit()

    def periodograma(self, best=3, plot=True, show=True):
        periodo = Periodograma(self.ts)
        print("\nAnálisis de Periodicidad:")
        print(f"Mejores frecuencias: {periodo.mejor_freq(best)}")
        print(f"Mejores períodos:    {periodo.mejor_periodos(best)}")
        if plot:
            periodo.plot_periodograma(best, show=show)
        return periodo

    @staticmethod
    def calcular_errores(predicciones, valores_reales, nombres=None):
        errores = ts_error(predicciones, valores_reales, nombres)
        df_errores = errores.df_errores()
        print("\nMétricas de Error:")
        print(df_errores)
        return errores

    def train_test_split(self, test_size=0.2):
        n_test = int(len(self.ts) * test_size)
        train = self.ts[:-n_test]
        test = self.ts[-n_test:]
        if hasattr(self.ts.index, 'freq') and self.ts.index.freq is not None:
            train.index.freq = self.ts.index.freq
            test.index.freq = self.ts.index.freq
        print(
            f"Train: {len(train)} observaciones, Test: {len(test)} observaciones")
        return train, test

    def arima(self, order=None, seasonal_order=(0, 0, 0, 0), trend=None,
              test=None, walk_forward=True):
        """ARIMA con auto-detect d + walk-forward."""
        if not STATSMODELS_AVAILABLE:
            print("statsmodels no disponible")
            return None
        if order is None:
            d = _auto_d(self.ts)
            order = (1, d, 1)
            print(f"ADF → d={d}, order={order}")
        else:
            d = order[1]
        if trend is None:
            trend = 'c' if d == 0 else 'n'
        try:
            modelo = SARIMAX(self.ts, order=order, seasonal_order=seasonal_order,
                             trend=trend, enforce_stationarity=True, enforce_invertibility=True)
            resultado = modelo.fit(disp=False, maxiter=500)
            if test is not None and walk_forward:
                pred_wf = _walk_forward_arima(
                    self.ts, test, order, seasonal_order)

                class _WFResult:
                    def __init__(self, p):
                        self._p = p
                        self.aic = resultado.aic

                    def forecast(self, steps=None):
                        return pd.Series(self._p)
                print(f"ARIMA{order} walk-forward")
                return _WFResult(pred_wf)
            print(f"ARIMA{order} AIC={resultado.aic:.2f}")
            return resultado
        except Exception as e:
            print(f"Error ARIMA: {e}")
            return None

    def arima_calibrado(self, test, p_values=None, d_values=None, q_values=None,
                        seasonal_order=(0, 0, 0, 0), walk_forward=True):
        """ARIMA calibrado con búsqueda por AIC + walk-forward."""
        if not STATSMODELS_AVAILABLE:
            print("statsmodels no disponible")
            return None, None
        if p_values is None:
            p_values = (0, 1, 2)
        if d_values is None:
            d_opt = _auto_d(self.ts)
            d_values = tuple(
                sorted(set([max(0, d_opt - 1), d_opt, min(2, d_opt + 1)])))
            print(f"ADF → d={d_opt}, búsqueda d={sorted(d_values)}")
        if q_values is None:
            q_values = (0, 1, 2)
        best_score = float("inf")
        best_order = None
        best_result = None
        print("Fase 1: Búsqueda por AIC...")
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    if p == 0 and q == 0:
                        continue
                    trend = 'c' if d == 0 else 'n'
                    try:
                        modelo = SARIMAX(self.ts, order=(p, d, q), seasonal_order=seasonal_order,
                                         trend=trend, enforce_stationarity=True, enforce_invertibility=True)
                        resultado = modelo.fit(disp=False, maxiter=200)
                        aic = resultado.aic
                        if np.isfinite(aic) and aic < best_score:
                            best_score = aic
                            best_order = (p, d, q)
                            best_result = resultado
                    except Exception:
                        continue
        if best_result is None:
            print("No se pudo ajustar ARIMA")
            return None, None
        print(f"Mejor orden: {best_order} (AIC={best_score:.2f})")
        if walk_forward:
            print(f"Fase 2: Walk-forward ({len(test)} pasos)...")
            pred_wf = _walk_forward_arima(
                self.ts, test, best_order, seasonal_order)
            if pred_wf is not None:
                class _WFResult:
                    def __init__(self, p, r):
                        self._p = p
                        self.aic = r.aic

                    def forecast(self, steps=None):
                        return pd.Series(self._p)
                return _WFResult(pred_wf, best_result), best_order
        return best_result, best_order

    def benchmark(self, test_size=0.2, incluir_lstm=False,
                  lstm_kwargs=None, hw_kwargs=None, hw_cal_kwargs=None,
                  arima_order=(1, 1, 1), arima_cal_params=None):
        """Ejecuta un benchmark entre los modelos de series de tiempo disponibles.

        Parámetros
        ----------
        test_size     : fracción del dataset para test (default=0.2)
        incluir_lstm  : incluir Red Neuronal LSTM (requiere TensorFlow, default=False)
        lstm_kwargs   : kwargs para SeriesTiempo.lstm()
        hw_kwargs     : kwargs para holt_winters()
        hw_cal_kwargs : kwargs para holt_winters_calibrado()
        arima_order   : orden (p,d,q) para ARIMA fijo (default=(1,1,1))
        arima_cal_params : dict con p_values/d_values/q_values para ARIMA calibrado

        Retorna
        -------
        DataFrame con métricas MSE, RMSE, RE, CORR por modelo
        """
        return self.benchmark_personalizado(
            test_size=test_size,
            incluir_hw=True, incluir_hw_cal=True,
            incluir_arima=True, incluir_arima_cal=True,
            incluir_lstm=incluir_lstm,
            lstm_kwargs=lstm_kwargs, hw_kwargs=hw_kwargs,
            hw_cal_kwargs=hw_cal_kwargs,
            arima_order=arima_order, arima_cal_params=arima_cal_params,
        )

    def benchmark_personalizado(self, test_size=0.2,
                                incluir_hw=True, incluir_hw_cal=True,
                                incluir_arima=True, incluir_arima_cal=True,
                                incluir_lstm=False,
                                lstm_kwargs=None, hw_kwargs=None, hw_cal_kwargs=None,
                                arima_order=(1, 1, 1), arima_cal_params=None):
        """Benchmark personalizado con selección de modelos a incluir."""
        train, test = self.train_test_split(test_size)
        nombres = []
        predicciones = []

        def _agregar(nombre, pred):
            """Valida y agrega predicción a la lista."""
            if pred is not None and len(pred) == len(test) and not pd.isna(pred).any():
                predicciones.append(pd.Series(pred.values, index=test.index))
                nombres.append(nombre)
            else:
                n = len(pred) if pred is not None else 'None'
                print(f"Advertencia {nombre}: predicción inválida (len={n})")

        # Holt-Winters
        if incluir_hw:
            try:
                hw_kwargs = hw_kwargs or {}
                m = SeriesTiempo(ts=train).holt_winters(**hw_kwargs)
                if m is not None:
                    _agregar("Holt-Winters", m.forecast(len(test)))
            except Exception as e:
                print(f"Advertencia Holt-Winters: {e}")

        # Holt-Winters calibrado
        if incluir_hw_cal:
            try:
                hw_cal_kwargs = hw_cal_kwargs or {}
                m = SeriesTiempo(ts=train).holt_winters_calibrado(
                    test, **hw_cal_kwargs)
                if m is not None:
                    _agregar("Holt-Winters Calibrado", m.forecast(len(test)))
            except Exception as e:
                print(f"Advertencia Holt-Winters calibrado: {e}")

        # ARIMA
        if incluir_arima:
            try:
                m = SeriesTiempo(ts=train).arima(order=arima_order)
                if m is not None:
                    _agregar("ARIMA", m.forecast(steps=len(test)))
            except Exception as e:
                print(f"Advertencia ARIMA: {e}")

        # ARIMA calibrado
        if incluir_arima_cal:
            try:
                arima_cal_params = arima_cal_params or {
                    'p_values': (0, 1), 'd_values': (0, 1), 'q_values': (0, 1)}
                m, _ = SeriesTiempo(ts=train).arima_calibrado(
                    test, **arima_cal_params)
                if m is not None:
                    _agregar("ARIMA Calibrado", m.forecast(steps=len(test)))
            except Exception as e:
                print(f"Advertencia ARIMA calibrado: {e}")

        # LSTM (opcional)
        if incluir_lstm:
            try:
                lstm_kwargs = lstm_kwargs or {
                    'lstm_units': 20, 'dense_units': 1}
                m = SeriesTiempo(ts=train).lstm(**lstm_kwargs)
                if m is not None:
                    _agregar("Red Neuronal", m.forecast(len(test)))
            except Exception as e:
                print(f"Advertencia LSTM: {e}")

        if not predicciones:
            print("No se pudieron generar predicciones para el benchmark.")
            return None

        errores = SeriesTiempo.calcular_errores(
            predicciones, test, nombres=nombres)
        df_resultados = errores.df_errores().reset_index().rename(
            columns={'index': 'Modelo'})
        return df_resultados


# ============================================================================
# CLASE WEB SCRAPING
# ============================================================================


class WebScraping:
    def __init__(self, headers=None):
        self.session = None
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        print("WebScraping inicializado")

    def iniciar_sesion(self):
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update(self.headers)
            print("Sesión iniciada")
            return self
        except ImportError:
            print("ERROR: requests no instalado. Instala: pip install requests")
            return None

    def obtener_html(self, url, timeout=10):
        try:
            import requests
            response = (self.session or requests).get(
                url, headers=self.headers if not self.session else {}, timeout=timeout)
            response.raise_for_status()
            print(f"HTML obtenido de: {url}")
            return response.text
        except ImportError:
            print("ERROR: requests no instalado")
            return None
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def parsear_html(self, html):
        try:
            from bs4 import BeautifulSoup
            print("HTML parseado")
            return BeautifulSoup(html, 'html.parser')
        except ImportError:
            print(
                "ERROR: beautifulsoup4 no instalado. Instala: pip install beautifulsoup4")
            return None

    def scrape_tabla_simple(self, url, selector='table', indice_tabla=0):
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None
        tablas = soup.find_all(selector)
        if not tablas or indice_tabla >= len(tablas):
            print("No se encontraron tablas válidas")
            return None
        tabla = tablas[indice_tabla]
        headers = [th.get_text(strip=True) for th in tabla.find(
            'thead').find_all('th')] if tabla.find('thead') else []
        tbody = tabla.find('tbody') or tabla
        rows = [[td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                for tr in tbody.find_all('tr') if tr.find_all(['td', 'th'])]
        df = pd.DataFrame(rows, columns=headers if headers else None)
        print(f"Tabla extraída: {df.shape}")
        return df

    def scrape_texto(self, url, selector):
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None
        textos = [elem.get_text(strip=True) for elem in soup.select(selector)]
        print(f"Extraídos {len(textos)} textos")
        return textos

    def scrape_enlaces(self, url, filtro=None):
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None
        enlaces = [link['href'] for link in soup.find_all('a', href=True)
                   if filtro is None or filtro in link['href']]
        print(f"Extraídos {len(enlaces)} enlaces")
        return enlaces

    def scrape_imagenes(self, url, atributo_src='src'):
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None
        imagenes = [img.get(atributo_src)
                    for img in soup.find_all('img') if img.get(atributo_src)]
        print(f"Extraídas {len(imagenes)} imágenes")
        return imagenes

    def scrape_multiples_paginas(self, urls, funcion_scraping, **kwargs):
        import time
        print(f"Scraping de {len(urls)} páginas...")
        resultados = []
        for i, url in enumerate(urls, 1):
            print(f"Procesando {i}/{len(urls)}: {url}")
            resultados.append(funcion_scraping(url, **kwargs))
            if i < len(urls):
                time.sleep(1)
        print("Scraping completado")
        return resultados

    def extraer_metadata(self, url):
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None
        metadata = {
            'titulo':      soup.find('title').get_text(strip=True) if soup.find('title') else None,
            'descripcion': soup.find('meta', attrs={'name': 'description'}).get('content') if soup.find('meta', attrs={'name': 'description'}) else None,
            'keywords':    soup.find('meta', attrs={'name': 'keywords'}).get('content') if soup.find('meta', attrs={'name': 'keywords'}) else None,
            'autor':       soup.find('meta', attrs={'name': 'author'}).get('content') if soup.find('meta', attrs={'name': 'author'}) else None,
        }
        print("Metadata extraída:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")
        return metadata

    def descargar_archivo(self, url, nombre_archivo=None, directorio='descargas'):
        try:
            import requests
            import os
            if not os.path.exists(directorio):
                os.makedirs(directorio)
            nombre_archivo = nombre_archivo or url.split('/')[-1]
            ruta = os.path.join(directorio, nombre_archivo)
            response = requests.get(url, headers=self.headers, stream=True)
            response.raise_for_status()
            with open(ruta, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Archivo descargado: {ruta}")
            return ruta
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def cerrar_sesion(self):
        if self.session:
            self.session.close()
            print("Sesión cerrada")
        self.session = None


# ============================================================================
# CLASE WEB MINING (extiende WebScraping con métodos de minería estructurada)
# ============================================================================


class WebMining(WebScraping):
    """Extensión de WebScraping orientada a minería web estructurada."""

    def __init__(self, headers=None):
        super().__init__(headers)
        print("WebMining inicializado")

    def scrape_productos(self, url, selector_productos,
                         selector_nombre=None, selector_precio=None,
                         selector_precio_descuento=None, selector_agotado=None):
        """
        Extrae una lista de productos de un sitio e-commerce.

        Parámetros
        ----------
        url                       : URL de la página
        selector_productos        : selector CSS del contenedor de cada producto
        selector_nombre           : selector CSS del nombre (dentro del producto)
        selector_precio           : selector CSS del precio original
        selector_precio_descuento : selector CSS del precio con descuento
        selector_agotado          : selector CSS del indicador de agotado

        Retorna
        -------
        DataFrame con columnas: Nombre, Precio, PrecioDescuento, Agotado
        """
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None

        contenedores = soup.select(selector_productos)
        print(f"Productos encontrados: {len(contenedores)}")

        registros = []
        for producto in contenedores:
            def _texto(sel):
                if sel is None:
                    return None
                elem = producto.select_one(sel)
                return elem.get_text(strip=True) if elem else None

            agotado = 'Sí' if selector_agotado and producto.select_one(selector_agotado) else 'No'
            registros.append({
                'Nombre':          _texto(selector_nombre),
                'Precio':          _texto(selector_precio),
                'PrecioDescuento': _texto(selector_precio_descuento),
                'Agotado':         agotado,
            })

        df = pd.DataFrame(registros)
        print(f"Tabla de productos: {df.shape}")
        return df

    def extraer_lista_estructurada(self, url, selector_padre, selectores_campos):
        """
        Extrae una lista de elementos con múltiples campos definidos por selectores.

        Parámetros
        ----------
        url               : URL de la página
        selector_padre    : selector CSS del contenedor de cada elemento
        selectores_campos : dict {nombre_campo: selector_CSS}

        Ejemplo
        -------
        extraer_lista_estructurada(url, '.item',
            {'titulo': 'h2.title', 'precio': '.price', 'rating': '.stars'})
        """
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None

        contenedores = soup.select(selector_padre)
        print(f"Elementos encontrados: {len(contenedores)}")

        registros = []
        for elem in contenedores:
            fila = {}
            for campo, selector in selectores_campos.items():
                nodo = elem.select_one(selector)
                fila[campo] = nodo.get_text(strip=True) if nodo else None
            registros.append(fila)

        df = pd.DataFrame(registros)
        print(f"Datos extraídos: {df.shape}")
        return df

    def extraer_atributos(self, url, selector, atributos):
        """
        Extrae atributos HTML (href, src, alt, …) de elementos seleccionados.

        Parámetros
        ----------
        url       : URL de la página
        selector  : selector CSS de los elementos
        atributos : lista de atributos a extraer, ej: ['href', 'src', 'alt']
        """
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None

        elementos = soup.select(selector)
        registros = []
        for elem in elementos:
            fila = {'texto': elem.get_text(strip=True)}
            for attr in atributos:
                fila[attr] = elem.get(attr)
            registros.append(fila)

        df = pd.DataFrame(registros)
        print(f"Atributos extraídos: {df.shape}")
        return df

    def consultar_api(self, url, headers=None, json_body=None, method='GET'):
        """Llama a un endpoint JSON y devuelve la respuesta como dict.

        Parámetros
        ----------
        url       : URL del endpoint
        headers   : dict de cabeceras HTTP
        json_body : dict del cuerpo (solo para method='POST')
        method    : 'GET' o 'POST'
        """
        import requests as _requests
        _hdrs = headers or {}
        try:
            if method.upper() == 'POST':
                resp = _requests.post(url, headers=_hdrs, json=json_body, timeout=20)
            else:
                resp = _requests.get(url, headers=_hdrs, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            print(f"API respuesta OK: {url}")
            return data
        except Exception as e:
            print(f"Error consultando API: {e}")
            return None

    def scrape_json_api(self, url, headers=None, json_body=None, method='POST',
                        campo_items='hits', campo_cursor='cursor'):
        """Descarga todos los registros de una API JSON con paginación por cursor.

        Parámetros
        ----------
        url           : endpoint de la API
        headers       : dict de cabeceras HTTP
        json_body     : dict base del cuerpo de cada petición
        method        : 'GET' o 'POST'
        campo_items   : clave del JSON que contiene la lista de registros
        campo_cursor  : clave del cursor de paginación (None = sin paginación)

        Retorna
        -------
        list de dicts con todos los registros concatenados de todas las páginas
        """
        import requests as _requests
        _hdrs = headers or {}
        todos, cursor, pagina = [], None, 1
        while True:
            body = dict(json_body or {})
            if campo_cursor and cursor:
                body[campo_cursor] = cursor
            try:
                if method.upper() == 'POST':
                    resp = _requests.post(url, headers=_hdrs, json=body, timeout=20)
                else:
                    resp = _requests.get(url, headers=_hdrs, params=body, timeout=20)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"Error en página {pagina}: {e}")
                break
            todos.extend(data.get(campo_items, []))
            print(f"  Página {pagina} — acumulado: {len(todos)} registros")
            cursor = data.get(campo_cursor) if campo_cursor else None
            if not cursor:
                break
            pagina += 1
        print(f"Total descargado: {len(todos)} registros")
        return todos

    def limpiar_texto(self, texto):
        """Colapsa espacios y saltos de línea en un texto."""
        import re
        if texto is None:
            return None
        return re.sub(r'\s+', ' ', texto).strip()

    def limpiar_precio(self, texto):
        """Extrae el primer valor numérico de un texto tipo '₡1,234.56' → 1234.56."""
        import re
        if texto is None:
            return None
        numeros = re.findall(r'[\d,\.]+', texto)
        if not numeros:
            return None
        try:
            return float(numeros[0].replace(',', ''))
        except ValueError:
            return None

    def exportar_csv(self, df, ruta, sep=',', decimal='.', index=False):
        """Guarda un DataFrame en disco como CSV."""
        df.to_csv(ruta, sep=sep, decimal=decimal, index=index)
        print(f"Exportado: {ruta} ({df.shape[0]} filas × {df.shape[1]} columnas)")
        return ruta

    def resumen_datos(self, df):
        """Imprime un resumen del DataFrame extraído."""
        print("\n" + "="*70)
        print("RESUMEN DE DATOS EXTRAÍDOS")
        print("="*70)
        print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
        print(f"Columnas: {list(df.columns)}")
        print(f"\nNulos:\n{df.isnull().sum()}")
        print(f"\nMuestra:\n{df.head()}")
        return df


# ============================================================================
# CLASE REGLAS DE ASOCIACION
# ============================================================================


class ReglasAsociacion:
    """Análisis de reglas de asociación mediante el algoritmo Apriori (mlxtend)."""

    def __init__(self, df=None, col_id='id_compra', col_item='item', matriz=None):
        """
        Parámetros
        ----------
        df      : DataFrame en formato largo con columnas [col_id, col_item]
        col_id  : nombre de la columna identificadora de transacción
        col_item: nombre de la columna de ítem
        matriz  : DataFrame binario ya codificado (alternativa a df)
        """
        if not MLXTEND_AVAILABLE:
            raise ImportError(
                "mlxtend no instalado. Instala: pip install mlxtend")

        self.col_id   = col_id
        self.col_item = col_item
        self.df       = None
        self.matriz   = None
        self.itemsets = None
        self.reglas   = None

        if matriz is not None:
            self.matriz = matriz.copy()
            print(f"Matriz binaria cargada: "
                  f"{matriz.shape[0]} transacciones × {matriz.shape[1]} ítems")
        elif df is not None:
            self.df = df.copy()
            print(f"Datos cargados: {df[col_id].nunique()} transacciones, "
                  f"{df[col_item].nunique()} ítems únicos")
        else:
            raise ValueError("Debe proporcionar 'df' o 'matriz'")

    def encodificar(self):
        """Convierte el DataFrame transaccional a matriz binaria con TransactionEncoder."""
        transacciones = (self.df.groupby(self.col_id)[self.col_item]
                         .apply(list).to_list())
        encoder = TransactionEncoder()
        encoded = encoder.fit(transacciones).transform(transacciones)
        self.matriz  = pd.DataFrame(encoded, columns=encoder.columns_)
        self.encoder = encoder
        print(f"Matriz codificada: "
              f"{self.matriz.shape[0]} transacciones × {self.matriz.shape[1]} ítems")
        return self

    def itemsets_frecuentes(self, min_support=0.01, max_len=None):
        """Calcula itemsets frecuentes con Apriori.

        Parámetros
        ----------
        min_support : soporte mínimo (fracción entre 0 y 1, o entero absoluto)
        max_len     : longitud máxima de los itemsets (None = sin límite)
        """
        if self.matriz is None:
            self.encodificar()

        soporte = (min_support if min_support <= 1.0
                   else min_support / self.matriz.shape[0])
        print(f"Apriori — soporte mínimo: {soporte:.4f}...")

        kwargs = {'min_support': soporte, 'use_colnames': True}
        if max_len:
            kwargs['max_len'] = max_len

        self.itemsets = apriori(self.matriz, **kwargs)
        self.itemsets['n_items'] = self.itemsets['itemsets'].apply(len)
        print(f"Itemsets encontrados: {len(self.itemsets)}")
        return self

    def generar_reglas(self, metric='confidence', min_threshold=0.5):
        """Genera reglas de asociación desde los itemsets frecuentes.

        Parámetros
        ----------
        metric        : métrica de filtro ('confidence', 'lift', 'support')
        min_threshold : umbral mínimo para la métrica elegida
        """
        if self.itemsets is None:
            self.itemsets_frecuentes()

        self.reglas = association_rules(
            self.itemsets, metric=metric, min_threshold=min_threshold)
        self.reglas['items'] = (
            self.reglas[['antecedents', 'consequents']]
            .apply(lambda x: set().union(*x), axis=1))

        print(f"Reglas generadas: {len(self.reglas)}")
        if len(self.reglas) > 0:
            print(f"  Confianza: [{self.reglas['confidence'].min():.3f}, "
                  f"{self.reglas['confidence'].max():.3f}]")
            print(f"  Lift:      [{self.reglas['lift'].min():.3f}, "
                  f"{self.reglas['lift'].max():.3f}]")
        return self

    def top_itemsets(self, n=10, n_items_min=1):
        """Retorna los N itemsets con mayor soporte."""
        if self.itemsets is None:
            raise ValueError("Ejecute primero itemsets_frecuentes()")
        return (self.itemsets[self.itemsets['n_items'] >= n_items_min]
                .sort_values('support', ascending=False)
                .head(n))

    def filtrar_itemsets_con(self, items):
        """Filtra itemsets que contienen todos los ítems indicados."""
        if self.itemsets is None:
            raise ValueError("Ejecute primero itemsets_frecuentes()")
        if isinstance(items, str):
            items = {items}
        items = set(items)
        mask = self.itemsets['itemsets'].map(lambda x: x.issuperset(items))
        return self.itemsets.loc[mask].sort_values('support', ascending=False)

    def filtrar_reglas_por_consecuente(self, item):
        """Filtra reglas cuyo consecuente contiene el ítem dado."""
        if self.reglas is None:
            raise ValueError("Ejecute primero generar_reglas()")
        mask = self.reglas['consequents'].map(lambda x: item in x)
        return self.reglas.loc[mask].sort_values('confidence', ascending=False)

    def recomendar(self, antecedente, top_n=5):
        """
        Dado un ítem o conjunto, retorna los consecuentes más probables.

        Parámetros
        ----------
        antecedente : str o set de ítems ya presentes en el carrito
        top_n       : cantidad máxima de recomendaciones
        """
        if self.reglas is None:
            raise ValueError("Ejecute primero generar_reglas()")
        if isinstance(antecedente, str):
            antecedente = {antecedente}
        antecedente = set(antecedente)

        mask = self.reglas['antecedents'].map(lambda x: antecedente.issubset(x))
        recomendaciones = (self.reglas.loc[mask]
                           .sort_values('confidence', ascending=False)
                           .head(top_n))

        print(f"\nRecomendaciones para {antecedente}:")
        if recomendaciones.empty:
            print("  No se encontraron reglas para ese antecedente")
        else:
            for _, row in recomendaciones.iterrows():
                cons = ', '.join(list(row['consequents']))
                print(f"  → {cons}  "
                      f"(conf={row['confidence']:.3f}, lift={row['lift']:.3f})")
        return recomendaciones

    def resumen(self):
        """Imprime un resumen completo del análisis."""
        print("\n" + "="*70)
        print("RESUMEN - REGLAS DE ASOCIACION")
        print("="*70)
        if self.matriz is not None:
            soporte_medio = self.matriz.mean(axis=0).mean()
            print(f"Transacciones : {self.matriz.shape[0]}")
            print(f"Ítems únicos  : {self.matriz.shape[1]}")
            print(f"Soporte medio : {soporte_medio:.4f}")
        if self.itemsets is not None:
            print(f"\nItemsets frecuentes: {len(self.itemsets)}")
            top = self.itemsets.nlargest(5, 'support')[['itemsets', 'support']]
            print(f"Top 5 por soporte:\n{top.to_string(index=False)}")
        if self.reglas is not None:
            print(f"\nReglas generadas : {len(self.reglas)}")
            print(f"Confianza media  : {self.reglas['confidence'].mean():.3f}")
            print(f"Lift medio       : {self.reglas['lift'].mean():.3f}")
        return self

    def grafico_soporte(self, figsize=(10, 4), show=True):
        """Histograma de distribución del soporte por ítem."""
        if self.matriz is None:
            raise ValueError("Ejecute encodificar() primero")
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        self.matriz.mean(axis=0).plot.hist(ax=ax, bins=30, color='steelblue', alpha=0.8)
        ax.set_title('Distribución del Soporte de los Ítems')
        ax.set_xlabel('Soporte')
        ax.set_ylabel('Frecuencia')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def grafico_top_items(self, n=15, figsize=(12, 5), show=True):
        """Gráfico de barras con los N ítems de mayor soporte."""
        if self.matriz is None:
            raise ValueError("Ejecute encodificar() primero")
        top = self.matriz.mean(axis=0).sort_values(ascending=False).head(n)
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        top.plot.bar(ax=ax, color='steelblue', alpha=0.85)
        ax.set_title(f'Top {n} Ítems por Soporte')
        ax.set_xlabel('Ítem')
        ax.set_ylabel('Soporte')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def grafico_reglas(self, figsize=(8, 6), show=True):
        """Dispersión Confianza vs Lift, coloreado por Soporte."""
        if self.reglas is None:
            raise ValueError("Ejecute generar_reglas() primero")
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        scatter = ax.scatter(
            self.reglas['confidence'], self.reglas['lift'],
            c=self.reglas['support'], cmap='viridis', alpha=0.7, s=60)
        plt.colorbar(scatter, ax=ax, label='Soporte')
        ax.set_xlabel('Confianza')
        ax.set_ylabel('Lift')
        ax.set_title('Reglas de Asociación: Confianza vs Lift')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig


# ============================================================================
# CLASE REDES NEURONALES (5 tipos)
# ============================================================================


class RedesNeuronales:
    """
    Comparación de 5 tipos de redes neuronales para clasificación o regresión.

    Tipos disponibles
    -----------------
    1. MLP sklearn          (MLPClassifier / MLPRegressor)
    2. Red Densa ReLU       (Keras Sequential, activación relu)
    3. Red Densa Tanh       (Keras Sequential, activación tanh)
    4. Red Convolucional 1D (Keras Conv1D — features como secuencia)
    5. Red Recurrente LSTM  (Keras LSTM)
    """

    def __init__(self, X_train, X_test, y_train, y_test, tarea='clasificacion'):
        """
        Parámetros
        ----------
        X_train, X_test : arrays o DataFrames de features
        y_train, y_test : arrays o Series de etiquetas / valores
        tarea           : 'clasificacion' o 'regresion'
        """
        self.X_train = np.array(X_train)
        self.X_test  = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test  = np.array(y_test)
        self.tarea   = tarea
        self._encoder  = None
        self._n_clases = None

        if tarea == 'clasificacion':
            from sklearn.preprocessing import LabelEncoder
            self._encoder      = LabelEncoder()
            self.y_train_enc   = self._encoder.fit_transform(y_train)
            self.y_test_enc    = self._encoder.transform(y_test)
            self._n_clases     = len(np.unique(self.y_train_enc))
        else:
            self.y_train_enc = self.y_train.astype(float)
            self.y_test_enc  = self.y_test.astype(float)

        print(f"RedesNeuronales — tarea: {tarea}")
        print(f"Train: {self.X_train.shape} | Test: {self.X_test.shape}")

    # ------------------------------------------------------------------
    # Métricas internas
    # ------------------------------------------------------------------
    def _calcular_metricas(self, y_true, y_pred):
        if self.tarea == 'clasificacion':
            MC = confusion_matrix(y_true, y_pred)
            pg = np.sum(MC.diagonal()) / np.sum(MC)
            return {'Matriz de Confusión': MC,
                    'Precisión Global': pg,
                    'Error Global':     1 - pg}
        else:
            mse = mean_squared_error(y_true, y_pred)
            return {'RMSE': np.sqrt(mse),
                    'MAE':  mean_absolute_error(y_true, y_pred),
                    'R²':   r2_score(y_true, y_pred)}

    def _imprimir_metricas(self, nombre, metricas):
        print(f"\n{'='*60}")
        print(f"  {nombre}")
        print(f"{'='*60}")
        for k, v in metricas.items():
            print(f"  {k}:\n{v}")

    def _escalar(self, X_tr, X_te):
        sc = StandardScaler()
        return sc.fit_transform(X_tr), sc.transform(X_te)

    # ------------------------------------------------------------------
    # Tipo 1: MLP sklearn
    # ------------------------------------------------------------------
    def red_mlp_sklearn(self, capas=(100, 50), activation='relu',
                        solver='adam', max_iter=500, scale=True):
        """Tipo 1 — MLP con scikit-learn (MLPClassifier / MLPRegressor)."""
        if not MLP_SKLEARN_AVAILABLE:
            print("sklearn.neural_network no disponible")
            return None, None

        print(f"\nTipo 1: MLP sklearn  (capas={capas}, "
              f"activation={activation}, solver={solver})")
        X_tr, X_te = (self._escalar(self.X_train, self.X_test)
                      if scale else (self.X_train, self.X_test))

        ModelClass = MLPClassifier if self.tarea == 'clasificacion' else MLPRegressor
        modelo = ModelClass(hidden_layer_sizes=capas, activation=activation,
                            solver=solver, max_iter=max_iter, random_state=42)
        modelo.fit(X_tr, self.y_train)
        y_pred = modelo.predict(X_te)

        metricas = self._calcular_metricas(self.y_test, y_pred)
        self._imprimir_metricas(f"MLP sklearn ({activation}+{solver})", metricas)
        return modelo, metricas

    # ------------------------------------------------------------------
    # Tipo 2: Red Densa ReLU (Keras)
    # ------------------------------------------------------------------
    def red_densa_relu(self, capas=(64, 32), epochs=100,
                       batch_size=32, scale=True):
        """Tipo 2 — Red densa Keras con activación ReLU."""
        return self._red_densa_keras(capas, 'relu', epochs, batch_size, scale,
                                     nombre='Red Densa ReLU (Tipo 2)')

    # ------------------------------------------------------------------
    # Tipo 3: Red Densa Tanh (Keras)
    # ------------------------------------------------------------------
    def red_densa_tanh(self, capas=(64, 32), epochs=100,
                       batch_size=32, scale=True):
        """Tipo 3 — Red densa Keras con activación Tanh."""
        return self._red_densa_keras(capas, 'tanh', epochs, batch_size, scale,
                                     nombre='Red Densa Tanh (Tipo 3)')

    def _red_densa_keras(self, capas, activacion, epochs, batch_size, scale, nombre):
        if not KERAS_NN_AVAILABLE:
            print("Keras/TensorFlow no disponible. Instala: pip install tensorflow")
            return None, None
        from keras.models import Sequential as _KerasSequential
        from keras.layers import Dense as _Dense

        print(f"\n{nombre}")
        X_tr, X_te = (self._escalar(self.X_train, self.X_test)
                      if scale else (self.X_train, self.X_test))

        modelo = _KerasSequential()
        for i, u in enumerate(capas):
            if i == 0:
                modelo.add(_Dense(u, activation=activacion,
                                  input_shape=(X_tr.shape[1],)))
            else:
                modelo.add(_Dense(u, activation=activacion))

        y_pred = self._compilar_ajustar_predecir(
            modelo, X_tr, X_te, epochs, batch_size)
        if y_pred is None:
            return None, None

        metricas = self._calcular_metricas(self.y_test, y_pred)
        self._imprimir_metricas(nombre, metricas)
        return modelo, metricas

    # ------------------------------------------------------------------
    # Tipo 4: Red Convolucional 1D (Keras)
    # ------------------------------------------------------------------
    def red_cnn_1d(self, filters=64, kernel_size=3, epochs=50,
                   batch_size=32, scale=True):
        """Tipo 4 — CNN 1D: trata cada fila como secuencia de features."""
        if not KERAS_NN_AVAILABLE:
            print("Keras/TensorFlow no disponible. Instala: pip install tensorflow")
            return None, None
        from keras.models import Sequential as _KerasSequential
        from keras.layers import Dense as _Dense, Conv1D as _Conv1D
        from keras.layers import MaxPooling1D as _MaxPooling1D, Flatten as _Flatten

        print(f"\nTipo 4: Red CNN 1D  (filters={filters}, kernel={kernel_size})")
        X_tr, X_te = (self._escalar(self.X_train, self.X_test)
                      if scale else (self.X_train, self.X_test))

        n_feat = X_tr.shape[1]
        X_tr_r = X_tr.reshape(X_tr.shape[0], n_feat, 1)
        X_te_r = X_te.reshape(X_te.shape[0], n_feat, 1)

        ks = min(kernel_size, n_feat)
        modelo = _KerasSequential([
            _Conv1D(filters=filters, kernel_size=ks, activation='relu',
                    input_shape=(n_feat, 1)),
            _MaxPooling1D(pool_size=2, padding='same'),
            _Flatten(),
            _Dense(64, activation='relu'),
        ])

        y_pred = self._compilar_ajustar_predecir(
            modelo, X_tr_r, X_te_r, epochs, batch_size)
        if y_pred is None:
            return None, None

        metricas = self._calcular_metricas(self.y_test, y_pred)
        self._imprimir_metricas("Red CNN 1D", metricas)
        return modelo, metricas

    # ------------------------------------------------------------------
    # Tipo 5: Red LSTM (Keras)
    # ------------------------------------------------------------------
    def red_lstm(self, units=64, epochs=50, batch_size=32, scale=True):
        """Tipo 5 — LSTM: trata cada fila como secuencia temporal."""
        if not KERAS_NN_AVAILABLE:
            print("Keras/TensorFlow no disponible. Instala: pip install tensorflow")
            return None, None
        from keras.models import Sequential as _KerasSequential
        from keras.layers import Dense as _Dense, LSTM as _KerasLSTM

        print(f"\nTipo 5: Red LSTM  (units={units})")
        X_tr, X_te = (self._escalar(self.X_train, self.X_test)
                      if scale else (self.X_train, self.X_test))

        n_feat = X_tr.shape[1]
        X_tr_r = X_tr.reshape(X_tr.shape[0], n_feat, 1)
        X_te_r = X_te.reshape(X_te.shape[0], n_feat, 1)

        modelo = _KerasSequential([
            _KerasLSTM(units=units, input_shape=(n_feat, 1)),
            _Dense(32, activation='relu'),
        ])

        y_pred = self._compilar_ajustar_predecir(
            modelo, X_tr_r, X_te_r, epochs, batch_size)
        if y_pred is None:
            return None, None

        metricas = self._calcular_metricas(self.y_test, y_pred)
        self._imprimir_metricas("Red LSTM", metricas)
        return modelo, metricas

    # ------------------------------------------------------------------
    # Helper: compila, ajusta y predice según la tarea
    # ------------------------------------------------------------------
    def _compilar_ajustar_predecir(self, modelo, X_tr, X_te, epochs, batch_size):
        from keras.layers import Dense as _Dense
        try:
            if self.tarea == 'clasificacion':
                if self._n_clases == 2:
                    modelo.add(_Dense(1, activation='sigmoid'))
                    modelo.compile(optimizer='adam',
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])
                    modelo.fit(X_tr, self.y_train_enc,
                               epochs=epochs, batch_size=batch_size, verbose=0)
                    pred_raw   = modelo.predict(X_te, verbose=0)
                    y_pred_enc = (pred_raw > 0.5).astype(int).ravel()
                else:
                    modelo.add(_Dense(self._n_clases, activation='softmax'))
                    modelo.compile(optimizer='adam',
                                   loss='sparse_categorical_crossentropy',
                                   metrics=['accuracy'])
                    modelo.fit(X_tr, self.y_train_enc,
                               epochs=epochs, batch_size=batch_size, verbose=0)
                    pred_raw   = modelo.predict(X_te, verbose=0)
                    y_pred_enc = np.argmax(pred_raw, axis=1)
                return self._encoder.inverse_transform(y_pred_enc)
            else:
                modelo.add(_Dense(1, activation='linear'))
                modelo.compile(optimizer='adam', loss='mse')
                modelo.fit(X_tr, self.y_train_enc,
                           epochs=epochs, batch_size=batch_size, verbose=0)
                return modelo.predict(X_te, verbose=0).ravel()
        except Exception as e:
            print(f"Error al entrenar: {e}")
            return None

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    def benchmark(self, capas=(64, 32), epochs=50, batch_size=32, scale=True):
        """Ejecuta los 5 tipos de redes y retorna tabla comparativa."""
        print("\n" + "="*70)
        print("BENCHMARK — 5 TIPOS DE REDES NEURONALES")
        print("="*70)

        configs = [
            ('MLP sklearn (relu+adam)',
             lambda: self.red_mlp_sklearn(capas=capas, scale=scale)),
            ('Red Densa ReLU (Keras)',
             lambda: self.red_densa_relu(capas=capas, epochs=epochs,
                                         batch_size=batch_size, scale=scale)),
            ('Red Densa Tanh (Keras)',
             lambda: self.red_densa_tanh(capas=capas, epochs=epochs,
                                         batch_size=batch_size, scale=scale)),
            ('Red CNN 1D (Keras)',
             lambda: self.red_cnn_1d(epochs=epochs,
                                     batch_size=batch_size, scale=scale)),
            ('Red LSTM (Keras)',
             lambda: self.red_lstm(epochs=epochs,
                                   batch_size=batch_size, scale=scale)),
        ]

        resultados = []
        for nombre, func in configs:
            try:
                _, metricas = func()
                if metricas:
                    fila = {'Modelo': nombre}
                    if self.tarea == 'clasificacion':
                        fila['Precisión Global'] = metricas.get('Precisión Global', float('nan'))
                        fila['Error Global']     = metricas.get('Error Global',     float('nan'))
                    else:
                        fila['RMSE'] = metricas.get('RMSE', float('nan'))
                        fila['MAE']  = metricas.get('MAE',  float('nan'))
                        fila['R²']   = metricas.get('R²',   float('nan'))
                    resultados.append(fila)
            except Exception as e:
                print(f"Error en {nombre}: {e}")

        df_res = pd.DataFrame(resultados)
        sort_col  = 'Precisión Global' if self.tarea == 'clasificacion' else 'RMSE'
        ascending = self.tarea != 'clasificacion'
        if sort_col in df_res.columns:
            df_res = df_res.sort_values(sort_col, ascending=ascending)

        print(f"\n{'='*70}")
        print("RESULTADOS DEL BENCHMARK")
        print(f"{'='*70}")
        print(df_res.to_string(index=False))
        return df_res

    def grafico_benchmark(self, df_benchmark, figsize=(12, 5), show=True):
        """Gráfico de barras del benchmark."""
        metric = 'Precisión Global' if self.tarea == 'clasificacion' else 'RMSE'
        if metric not in df_benchmark.columns:
            print(f"Columna '{metric}' no encontrada en el benchmark")
            return None

        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        colores = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'goldenrod']
        bars = ax.bar(df_benchmark['Modelo'], df_benchmark[metric],
                      color=colores[:len(df_benchmark)], alpha=0.85)
        ax.set_title(f'Benchmark Redes Neuronales — {metric}')
        ax.set_xlabel('Modelo')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=20)
        for bar, val in zip(bars, df_benchmark[metric]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        if show:
            plt.show()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Métodos de regex para WebMining (monkey-patch)
# ─────────────────────────────────────────────────────────────────────────────

def _wm_scrape_texto_multiple(self, url, config_etiquetas):
    """Descarga la página y extrae texto de múltiples grupos de etiquetas.

    config_etiquetas: dict  {nombre_col: (tag, attrs)}
    Retorna DataFrame con una columna por cada grupo.
    """
    import requests as _requests
    from bs4 import BeautifulSoup as _BS
    try:
        resp = _requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.encoding = resp.apparent_encoding
        soup = _BS(resp.text, "html.parser")
    except Exception as e:
        print(f"Error al descargar {url}: {e}")
        return pd.DataFrame()

    max_len = 0
    cols = {}
    for nombre, (tag, attrs) in config_etiquetas.items():
        elementos = soup.find_all(tag, attrs) if attrs else soup.find_all(tag)
        textos = [el.get_text(strip=True) for el in elementos]
        cols[nombre] = textos
        if len(textos) > max_len:
            max_len = len(textos)

    for k in cols:
        while len(cols[k]) < max_len:
            cols[k].append(None)

    return pd.DataFrame(cols)


def _wm_filtrar_con_regex(self, lista, patron):
    """Devuelve solo los elementos de *lista* que contienen el patrón regex."""
    import re as _re
    return [item for item in lista if item and _re.search(patron, item)]


def _wm_extraer_grupos_regex(self, lista, patrones, limpiar_espacios=True):
    """Aplica múltiples patrones a cada elemento y construye un DataFrame.

    patrones: dict  {nombre_col: patron_regex_con_un_grupo}
    Solo incluye filas donde TODOS los patrones producen una coincidencia.
    """
    import re as _re
    filas = []
    for item in lista:
        if not item:
            continue
        fila = {}
        valido = True
        for col, pat in patrones.items():
            m = _re.search(pat, item)
            if m:
                val = m.group(1)
                if limpiar_espacios:
                    val = val.strip()
                fila[col] = val
            else:
                valido = False
                break
        if valido:
            filas.append(fila)
    return pd.DataFrame(filas)


def _wm_limpiar_columnas_regex(self, df, columnas, sustituciones):
    """Aplica re.sub en masa a columnas de un DataFrame.

    sustituciones: list de (patron, reemplazo)
    """
    import re as _re
    df = df.copy()
    for col in columnas:
        if col not in df.columns:
            continue
        for patron, reemplazo in sustituciones:
            df[col] = df[col].astype(str).apply(
                lambda x: _re.sub(patron, reemplazo, x))
    return df


def _wm_grafico_top_productos(self, df, col_nombre, col_valor, n=10,
                               titulo="Top productos", figsize=(10, 6),
                               color="steelblue", show=True):
    """Gráfico horizontal de barras para los n productos más caros/frecuentes."""
    df_top = df.nlargest(n, col_valor)[[col_nombre, col_valor]].copy()
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.barh(df_top[col_nombre].astype(str), df_top[col_valor], color=color, alpha=0.85)
    ax.set_title(titulo)
    ax.set_xlabel(col_valor)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# Inyectar métodos en WebMining
WebMining.scrape_texto_multiple  = _wm_scrape_texto_multiple
WebMining.filtrar_con_regex      = _wm_filtrar_con_regex
WebMining.extraer_grupos_regex   = _wm_extraer_grupos_regex
WebMining.limpiar_columnas_regex = _wm_limpiar_columnas_regex
WebMining.grafico_top_productos  = _wm_grafico_top_productos


# ─────────────────────────────────────────────────────────────────────────────
# WebMiningSelenium — scraping de páginas con JavaScript
# ─────────────────────────────────────────────────────────────────────────────

class WebMiningSelenium:
    """Scraping de sitios con contenido generado por JavaScript usando Selenium.

    Requiere: selenium, un WebDriver compatible (geckodriver para Firefox
    o chromedriver para Chrome) en el PATH del sistema.
    """

    def __init__(self, browser='firefox', headless=False):
        if not SELENIUM_AVAILABLE:
            raise ImportError("selenium no está instalado. Ejecuta: pip install selenium")
        opts_firefox = None
        opts_chrome  = None
        browser = browser.lower()
        if browser == 'firefox':
            from selenium.webdriver.firefox.options import Options as _FFOpts
            opts_firefox = _FFOpts()
            if headless:
                opts_firefox.add_argument("--headless")
            self.driver = _selenium_webdriver.Firefox(options=opts_firefox)
        elif browser in ('chrome', 'chromium'):
            from selenium.webdriver.chrome.options import Options as _CROpts
            opts_chrome = _CROpts()
            if headless:
                opts_chrome.add_argument("--headless")
                opts_chrome.add_argument("--no-sandbox")
                opts_chrome.add_argument("--disable-dev-shm-usage")
            self.driver = _selenium_webdriver.Chrome(options=opts_chrome)
        else:
            raise ValueError(f"Browser no soportado: {browser}. Usa 'firefox' o 'chrome'.")
        self._browser_name = browser

    # ------------------------------------------------------------------
    # Navegación
    # ------------------------------------------------------------------

    def abrir_pagina(self, url, espera=2):
        """Carga *url* y espera *espera* segundos para que el JS se ejecute."""
        import time as _time
        self.driver.get(url)
        _time.sleep(espera)

    def obtener_html_actual(self):
        """Retorna objeto BeautifulSoup del HTML actual del navegador."""
        from bs4 import BeautifulSoup as _BS
        return _BS(self.driver.page_source, "html.parser")

    # ------------------------------------------------------------------
    # Búsquedas XPath
    # ------------------------------------------------------------------

    def buscar_xpath(self, xpath):
        """Devuelve lista de WebElements que coinciden con *xpath*."""
        return self.driver.find_elements(_By.XPATH, xpath)

    def obtener_atributos_xpath(self, xpath, atributo="innerHTML"):
        """Extrae *atributo* de todos los elementos que coinciden con *xpath*."""
        elementos = self.buscar_xpath(xpath)
        return [el.get_attribute(atributo) for el in elementos]

    def obtener_texto_xpath(self, xpath):
        """Extrae .text de todos los elementos que coinciden con *xpath*."""
        elementos = self.buscar_xpath(xpath)
        return [el.text.strip() for el in elementos]

    # ------------------------------------------------------------------
    # Scraping de productos en una sola página
    # ------------------------------------------------------------------

    def scrape_productos_js(self, url, xpath_nombres, xpath_precios,
                             xpath_descuentos=None, espera=2):
        """Extrae nombres, precios y descuentos de una página con JS.

        Retorna DataFrame con columnas: Nombre, Precio, Descuento (si aplica).
        """
        self.abrir_pagina(url, espera)
        nombres   = self.obtener_texto_xpath(xpath_nombres)
        precios   = self.obtener_texto_xpath(xpath_precios)
        max_len   = max(len(nombres), len(precios))
        while len(nombres) < max_len:
            nombres.append(None)
        while len(precios) < max_len:
            precios.append(None)
        data = {"Nombre": nombres, "Precio": precios}
        if xpath_descuentos:
            desc = self.obtener_texto_xpath(xpath_descuentos)
            while len(desc) < max_len:
                desc.append(None)
            data["Descuento"] = desc
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Paginación
    # ------------------------------------------------------------------

    def generar_urls_paginacion(self, base_url, total_paginas,
                                 patron="{page}", inicio=1):
        """Genera lista de URLs reemplazando *patron* con el número de página."""
        return [base_url.replace(patron, str(i))
                for i in range(inicio, inicio + total_paginas)]

    def detectar_total_paginas(self, xpath_paginacion, atributo="href"):
        """Intenta detectar el número total de páginas desde un elemento de paginación.

        Devuelve el máximo entero encontrado en los atributos de los elementos
        que coinciden con *xpath_paginacion*, o None si no puede determinarlo.
        """
        import re as _re
        valores = self.obtener_atributos_xpath(xpath_paginacion, atributo)
        numeros = []
        for v in valores:
            if v:
                encontrados = _re.findall(r'\d+', v)
                numeros.extend([int(n) for n in encontrados])
        return max(numeros) if numeros else None

    def scrape_multiples_paginas_js(self, urls, xpath_nombres, xpath_precios,
                                     xpath_descuentos=None, espera=2,
                                     verbose=True):
        """Itera *urls* y concatena los DataFrames de cada página."""
        dfs = []
        for i, url in enumerate(urls, 1):
            if verbose:
                print(f"  Página {i}/{len(urls)}: {url}")
            try:
                df_p = self.scrape_productos_js(
                    url, xpath_nombres, xpath_precios, xpath_descuentos, espera)
                df_p["Pagina"] = i
                dfs.append(df_p)
            except Exception as e:
                print(f"  Error en página {i}: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # ------------------------------------------------------------------
    # Limpieza y visualización
    # ------------------------------------------------------------------

    def limpiar_precios(self, df, columnas, sustituciones=None):
        """Aplica re.sub a *columnas* y convierte a numérico.

        sustituciones: list de (patron, reemplazo). Por defecto elimina $ y comas.
        """
        import re as _re
        if sustituciones is None:
            sustituciones = [('[\\$,]', ''), ('\\s+', '')]
        df = df.copy()
        for col in columnas:
            if col not in df.columns:
                continue
            for pat, rep in sustituciones:
                df[col] = df[col].astype(str).apply(lambda x: _re.sub(pat, rep, x))
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def grafico_top_caros(self, df, col_nombre, col_precio, n=10,
                           titulo="Top productos más caros",
                           figsize=(10, 6), color="steelblue", show=True):
        """Gráfico horizontal de barras con los n productos más caros."""
        df_top = df.nlargest(n, col_precio)[[col_nombre, col_precio]].copy()
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        ax.barh(df_top[col_nombre].astype(str), df_top[col_precio],
                color=color, alpha=0.85)
        ax.set_title(titulo)
        ax.set_xlabel("Precio")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def grafico_frecuencia(self, serie, titulo="Frecuencia", n=20,
                            figsize=(12, 5), color="coral", show=True):
        """Gráfico de barras con las n categorías más frecuentes de *serie*."""
        conteo = serie.value_counts().head(n)
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        conteo.plot(kind='bar', ax=ax, color=color, alpha=0.85)
        ax.set_title(titulo)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Cierre
    # ------------------------------------------------------------------

    def cerrar(self):
        """Cierra el navegador controlado por Selenium."""
        try:
            self.driver.quit()
        except Exception:
            pass

    def __del__(self):
        self.cerrar()
