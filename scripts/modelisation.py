import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


class Modelisation:
    def __init__(self):
        pass

    ###### ARIMA ######
    def modelisation_arima(self, df, horizons):
        """
        Effectue une modélisation ARIMA sur les ventes, en maintenant la même structure que modelisation_xgboost.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame contenant les colonnes 'Order Date' et 'Sales'.
        horizons : list of int
            Liste des horizons de prédiction.

        Returns
        -------
        None
            Les graphiques sont affichés directement.
        """
        df = df.sort_values(by="Order Date")
        train = df[df["Order Date"] < "2017-01-01"]
        test = df[df["Order Date"] >= "2017-01-01"]

        y_train = train["Sales"]
        y_test = test["Sales"]
        arima_order = (1, 1, 1)  # (p, d, q)
        predictions = {h: self.rolling_window_arima(arima_order, y_train, h) for h in horizons}
        
        rmse_values = {h: np.sqrt(mean_squared_error(y_test[:h], predictions[h])) for h in horizons}
        for h, rmse in rmse_values.items():
            print(f"RMSE de ARIMA pour un forecast à {h} jours: {rmse:.2f}")
        self.plot_forecasts_for_arima(y_test, predictions, horizons, df, rmse_values)

    def rolling_window_arima(self, order, train_series, horizon):

        """
        Effectue une prédiction sur une série temporelle avec un modèle ARIMA,
        en actualisant la série temporelle avec les prédictions.

        Parameters
        ----------
        order : tuple of int
            (p, d, q) pour le modèle ARIMA.
        train_series : pandas Series
            Série temporelle utilisée pour l'apprentissage.
        horizon : int
            Nombre de jours à prédire.

        Returns
        -------
        numpy array
            Tableau des prédictions faites par le modèle.
        """
        predictions = []
        history = list(train_series)  # Liste des valeurs connues pour mise à jour
        
        for _ in range(horizon):
            model_fit = ARIMA(history, order=order).fit()
            pred = model_fit.forecast()[0]
            predictions.append(pred)
            history.append(pred)

        return np.array(predictions)

    ####### XGBoost #######
    def modelisation_xgboost(self, df, horizons):
        """
        Effectue une modélisation XGBoost sur les ventes, en maintenant la même structure que modelisation_arima.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame contenant les colonnes 'Order Date' et 'Sales'.
        horizons : list of int
            Liste des horizons de prédiction.

        Returns
        -------
        None
            Les graphiques sont affichés directement.
        """
        df = df.sort_values(by="Order Date")
        train = df[df["Order Date"] < "2017-01-01"].drop(columns=["Order Date"])
        test = df[df["Order Date"] >= "2017-01-01"].drop(columns=["Order Date"])

        X_train = train.drop(columns=["Sales"])
        y_train = train["Sales"]
        X_test = test.drop(columns=["Sales"])
        y_test = test["Sales"]

        params = {'n_estimators': 282,
                  'max_depth': 3,
                  'learning_rate': 0.053147255012821776,
                  'subsample': 0.9149696049647875,
                  'colsample_bytree': 0.9261206654896488,
                  'alpha': 0.13092999736504427,
                  'lambda': 3.3872295257456323e-06,
                  'gamma': 0.00012319121922571826}

        xgb_model = xgb.XGBRegressor(**params)
        xgb_model.fit(X_train, y_train)
        predictions = {h: self.rolling_window_xgboost(xgb_model, X_test, h) for h in horizons}
        rmse_values = {h: np.sqrt(mean_squared_error(y_test[:h], predictions[h][:h])) for h in horizons}

        for h, rmse in rmse_values.items():
            print(f"RMSE du XGBoost pour un horizon de {h} jours: {rmse:.2f}")
        self.plot_forecasts_for_xgboost(y_test, predictions, horizons, df, rmse_values)

        self.plot_feature_importance(xgb_model, X_train.columns)


    def rolling_window_xgboost(self, model, X_test, horizon):

        """
        Effectue une prédiction sur une série temporelle avec un modèle XGBoost,
        en utilisant une fenêtre mobile pour générer des prédictions successives.

        Parameters
        ----------
        model : xgboost.XGBRegressor
            Modèle XGBoost entraîné utilisé pour effectuer les prédictions.
        X_test : pandas DataFrame
            Données de test utilisées pour générer les prédictions.
        horizon : int
            Nombre de jours à prédire.

        Returns
        -------
        numpy array
            Tableau des prédictions faites par le modèle.
        """
        predictions = []
        X_temp = X_test.copy()

        for i in range(horizon):
            pred = model.predict(X_temp.iloc[[i]])[0]
            predictions.append(pred)

        return np.array(predictions)


    ### plots des forecasts ARIMA et XGBoost ###

    def plot_forecasts_for_arima(self, y_test, predictions, horizons, df, rsme_values):
        """
        Trace des graphiques des prédictions faites par le modèle ARIMA.

        Les graphiques comprennent:
        - Les ventes réelles en vert
        - Les prédictions ARIMA en rouge

        Parameters
        ----------
        y_test : numpy array
            Ventes réelles à comparer aux prédictions.
        predictions : dict
            Dictionnaire des prédictions faites par le modèle ARIMA,
            avec les clés correspondant aux horizons de prédiction.
        horizons : list of int
            Liste des horizons de prédiction.
        df : pandas DataFrame
            DataFrame contenant les colonnes 'Order Date' et 'Sales'.
        rsme_values : dict
            Dictionnaire des valeurs de RMSE pour chaque horizon de prédiction.
        """
        fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 5))

        if len(horizons) == 1:
            axes = [axes]

        for ax, horizon in zip(axes, horizons):
            dates_test = df[df["Order Date"] >= "2017-01-01"]["Order Date"].values[:horizon]
            ax.plot(dates_test, y_test[:horizon], label="Ventes réelles", linestyle="-", color="green")
            ax.plot(dates_test, predictions[horizon], label=f"Prédictions ARIMA ({horizon} jours)", linestyle="--", color="red")
            ax.legend()
            rmse = rsme_values[horizon]
            ax.set_title(f"Prédictions à {horizon} jours (RMSE: {rmse:.2f})")
            ax.grid(True)
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_forecasts_for_xgboost(self, y_test, predictions, horizons, df, rmse_values):
        """
        Trace des graphiques des prédictions faites par le modèle XGBoost.

        Les graphiques comprennent:
        - Les ventes réelles en vert
        - Les prédictions XGBoost en rouge

        Parameters
        ----------
        y_test : numpy array
            Ventes réelles à comparer aux prédictions.
        predictions : dict
            Dictionnaire des prédictions faites par le modèle XGBoost,
            avec les clés correspondant aux horizons de prédiction.
        horizons : list of int
            Liste des horizons de prédiction.
        df : pandas DataFrame
            DataFrame contenant les colonnes 'Order Date' et 'Sales'.
        rsme_values : dict
            Dictionnaire des valeurs de RMSE pour chaque horizon de prédiction.
        """
        fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 5))
        if len(horizons) == 1:
            axes = [axes]

        for ax, horizon in zip(axes, horizons):
            dates_test = df[df["Order Date"] >= "2017-01-01"]["Order Date"].values[:horizon]
            ax.plot(dates_test, y_test[:horizon], label="Ventes réelles", linestyle="-", color="green")
            ax.plot(dates_test, predictions[horizon][:horizon], label=f"Prédictions ({horizon} jours)", linestyle="--", color="red")
            rmse = rmse_values[horizon]
            ax.set_title(f"Prédictions à {horizon} jours (RMSE: {rmse:.2f})")
            ax.legend()
            ax.grid(True)
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, model, feature_names):
        """
        Affiche l'importance des features pour le modèle XGBoost.

        Parameters
        ----------
        model : xgboost.XGBRegressor
            Modèle XGBoost entraîné.
        feature_names : list
            Liste des noms des features.

        Returns
        -------
        None
        """
        plt.figure(figsize=(10, 6))
        sorted_idx = model.feature_importances_.argsort()
        plt.barh([feature_names[i] for i in sorted_idx], model.feature_importances_[sorted_idx])
        plt.xlabel("Importance des features")
        plt.ylabel("Features")
        plt.title("Importance des features pour le modèle XGBoost")
        plt.show()
