import pandas as pd


class FeatureEngineering:
    def __init__(self):
        pass

    def get_feature_engineering(self, df, lag):
        """
        Effectue l'ensemble des opérations de feature engineering sur un DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame contenant les données de base.
        lag : int
            Nombre de jours pour lequel on ajoute un lag sur les ventes.

        Returns
        -------
        pandas DataFrame
            DataFrame contenant les nouvelles colonnes de feature engineering.
        """
        print("création des featues liées à la date...")
        df = self.create_time_features(df)
        print("Creation de la variable Season...")
        df["Order Season"] = df["Order Month"].apply(self.get_season)
        print("Création de la variable Shipping Delay...")
        df = self.get_shipping_delay(df)
        print("one hot encoding des variables catégorielles...")
        df = self.one_hot_encoding(df)
        print("leave one out target encoding...")
        df = self.leave_one_out_target_encoding(df, "State", "Sales")
        print(f"ajout de lag de {lag} jours sur les Sales...")
        df = self.add_lag_to_sales(df, lag)
        print("feature engineering terminé ✅")
        return df

    def get_shipping_delay(self, df):
        """
        Crée une colonne "Shipping Delay" qui correspond au nombre de jours entre la date de livraison et la date de commande.

        :param df: DataFrame contenant les colonnes "Order Date" et "Ship Date".
        :return: DataFrame avec la colonne "Shipping Delay" ajoutée.
        """
        df["Shipping Delay"] = (df["Ship Date"] - df["Order Date"]).dt.days
        df.drop(columns=["Ship Date"], axis=1,inplace=True)
        return df

    def one_hot_encoding(self, df):
        """
        Effectue un One-Hot Encoding sur les variables catégorielles à faible cardinalité avec 1/0.

        :param df: DataFrame contenant des variables catégorielles.
        :param threshold: Seuil de cardinalité maximale pour appliquer le One-Hot Encoding (par défaut 10).
        :return: DataFrame avec les variables encodées.
        """
        cal_cols = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() <= 12]

        df_encoded = pd.get_dummies(df, columns=cal_cols, drop_first=True)

        # Convertir uniquement les nouvelles colonnes encodées en 1/0
        encoded_cols = [col for col in df_encoded.columns if col not in df.columns or col in cal_cols]
        df_encoded[encoded_cols] = df_encoded[encoded_cols].astype(int)

        return df_encoded

    def leave_one_out_target_encoding(self, df, feature, target):
        """
        Effectue un Leave-One-Out Target Encoding sur la colonne feature en remplaçant chaque valeur par la moyenne des valeurs cibles pour cette classe, en excluant la ligne actuelle.

        :param df: DataFrame contenant la colonne feature et la colonne cible target.
        :param feature: Nom de la colonne à encoder.
        :param target: Nom de la colonne cible.
        :return: DataFrame avec la colonne encodée ajoutée et la colonne d'origine supprimée.
        """
        target_sum = df.groupby(feature)[target].transform('sum')
        target_count = df.groupby(feature)[target].transform('count')

        loo_encoded = (target_sum - df[target]) / (target_count - 1)

        # Remplace les valeurs NaN
        global_mean = df[target].mean()
        loo_encoded = loo_encoded.fillna(global_mean)

        # Ajouter la colonne encodée au DataFrame et le retourner
        df[f"{feature}_loo_encoded"] = loo_encoded
        df.drop(columns=[feature], axis=1, inplace=True)
        return df

    def create_time_features(self, df):
        """
        Crée des colonnes de date utiles pour l'apprentissage automatique.

        Ajoute deux colonnes:
        - "Order Month" : le mois de l'année de la commande (1 à 12) en tant que nombre entier.
        - "Order Day" : le jour de la semaine de la commande en tant que nombre entier.

        :param df: DataFrame contenant la colonne "Order Date".
        :return: DataFrame avec les nouvelles colonnes ajoutées.
        """
        df["Order Month"] = df["Order Date"].dt.month.astype(int)
        df["Order Day"] = df["Order Date"].dt.weekday
        return df

    def get_season(self, month):
        """
        Détermine la saison de l'année en fonction du mois donné.

        Parameters
        ----------
        month : int
            Numéro du mois (1 pour janvier, 2 pour février, ..., 12 pour décembre).

        Returns
        -------
        str
            La saison correspondant au mois: "Winter", "Spring", "Summer", ou "Autumn".
        """
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    def add_lag_to_sales(self, df, lag):
        """
        Ajoute une colonne "Sales_lag_<lag>" au DataFrame en décalant les ventes de <lag> jours.

        :param df: DataFrame contenant la colonne "Sales".
        :param lag: Nombre de jours pour lequel on ajoute un lag sur les ventes.
        :return: DataFrame avec la colonne "Sales_lag_<lag>" ajoutée.
        """
        df = df.sort_values(by="Order Date")
        df[f"Sales_lag_{lag}"] = df["Sales"].shift(lag)
        df = df.dropna()
        return df
