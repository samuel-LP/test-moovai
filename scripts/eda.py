import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class EDA:
    def __init__(self):
        pass

    def statistics(self, df):
        """
        Return a pandas Series with some statistics about df.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame to compute statistics on.

        Returns
        -------
        pandas Series
            Series containing some statistics about df.
        """
        return df.describe()

    def informations_about_dataframe(self, df):
        """
        Identifies and returns the categorical and numerical columns of a DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame to analyze.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - cat_cols: Index of categorical column names.
            - num_cols: Index of numerical column names.
        """
        cat_cols = df.select_dtypes(exclude=np.number).columns
        num_cols = df.select_dtypes(include=np.number).columns
        return cat_cols, num_cols

    def missing_values(self, df):
        """
        Affiche le nombre total de valeurs manquantes dans le DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            Le DataFrame à analyser pour les valeurs manquantes.
        """
        print(f"Le dataframe a {df.isnull().sum().sum()} valeurs manquantes:")
        return ""
    
    def unique_values(self, df):
        """
        Identify and report the number of unique values for each categorical column in the DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame to analyze.

        Returns
        -------
        str
            An empty string (output is printed directly during the process).
        """
        cat_cols = df.select_dtypes(exclude=np.number).columns
        result = []
        for col in cat_cols:
            result.append(f"La feature {col} a {df[col].nunique()} valeurs uniques")
        return ""
    
    def duplicate_rows(self, df):
        """
        Affiche le nombre total de lignes dupliquées dans le DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            Le DataFrame à analyser pour les lignes dupliquées.
        """
        print("Number of duplicate rows:", df.duplicated().sum())
        return ""

    ###### graphs ######

    def plot_distribution(self, df, col):
        """
        Trace un graphique qui affiche la distribution de la colonne {col} de {df}.

        Le graphique contient un boxplot et un histogramme de la distribution de la colonne {col}.
        Les paramètres sont les suivants:
        - {df}: DataFrame contenant la colonne à analyser.
        - {col}: Nom de la colonne à analyser.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.boxplot(y=df[col], ax=axes[0], color="lightblue")
        axes[0].set_title(f"Boxplot de {col}")
        
        sns.histplot(df[col], bins=30, kde=True, color="darkblue", edgecolor="black", ax=axes[1])
        axes[1].set_title(f"Distribution de {col}")
        
        plt.tight_layout()
        plt.grid()
        plt.show()

    def get_plot_for_sales_day_by_day(self, df, date_features):
        """
        Trace un graphique montrant l'évolution des ventes par jour pour deux colonnes de dates différentes.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame contenant les colonnes de date et de ventes.
        date_features : list of str
            Liste de deux noms de colonnes de date dans le DataFrame.

        Returns
        -------
        None
            Le graphique est affiché directement.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for i, date_col in enumerate(date_features):
            # Convertir la colonne de date en datetime
            df[date_col] = pd.to_datetime(df[date_col])

            # Regrouper les ventes par date
            df_grouped = df.groupby(date_col)["Sales"].sum().reset_index()

            axes[i].plot(df_grouped[date_col], df_grouped["Sales"], linestyle="-")
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel("Total des ventes")
            axes[i].set_title(f"Évolution des ventes ({date_col})")

            # Sélectionner un sous-ensemble des dates pour afficher sur l'axe X
            tick_positions = df_grouped[date_col][::30]  # Prendre une date tous les 30 jours
            tick_labels = tick_positions.dt.strftime("%Y-%m-%d")

            axes[i].set_xticks(tick_positions)
            axes[i].set_xticklabels(tick_labels, rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

    def get_plot_for_sales_every_year(self, df, year, order_or_ship_date):
        
        """
        Trace deux graphiques montrant l'évolution des ventes pour deux colonnes de dates différentes sur une année donnée.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame contenant les colonnes de date et de ventes.
        year : int
            Année pour laquelle les ventes doivent être affichées.
        order_or_ship_date : list of str
            Liste de deux noms de colonnes de date dans le DataFrame.

        Returns
        -------
        None
            Les graphiques sont affichés directement.
        """
        df[order_or_ship_date] = df[order_or_ship_date].apply(pd.to_datetime)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for i, date_col in enumerate(order_or_ship_date):
            # Filtrer les données de l'année en fonction du type de date
            df_reduced = df[(df[date_col] >= f"{year}-01-01") & (df[date_col] <= f"{year}-12-31")]

            # Agréger les ventes par jour
            df_reduced_grouped = df_reduced.groupby(date_col)["Sales"].sum().reset_index()

            axes[i].plot(df_reduced_grouped[date_col], df_reduced_grouped["Sales"], marker="o", linestyle="-")
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel("Total des ventes")
            axes[i].set_title(f"Évolution des ventes en {year} ({date_col})")
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def sales_per_seasons(self, df, seasons):
        
        """
        Trace un graphique en camembert montrant la répartition des ventes par saison.

        :param df: DataFrame contenant la colonne 'Order Season' et 'Sales'.
        :param seasons: Liste de saisons à inclure dans le graphique.
        """
        df_season_sales = df.groupby("Order Season")["Sales"].sum().reset_index()

        df_season_sales = df_season_sales.set_index("Order Season").loc[seasons]

        pastel_colors = sns.color_palette("pastel")

        plt.figure(figsize=(8, 8))
        plt.pie(df_season_sales["Sales"], labels=df_season_sales.index, autopct='%1.1f%%', colors=pastel_colors, startangle=90)
        plt.title("Répartition des ventes par saison (%)")
        plt.show()

    def plot_bar_chart(self, df, feature):
        """
        Trace un graphique en barres pour la somme d'une variable donnée en fonction de 'State'.

        :param df: DataFrame contenant la colonne 'State' et la variable à analyser.
        :param var: Nom de la colonne numérique à visualiser.
        """
        if feature not in df.columns:
            print(f"La colonne '{feature}' n'existe pas dans le DataFrame.")
            return

        grouped_data = df.groupby("State")[feature].sum().sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        grouped_data.plot(kind="bar")

        plt.xlabel("État")
        plt.ylabel(f"Total de {feature}")
        plt.title(f"Relation entre {feature} et State")
        plt.xticks(rotation=90)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    def plot_pie_chart(self, df, feature):
        """
        Trace un graphique en camembert pour la répartition d'une featureiable donnée.
        Utilise seaborn avec une palette pastel.

        :param df: DataFrame contenant la featureiable à analyser.
        :param feature: Nom de la colonne catégorielle ou numérique à visualiser.
        """
        if feature not in df.columns:
            print(f"La colonne '{feature}' n'existe pas dans le DataFrame.")
            return

        # Vérifier si la feature est catégorielle ou numérique
        if df[feature].dtype == 'O':  # Catégorielle
            grouped_data = df[feature].value_counts()
        else:  # Numérique, on agrège par 'State' par défaut
            grouped_data = df.groupby("Sales")[feature].sum()

        plt.figure(figsize=(10, 10))
        colors = sns.color_palette("pastel", len(grouped_data))
        plt.pie(grouped_data, labels=grouped_data.index, autopct="%1.1f%%", startangle=90, colors=colors)

        plt.title(f"Répartition de {feature}")
        plt.show()


    def plot_ship_mode_frequency(self, df):
        """
        Trace un graphique en barres montrant la fréquence des modes de livraison (Ship Mode).
        Utilise seaborn avec une palette pastel.

        :param df: DataFrame contenant la colonne 'Ship Mode'.
        """
        if "Ship Mode" not in df.columns:
            print("La colonne 'Ship Mode' n'existe pas dans le DataFrame.")
            return

        ship_mode_counts = df["Ship Mode"].value_counts()

        plt.figure(figsize=(8, 6))
        sns.barplot(x=ship_mode_counts.index, y=ship_mode_counts.values, palette="pastel")

        plt.xlabel("Mode de livraison")
        plt.ylabel("Nombre d'apparitions")
        plt.title("Fréquence des Ship Mode")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.show()


    def plot_sales_vs_discount(self, df):
        """
        Trace un nuage de points montrant la relation entre Sales et Discount.
        Utilise seaborn avec une palette pastel.

        :param df: DataFrame contenant les colonnes 'Sales' et 'Discount'.
        """
        if "Sales" not in df.columns or "Discount" not in df.columns:
            print("Les colonnes 'Sales' et 'Discount' doivent exister dans le DataFrame.")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="Sales", y="Discount", data=df, palette="pastel", alpha=0.6)

        plt.xlabel("Ventes (Sales)")
        plt.ylabel("Remise (Discount)")
        plt.title("Relation entre Sales et Discount")
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.show()


    def plot_discount_vs_profit(self, df):
        """
        Trace un nuage de points montrant la relation entre Discount et Profit.
        Utilise seaborn avec une palette pastel.

        :param df: DataFrame contenant les colonnes 'Discount' et 'Profit'.
        """
        if "Discount" not in df.columns or "Profit" not in df.columns:
            print("Les colonnes 'Discount' et 'Profit' doivent exister dans le DataFrame.")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="Discount", y="Profit", data=df, palette="pastel", alpha=0.6)

        plt.xlabel("Remise (Discount)")
        plt.ylabel("Profit")
        plt.title("Relation entre Discount et Profit")
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.show()

    def plot_monthly_sales_trend(self, df, window):
        """
        Trace un graphique des tendances des ventes mensuelles par sous-catégorie avec une moyenne mobile.

        :param df: DataFrame contenant les colonnes 'Order Date', 'Sub-Category' et 'Sales'.
        :param window: Fenêtre de lissage (nombre de mois pour la moyenne mobile).
        """
        if "Order Date" not in df.columns or "Sub-Category" not in df.columns or "Sales" not in df.columns:
            print("Le DataFrame doit contenir les colonnes 'Order Date', 'Sub-Category' et 'Sales'.")
            return

        # Regrouper par mois et sous-catégorie
        df["Order Month"] = df["Order Date"].dt.to_period("M")
        sales_by_subcategory_monthly = df.groupby(["Order Month", "Sub-Category"])["Sales"].sum().reset_index()
        
        sales_by_subcategory_monthly["Order Month"] = sales_by_subcategory_monthly["Order Month"].dt.to_timestamp()

        # Appliquer une moyenne mobile sur les ventes
        sales_by_subcategory_monthly["Sales_rolling"] = (
            sales_by_subcategory_monthly.groupby("Sub-Category")["Sales"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=sales_by_subcategory_monthly, x="Order Month", y="Sales_rolling", hue="Sub-Category", palette="pastel")

        plt.xlabel("Date")
        plt.ylabel(f"Ventes moyennes ({window} mois)")
        plt.title(f"Tendances des ventes mensuelles par sous-catégorie (lissage de {window} mois)")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.show()
