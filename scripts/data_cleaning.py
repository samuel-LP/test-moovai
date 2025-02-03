import pandas as pd

class DataCleaning:
    def __init__(self):
        pass
    ### fonction principale pour le data cleaning ###
    def cleaning(self, df, cols_to_drop, date_column, id_columns, numeric_cols):
        """
        Fonction principale de data cleaning qui appelle les fonctions 
        - date_to_datetime pour convertir les colonnes de date en datetime
        - aggregate_duplicate_rows pour agr ger les lignes dupliqu es
        - drop_unnecessary_columns pour supprimer les colonnes inutiles
        - remove_outliers_from_profit pour supprimer les outliers de la colonne Sales

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame  contenant les donn es  nettoyer
        cols_to_drop : list of str
            Liste des noms de colonnes  supprimer
        date_column : str
            Nom de la colonne contenant les dates  convertir en datetime
        id_columns : list of str
            Liste des noms des colonnes identifiant les lignes dupliqu es
        numeric_cols : list of str
            Liste des noms des colonnes num riques  agr ger
        threshold : int
            Seuil de d tection des outliers dans la colonne Sales

        Returns
        -------
        pandas DataFrame
            DataFrame nettoy  et pr par  pour la mod lisation
        """
        print("cleaning data...")

        print("passage des dates en format datetime...")
        df_temp = self.date_to_datetime(df, date_column)

        print("aggregation des doublons...")
        df_temp = self.aggregate_duplicate_rows(df_temp, id_columns, numeric_cols)

        print("suppression des colonnes inutiles...")
        df_final = self.drop_unnecessary_columns(df_temp, cols_to_drop)

        print("cleaning terminé ✅")
        return df_final

    def drop_unnecessary_columns(self, df, columns_to_drop):
        """
        Supprime les colonnes spécifiées du DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            Le DataFrame à partir duquel les colonnes doivent être supprimées.
        columns_to_drop : list of str
            Liste des noms des colonnes à supprimer.

        Returns
        -------
        pandas DataFrame
            DataFrame avec les colonnes spécifiées supprimées. Les colonnes manquantes sont ignorées.
        """
        return df.drop(columns=columns_to_drop, errors='ignore')  # Ignore les colonnes manquantes

    def date_to_datetime(self, df, date_columns):
        """
        Convertit les colonnes de date spécifiées du DataFrame en format datetime.

        Parameters
        ----------
        df : pandas DataFrame
            Le DataFrame contenant les colonnes de date à convertir.
        date_columns : list of str
            Liste des noms des colonnes de date à convertir au format datetime.

        Returns
        -------
        pandas DataFrame
            DataFrame avec les colonnes de date converties en format datetime.
        """
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], format="%m/%d/%Y")
        return df

    def identify_duplicate_rows(self, df, id_columns):
        """
        Identifie les lignes dupliquées dans le DataFrame en fonction des colonnes spécifiées.

        Parameters
        ----------
        df : pandas DataFrame
            Le DataFrame à analyser.
        id_columns : list of str
            Liste des noms des colonnes d'identification pour la duplication.

        Returns
        -------
        pandas DataFrame
            DataFrame contenant uniquement les lignes dupliquées.
        """
        return df[df.duplicated(subset=id_columns, keep=False)]

    def aggregate_duplicate_rows(self, df, id_columns, numeric_cols):
        """
        Agrège les lignes dupliqu es dans le DataFrame en fonction des colonnes d'identification
        spécifi es.

        Les colonnes num riques sont additionn es (somme) et les colonnes non num riques sont
        conserv es en prenant la premi re valeur (par d faut).

        Parameters
        ----------
        df : pandas DataFrame
            Le DataFrame  agr ger.
        id_columns : list of str
            Liste des noms des colonnes d'identification pour l'agr gation.
        numeric_cols : list of str
            Liste des noms des colonnes num riques  agr ger.

        Returns
        -------
        pandas DataFrame
            DataFrame agr g  avec les colonnes num riques somm es et les colonnes non num riques
            conserv es.
        """
        return df.groupby(id_columns, as_index=False).agg({
            **{col: 'sum' for col in numeric_cols},
            **{col: 'first' for col in df.columns if col not in numeric_cols + id_columns}
        })