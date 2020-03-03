# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# Chargement des datasets
df_demissionnaire = pd.read_csv("donnees/data_mining_DB_clients_tbl.csv",
                    usecols = ["CDSEXE", "NBENF", "CDSITFAM", "CDTMT", "CDCATCL", "agedem", "adh"])
df_random = pd.read_csv("donnees/data_mining_DB_clients_tbl_bis.csv",
                    usecols = ["CDSEXE", "NBENF", "CDSITFAM", "DTADH", "CDTMT", "CDCATCL", "CDMOTDEM", "DTDEM", "DTNAIS"])

# On remplace le motif de démission par une colonne "démissionnaire"
df_demissionnaire["demissionnaire"] = True
df_random["demissionnaire"] = np.where(df_random["CDMOTDEM"].notnull(), True, False)
#df_random = df_random.drop(columns = "CDMOTDEM")

# On renomme les colonnes
df_demissionnaire = df_demissionnaire.rename(columns = {"agedem": "age"})
df_demissionnaire = df_demissionnaire.rename(columns = {"adh": "duree"})

# On supprime les lignes contenant des erreurs de saisies.
# C'est à dire les démissionnaires n'ayant pas de date de démission ou les dates de naissances vides
df_random = df_random.drop(df_random[(df_random["CDMOTDEM"] == "DC") & (df_random["DTDEM"] == "1900-12-31")].index)
df_random = df_random.drop(df_random[df_random["DTNAIS"] == "0000-00-00"].index)

# On calcule l'age de démission, ou l'age si il n'y a pas de démission dans le df_random
df_random["age"] = np.where(df_random["demissionnaire"] == True,
                            df_random["DTDEM"].str.slice(stop = 4).astype(int) - df_random["DTNAIS"].str.slice(stop = 4).astype(int),
                            2007 - df_random["DTNAIS"].str.slice(stop = 4).astype(int))

# On calcule la durée d'adhésion (jusqu'à démission si démission)
df_random["duree"] = np.where(df_random["demissionnaire"] == True,
                              df_random["DTDEM"].str.slice(stop = 4).astype(int) - df_random["DTADH"].str.slice(stop = 4).astype(int),
                              2007 - df_random["DTADH"].str.slice(stop = 4).astype(int))

# On ordonne les colonnes de la même façon
df_demissionnaire = df_demissionnaire[["CDSEXE", "NBENF", "CDSITFAM", "CDTMT", "CDCATCL", "age", "duree", "demissionnaire"]]
df_random = df_random[["CDSEXE", "NBENF", "CDSITFAM", "CDTMT", "CDCATCL", "age", "duree", "demissionnaire"]]

# On concatène les deux datasets
df = pd.concat([df_demissionnaire, df_random], ignore_index = True)
df["age"] = df["age"].astype(int)
df["duree"] = df["duree"].astype(int)

# La colonne CDSITFAM est groupée par des lettres décrivant la situation familiale
# Remplacement de ces lettres par les numéro de lettre correspondant
df["CDSITFAM"] = df["CDSITFAM"].apply(lambda x: ord(x.lower()) - 96).astype(int)

print df.head(1)