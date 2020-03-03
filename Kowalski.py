# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# Chargement des datasets
df_demissionnaire = pd.read_csv("donnees/data_mining_DB_clients_tbl.csv",
                    usecols = ["CDSEXE", "NBENF", "CDSITFAM", "DTADH", "CDTMT", "CDCATCL", "agedem"])
df_random = pd.read_csv("donnees/data_mining_DB_clients_tbl_bis.csv",
                    usecols = ["CDSEXE", "NBENF", "CDSITFAM", "DTADH", "CDTMT", "CDCATCL", "CDMOTDEM", "DTDEM", "DTNAIS"])

# On remplace le motif de démission par une colonne "démissionnaire"
df_demissionnaire["demissionnaire"] = True
df_random["demissionnaire"] = np.where(df_random["CDMOTDEM"].notnull(), True, False)
#df_random = df_random.drop(columns = "CDMOTDEM")

# On renomme la colonne agedem en age
df_demissionnaire = df_demissionnaire.rename(columns = {"agedem": "age"})

# On supprime les lignes contenant des erreurs de saisies.
# C'est à dire les démissionnaires n'ayant pas de date de démission ou les dates de naissances vides
df_random = df_random.drop(df_random[(df_random["CDMOTDEM"] == "DC") & (df_random["DTDEM"] == "1900-12-31")].index)
df_random = df_random.drop(df_random[df_random["DTNAIS"] == "0000-00-00"].index)

# On calcule l'age de démission, ou l'age si il n'y a pas de démission dans le df_random
df_random.loc[df_random["demissionnaire"] == True, "age"] = df_random["DTDEM"].str.slice(stop = 4).astype(int) - df_random["DTADH"].str.slice(stop = 4).astype(int)
df_random.loc[df_random["demissionnaire"] == False, "age"] = 2007 - df_random["DTNAIS"].str.slice(stop = 4).astype(int)

# On ordonne les colonnes de la même façon
df_demissionnaire = df_demissionnaire[["CDSEXE", "NBENF", "CDSITFAM", "DTADH", "CDTMT", "CDCATCL", "age", "demissionnaire"]]
df_random = df_random[["CDSEXE", "NBENF", "CDSITFAM", "DTADH", "CDTMT", "CDCATCL", "age", "demissionnaire"]]

# On concatène les deux datasets
df = pd.concat([df_demissionnaire, df_random], ignore_index = True)
df["age"] = df["age"].astype(int)

# La colonne CDSITFAM est groupée par des lettres décrivant la situation familiale
# Remplacement de ces lettres par les numéro de lettre correspondant
df["CDSITFAM"] = df["CDSITFAM"].apply(lambda x: ord(x.lower()) - 96).astype(int)

# Transformation de date d'adhésion en durée d'adhésion
df["DTADH"] = 2007 - df["DTADH"].str.slice(stop = 4).astype(int)
df = df.rename(columns = {"DTADH": "dureeadh"})

print df.head(1)
print df.dtypes