# -*- coding: utf-8 -*-
import pandas as pd

# Chargement des données avec les colonnes utiles
df = pd.read_csv("donnees/data_mining_DB_clients_tbl.csv",
                      usecols = ["CDSEXE", "NBENF", "CDSITFAM", "CDCATCL", "rangagead", "rangagedem", "rangadh"])

# On affiche la forme des données avant traitement
print df.head(5)

# Les colonnes catégorielles sont formattées "Groupe Description"
# On ne garde donc que le Groupe
df["rangagead"] = df["rangagead"].str.split(" ").str[0].fillna(0).astype(int)
df["rangagedem"] = df["rangagedem"].str.split(" ").str[0].fillna(0).astype(str)
df["rangadh"] = df["rangadh"].str.split(" ").str[0].fillna(0).astype(int)

# Le groupe de la colonne rangagedem est écrit en hexadécimal, conversion en décimal
df["rangagedem"] = df["rangagedem"].apply(lambda x: int(x, 16))

# La colonne CDSITFAM est groupée par des lettres décrivant la situation familiale
# Remplacement de ces lettres par les numéro de lettre correspondant
df["CDSITFAM"] = df["CDSITFAM"].apply(lambda x: ord(x.lower()) - 96).astype(int)

# Tous ces gens la ont démissionné
df["demission"] = True

# On affiche la forme des données après traitement
print df.head(5)


# COMPARER
# Id,CDSEXE,MTREV,NBENF,CDSITFAM,DTADH,CDTMT,CDDEM,DTDEM,ANNEE_DEM,CDMOTDEM,CDCATCL,AGEAD,rangagead,agedem,rangagedem,rangdem,adh,rangadh
# Id,CDSEXE,DTNAIS,MTREV,NBENF,CDSITFAM,DTADH,CDTMT,CDMOTDEM,CDCATCL,Bpadh,DTDEM