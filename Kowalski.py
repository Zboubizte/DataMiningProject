# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix

# Chargement des datasets
df_demissionnaire = pd.read_csv("donnees/data_mining_DB_clients_tbl.csv",
                    usecols = ["CDSEXE", "NBENF", "CDSITFAM", "DTADH", "CDTMT", "CDCATCL", "agedem", "adh"])
df_random = pd.read_csv("donnees/data_mining_DB_clients_tbl_bis.csv",
                    usecols = ["CDSEXE", "NBENF", "CDSITFAM", "DTADH", "CDTMT", "CDCATCL", "CDMOTDEM", "DTDEM", "DTNAIS"])

# On remplace le motif de démission par une colonne "démissionnaire"
df_demissionnaire["demissionnaire"] = True
df_random["demissionnaire"] = np.where(df_random["CDMOTDEM"].notnull(), True, False)
#df_random = df_random.drop(columns = "CDMOTDEM")

# On renomme les colonnes
df_demissionnaire = df_demissionnaire.rename(columns = {"agedem": "age", "adh": "duree"})

# On supprime les lignes contenant des erreurs de saisies.
# C'est à dire les démissionnaires n'ayant pas de date de démission ou les dates de naissances vides
df_random = df_random.drop(df_random[(df_random["CDMOTDEM"] == "DC") & (df_random["DTDEM"] == "1900-12-31")].index)
df_random = df_random.drop(df_random[df_random["DTNAIS"] == "0000-00-00"].index)
df_random = df_random.drop(df_random[df_random["CDSEXE"] == 1].index)

# On calcule l'age de démission, ou l'age si il n'y a pas de démission dans le df_random
df_random["age"] = np.where(df_random["demissionnaire"] == True,
                            df_random["DTDEM"].str.slice(stop = 4).astype(int) - df_random["DTNAIS"].str.slice(stop = 4).astype(int),
                            2007 - df_random["DTNAIS"].str.slice(stop = 4).astype(int))

# On calcule la durée d'adhésion (jusqu'à démission si démission)
df_random["duree"] = np.where(df_random["demissionnaire"] == True,
                              df_random["DTDEM"].str.slice(stop = 4).astype(int) - df_random["DTADH"].str.slice(stop = 4).astype(int),
                              2007 - df_random["DTADH"].str.slice(stop = 4).astype(int))

# On ordonne les colonnes de la même façon
df_demissionnaire = df_demissionnaire[["CDSEXE", "NBENF", "CDSITFAM", "CDTMT", "CDCATCL", "DTADH", "age", "duree", "demissionnaire"]]
df_random = df_random[["CDSEXE", "NBENF", "CDSITFAM", "CDTMT", "CDCATCL", "DTADH", "age", "duree", "demissionnaire"]]

# On concatène les deux datasets
df = pd.concat([df_demissionnaire, df_random], ignore_index = True)
df["age"] = df["age"].astype(int)
df["duree"] = df["duree"].astype(int)
df = df.rename(columns = {"CDSEXE": "sexe",
                          "NBENF": "nb_enfants",
                          "CDSITFAM": "situation_fam",
                          "CDTMT": "statut",
                          "CDCATCL": "categorie",
                          "DTADH": "annee_adh"})

# La colonne CDSITFAM est groupée par des lettres décrivant la situation familiale
# Remplacement de ces lettres par les numéro de lettre correspondant
df["situation_fam"] = df["situation_fam"].apply(lambda x: ord(x.lower()) - 96).astype(int)

df["annee_adh"] = df["annee_adh"].str.slice(stop = 4).astype(int)

# Discrétisation des données catégorielles
# df = pd.concat([pd.get_dummies(df["categorie"], prefix = "categorie"), df.drop(columns = "categorie")], axis = 1)
# df = pd.concat([pd.get_dummies(df["statut"], prefix = "statut"), df.drop(columns = "statut")], axis = 1)
# df = pd.concat([pd.get_dummies(df["situation_fam"], prefix = "situation_fam"), df.drop(columns = "situation_fam")], axis = 1)
# df = pd.concat([pd.get_dummies(df["sexe"], prefix = "sexe"), df.drop(columns = "sexe")], axis = 1)

X = df.drop(columns = "demissionnaire")
Y = df["demissionnaire"]

# Initialisation des variables de classification
dummycl = DummyClassifier(strategy = "most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
logreg = LogisticRegression(solver = "lbfgs")
svc = SVC(gamma = "auto")

# Initialisation des listes de classifers
lst_classif = [dummycl, gmb, dectree, logreg, svc]
lst_classif_names = ["Dummy", "Naive Bayes", "Decision tree", "Logistic regression", "SVC"]

# Test des différents classifiers avec 5 passes et de la cross validation
for clf, name_clf in zip(lst_classif, lst_classif_names):
    scores = cross_val_score(clf, X, Y, cv = 5)
    print("Accuracy of " + name_clf + " classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(confusion_matrix(Y, cross_val_predict(clf, X, Y, cv = 5)))