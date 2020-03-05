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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def main():
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
    # Et les personnes mal renseignées, c'est à dire le sexe 1
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

    # On ne garde que l'année de l'adhésion
    df["annee_adh"] = df["annee_adh"].str.slice(stop = 4).astype(int)

    # Ligne 1: on ne discretise pas; Ligne 2: on discretise
    # df["situation_fam"] = df["situation_fam"].apply(lambda x: ord(x.lower()) - 96).astype(int)
    df = discretization(df, ["categorie", "statut", "situation_fam", "sexe"])

    X, Y = get_XY(df, "d")

    scaler = StandardScaler()
    X_cols = X.columns
    x_scaled = scaler.fit_transform(X.values)
    X = pd.DataFrame(data = x_scaled, columns = X_cols)

    print X

    # ACP sur les données X
    acp = PCA(svd_solver = "full")
    coord = acp.fit_transform(X)
    print(acp.explained_variance_ratio_)
    save_acp_graph(acp, coord, X, 0, 1)
    save_acp_graph(acp, coord, X, 2, 3)

    corvar, n, p = get_corvar(X, acp)

    correlation_circle(X, p, 0, 1, corvar)
    correlation_circle(X, p, 2, 3, corvar)

# Discrétisation des données catégorielles passées en paramètre
def discretization(df, col_list):
    for col in col_list:
        df = pd.concat([pd.get_dummies(df[col], prefix = col), df.drop(columns = col)], axis = 1)
    return df

# Renvoie les données au format X et Y en fonction d'un choix :
# d : X = démissionnaires
# n : X = non démissionnaires
# f : X = corpus complet
def get_XY(df, choix):
    if choix == "d":
        return df[df["demissionnaire"] == True].drop(columns = "demissionnaire"), df[df["demissionnaire"] == True]["demissionnaire"]
    elif choix == "n":
        return df[df["demissionnaire"] == False].drop(columns = "demissionnaire"), df[df["demissionnaire"] == False]["demissionnaire"]
    elif choix == "f":
        return df.drop(columns = "demissionnaire"), df["demissionnaire"]
    else:
        exit("erreur")

# Teste de prédire les données avec 5 classifiers différents
def test_predict(X, Y):
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

# Sauvegarde du graphe de l'acp par rapport à deux composantes cp1 et cp2
def save_acp_graph(acp, coord, data, cp1, cp2, fixed = False):
    # Calcul des valeurs propres et de la matrice de corrélation des variables
    n = np.size(data, 0)
    p = np.size(data, 1)
    eigval = float(n - 1) / n * acp.explained_variance_
    sqrt_eigval = np.sqrt(eigval)
    corvar = np.zeros((p, p))

    for k in range(p):
        corvar[:, k] = acp.components_[k, :] * sqrt_eigval[k]

    # Affichage des instances étiquetées par le code du pays suivant les 2 facteurs principaux de l"ACP
    fig, axes = plt.subplots(figsize = (12, 12))
    xmin = min(coord[:, cp1]) if not fixed else (-45 if cp1 == 0 else -5)
    xmax = max(coord[:, cp1]) if not fixed else (45 if cp1 == 0 else 5)
    ymin = min(coord[:, cp2]) if not fixed else (-45 if cp1 == 0 else -5)
    ymax = max(coord[:, cp2]) if not fixed else (45 if cp1 == 0 else 5)
    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)

    for i in range(n):
        plt.annotate("+", (coord[i, cp1], coord[i, cp2]))

    plt.plot([xmin, xmax], [0, 0], color = "silver", linestyle = "-", linewidth = 1)
    plt.plot([0, 0], [ymin, ymax], color = "silver", linestyle = "-", linewidth = 1)
    axes.set_title("Composantes principales " + str(cp1) + " et " + str(cp2))
    axes.set_xlabel("CP " + str(cp1 + 1))
    axes.set_ylabel("CP " + str(cp2 + 1))
    plt.savefig("fig/acp_instances_plan_" + str(cp1) + "-" + str(cp2))
    plt.close(fig)

# Sauvegarde le cercle des corrélations
def correlation_circle(df, nb_var, x_axis, y_axis, corvar):
    fig, axes = plt.subplots(figsize = (8, 8))
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    axes.set_xlabel("CP" + str(x_axis + 1))
    axes.set_ylabel("CP" + str(y_axis + 1))

    for j in range(nb_var):
        plt.annotate(df.columns[j], (corvar[j, x_axis], corvar[j, y_axis]))

    plt.plot([-1, 1], [0, 0], color = "silver", linestyle = "-", linewidth = 1)
    plt.plot([0, 0], [-1, 1], color = "silver", linestyle = "-", linewidth = 1)

    cercle = plt.Circle((0, 0), 1, color = "blue", fill = False)
    axes.add_artist(cercle)
    plt.savefig("fig/acp_correlation_circle_axes_" + str(x_axis) + "_" + str(y_axis))
    plt.close(fig)

# Calcule la corvar de l'ACP
def get_corvar(X, acp):
    n = np.size(X, 0)
    p = np.size(X, 1)
    eigval = float(n - 1) / n * acp.explained_variance_
    sqrt_eigval = np.sqrt(eigval)
    corvar = np.zeros((p, p))

    for k in range(p):
        corvar[:, k] = acp.components_[k, :] * sqrt_eigval[k]
    
    return corvar, n, p

if __name__ == "__main__":
    main()