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
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
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

    # La colonne CDSITFAM est groupée par des lettres décrivant la situation familiale
    # Remplacement de ces lettres par les numéro de lettre correspondant
    df["situation_fam"] = df["situation_fam"].apply(lambda x: ord(x.lower()) - 96).astype(int)

    # On ne garde que l'année de l'adhésion
    df["annee_adh"] = df["annee_adh"].str.slice(stop = 4).astype(int)

    # df = discretization(df, ["categorie", "statut", "situation_fam", "sexe"])

    X, Y = get_XY(df, "d")

    # ACP sur les données X_scaled
    acp = PCA(svd_solver = "full")
    coord = acp.fit_transform(X)
    save_acp_graph(acp, coord, X, 0, 1)
    save_acp_graph(acp, coord, X, 2, 3)
    print(acp.explained_variance_ratio_)

    # make_elbow(X);
    k = 4
    pca_components = pd.DataFrame(coord)
    make_Kmeans(k, pca_components);

def make_Kmeans(k, X):
    
    model = KMeans(n_clusters=k)
    fig = plt.figure(figsize=(8, 6))
    # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    f, ax = plt.subplots(1, 1, sharey=True,figsize=(10,6))
    model.fit(X)
    labels = model.labels_
    X['labels'] = labels
    # print(X.groupby(['labels','nb_enfants']).size())
    # print(X.groupby(['labels','categorie']).size())
    


# Function called to plot the elbow graph for choosing the kmeans number of cluster.
def make_elbow(X):
 # Plot elbow graphs for KMeans using R square and purity scores
    lst_k=range(2,10)
    lst_rsq = []
    for k in lst_k:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        lst_rsq.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    fig = plt.figure()
    plt.plot(lst_k, lst_rsq, 'bx-')
    plt.xlabel('k')
    plt.ylabel('RSQ score')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig('fig/k-means_elbow_method')
    plt.close()


# Discrétisation des données catégorielles
def discretization(df, col_list):
    for col in col_list:
        df = pd.concat([pd.get_dummies(df[col], prefix = col), df.drop(columns = col)], axis = 1)
    return df

def get_XY(df, choix):
    if choix == "d":
        return df[df["demissionnaire"] == True].drop(columns = "demissionnaire"), df[df["demissionnaire"] == True]["demissionnaire"]
    elif choix == "n":
        return df[df["demissionnaire"] == False].drop(columns = "demissionnaire"), df[df["demissionnaire"] == False]["demissionnaire"]
    elif choix == "f":
        return df.drop(columns = "demissionnaire"), df["demissionnaire"]
    else:
        exit("erreur")

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

def save_acp_graph(acp, coord, data, cp1, cp2):
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
    xmin = min(coord[:, cp1])
    xmax = max(coord[:, cp1])
    ymin = min(coord[:, cp2])
    ymax = max(coord[:, cp2])
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

if __name__ == "__main__":
    main()