# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import seaborn as sn
import matplotlib.pyplot as plt
import argparse
import os
import graphviz
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from R_square_clustering import r_square

scaler = None
type_exec = ""
pd.set_option("display.max_columns", None)

def main(args):
    global scaler, type_exec

    if args.full:
        type_exec = "f"
    elif args.demissionnaire:
        type_exec = "d"
    else:
        type_exec = "n"

    if not os.path.exists("./fig/" + type_exec + "/"):
	    os.makedirs("./fig/" + type_exec + "/")

    ##################################################
    ##################################################
    #####                                        #####
    #####       RECUPERATION DES DONNEES         #####
    #####                                        #####
    ##################################################
    ##################################################

    # Chargement des datasets
    df_demissionnaire = pd.read_csv("donnees/data_mining_DB_clients_tbl.csv",
                        usecols = ["CDSEXE", "NBENF", "CDSITFAM", "DTADH", "CDTMT", "CDCATCL", "agedem", "adh"])
    df_random = pd.read_csv("donnees/data_mining_DB_clients_tbl_bis.csv",
                        usecols = ["CDSEXE", "NBENF", "CDSITFAM", "DTADH", "CDTMT", "CDCATCL", "CDMOTDEM", "DTDEM", "DTNAIS"])

    ##################################################
    ##################################################
    #####                                        #####
    #####         NETTOYAGE DES DONNEES          #####
    #####                                        #####
    ##################################################
    ##################################################

    # On remplace le motif de démission par une colonne "démissionnaire"
    df_demissionnaire["demissionnaire"] = True
    df_random["demissionnaire"] = np.where(df_random["CDMOTDEM"].notnull(), True, False)

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

    ##################################################
    ##################################################
    #####                                        #####
    #####           FUSION DES DONNEES           #####
    #####                                        #####
    ##################################################
    ##################################################

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

    # Permet de voir la correlation des données brutes
    save_correlation(df, "_original")

    # Filtrage des colonnes
    #df = df[["sexe", "nb_enfants", "situation_fam", "statut", "categorie", "annee_adh", "age", "duree", "demissionnaire"]]
    df = df[["sexe", "nb_enfants", "situation_fam", "categorie", "age", "duree", "demissionnaire"]]

    # Permet de voir la correlation des données filtrées
    save_correlation(df, "_filtered")

    ##################################################
    ##################################################
    #####                                        #####
    #####         RECODAGE DES DONNEES           #####
    #####                                        #####
    ##################################################
    ##################################################
    
    # On prend autant de démissionnaires que de non démissionnaires
    mini = min([df[df["demissionnaire"] == True].count().iloc[0], df[df["demissionnaire"] == False].count().iloc[0]])
    df = df.groupby(["demissionnaire"]).apply(lambda grp: grp.sample(n = mini)).reset_index(level = [0, 1], drop = True)

    #df_base["situation_fam"] = df_base["situation_fam"].apply(lambda x: ord(x.lower()) - 96).astype(int)
    # Création des données X et Y. X_base et Y_base ne sont pas altérés à partir de maintenant

    X, Y = get_XY(discretization(df, ["categorie", "situation_fam", "sexe"]), type_exec)
    X_base_disc, Y_base_disc = get_XY(discretization(df, ["categorie", "situation_fam", "sexe"]), type_exec)
    X_base, Y_base = get_XY(df, type_exec)

    # Scale des données
    scaler = StandardScaler()
    X_cols = X.columns
    x_scaled = scaler.fit_transform(X.values)
    X = pd.DataFrame(data = x_scaled, columns = X_cols)
    Y = Y.reset_index()["demissionnaire"]

    ##################################################
    ##################################################
    #####                                        #####
    #####   ANALYSE / CLUSTERING DES DONNEES     #####
    #####                                        #####
    ##################################################
    ##################################################

    if args.predcluster or args.clustering:
        # ACP sur les données X
        acp = PCA(svd_solver = "full")
        coord = pd.DataFrame(acp.fit_transform(X))
        corvar, eigval, n, p = get_corvar(X, acp)

        # Affichage de la courpe de variable exprimée par les composantes
        save_eigval_graph(eigval, p)

        # Variance expliquée par composante principale
        print(acp.explained_variance_ratio_)
        
        # Cercle de corrélation des composantes 0-1 et 2-3
        correlation_circle(X, p, 0, 1, corvar)
        correlation_circle(X, p, 2, 3, corvar)

        # Affichage des données projetées sur ces mêmes composantes
        save_acp_graph(acp, coord, X, Y, 0, 1, n, p, corvar)
        save_acp_graph(acp, coord, X, Y, 2, 3, n, p, corvar)

        # Clustering hiérarchique des données
        if type_exec == "d":
            # make_dendrogram(X)
            make_elbow(X);
            make_Kmeans(5, X, coord.iloc[:, :6], Y, X_base);

    ##################################################
    ##################################################
    #####                                        #####
    #####      CLASSIFICATION DES DONNEES        #####
    #####                                        #####
    ##################################################
    ##################################################

    if args.predcluster or args.prediction:
        if type_exec == "f":
            save_tree(X_base_disc, Y_base_disc)
            test_predict(X, Y)

def make_Kmeans(k, X_raw, X_pca, Y, X_base):
    if not os.path.exists("fig/" + type_exec + "/kmeans"):
		os.makedirs("fig/" + type_exec + "/kmeans")

    model = KMeans(n_clusters = k, n_init = 20)
    # Compute cluster centers and predict cluster indices
    X_clustered = model.fit_predict(X_pca)

    # Plot the scatter digram
    plt.figure(figsize = (7,7))
    labels = model.labels_
    plt.scatter(X_pca.iloc[:, 0],X_pca.iloc[:, 1], c = labels.astype(np.float), alpha = 0.5) 
    plt.savefig("fig/" + type_exec + "/kmeans/Kmeans_" + str(k) + "_pca")
    model.fit(X_raw)
    X_raw = pd.DataFrame(scaler.inverse_transform(X_raw), columns = X_raw.columns)
    X_raw["labels"] = labels
    X_base["labels"] = labels
    bins = np.linspace(-10, 10, 30)
    plt.clf()
    # Plot the number of entity for each column and grouped by labels
    for col in X_base.columns:
        if col != "labels":
            res = X_base.groupby(["labels",col]).size().reset_index(name='counts')
            plt.figure(figsize=(10,6))
            sn.barplot(x=col, hue="labels", y="counts", data=res)
            plt.savefig("fig/" + type_exec + "/kmeans"+ col)

# Function called to plot the elbow graph for choosing the kmeans number of cluster.
def make_elbow(X):
    lst_k = range(1, 10)
    lst_rsq = []

    for k in lst_k:
        kmeanModel = KMeans(n_clusters = k)
        kmeanModel.fit(X)
        # lst_rsq.append(np.average(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1)) / X.shape[0])
        lst_rsq.append(r_square(X.values, kmeanModel.cluster_centers_, kmeanModel.labels_, k))

    fig = plt.figure()
    plt.plot(lst_k, lst_rsq, "bx-")
    plt.xlabel("k")
    plt.ylabel("RSQ score")
    plt.title("The Elbow Method showing the optimal k")
    plt.savefig("fig/" + type_exec + "/k-means_elbow_method")
    plt.close()

def make_dendrogram(X_norm):
    # hierarchical clustering
    # lst_labels = map(lambda pair: pair[0] + str(pair[1]), zip(fruits["fruit_name"].values, fruits.index))
    linkage_matrix = linkage(X_norm, "ward")
    fig = plt.figure()
    dendrogram(
        linkage_matrix,
        color_threshold=0,
        show_leaf_counts = True
    )
    plt.title("Hierarchical Clustering Dendrogram (Ward)")
    plt.xlabel("sample index")
    plt.ylabel("distance")
    plt.tight_layout()
    plt.savefig("fig/" + type_exec + "/hierarchical-clustering")
    plt.close()

# Discrétisation des données catégorielles
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
        return df[df["demissionnaire"] == True].drop(columns = "demissionnaire"), df[df["demissionnaire"] == True]["demissionnaire"].astype(int)
    elif choix == "n":
        return df[df["demissionnaire"] == False].drop(columns = "demissionnaire"), df[df["demissionnaire"] == False]["demissionnaire"].astype(int)
    elif choix == "f":
        return df.drop(columns = "demissionnaire"), df["demissionnaire"].astype(int)
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
def save_acp_graph(acp, coord, data, Y, cp1, cp2, n, p, corvar, fixed = False):
    print("Saving ACP instance graph (" + str(cp1) + ", " + str(cp2) + ")...")
    # Affichage des instances étiquetées par le code du pays suivant les 2 facteurs principaux de l"ACP
    fig, axes = plt.subplots(figsize = (12, 12))
    xmin = min(coord.iloc[:, cp1]) if not fixed else (-45 if cp1 == 0 else -5)
    xmax = max(coord.iloc[:, cp1]) if not fixed else (45 if cp1 == 0 else 5)
    ymin = min(coord.iloc[:, cp2]) if not fixed else (-45 if cp1 == 0 else -5)
    ymax = max(coord.iloc[:, cp2]) if not fixed else (45 if cp1 == 0 else 5)
    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)

    for i in range(n):
        plt.annotate(".", (coord.iloc[i, cp1], coord.iloc[i, cp2]), color = ("blue" if Y.loc[i] == 0 else "red"))

    plt.plot([xmin, xmax], [0, 0], color = "silver", linestyle = "-", linewidth = 1)
    plt.plot([0, 0], [ymin, ymax], color = "silver", linestyle = "-", linewidth = 1)
    axes.set_title("Composantes principales " + str(cp1) + " et " + str(cp2))
    axes.set_xlabel("CP " + str(cp1 + 1))
    axes.set_ylabel("CP " + str(cp2 + 1))
    plt.savefig("fig/" + type_exec + "/acp_instances_plan_" + str(cp1) + "-" + str(cp2))
    plt.close(fig)
    print("Done")

def save_tree(X, Y):
    print("Saving Decision Tree...")
    tree_clf = tree.DecisionTreeClassifier(random_state = 0)
    tree_clf = tree_clf.fit(X, Y)
    res = tree.export_graphviz(tree_clf, class_names = ["Non dem", "Dem"], filled = True, rounded = True, out_file = None, feature_names = X.columns, max_depth = 3)
    img = graphviz.Source(res)
    img.format = "png"
    img.render("./fig/" + type_exec + "/decision_tree", view = False)
    os.remove("./fig/" + type_exec + "/decision_tree")
    print("Done")

# Sauvegarde le cercle des corrélations
def correlation_circle(df, nb_var, x_axis, y_axis, corvar):
    print("Saving ACP correlation circle (" + str(x_axis) + ", " + str(y_axis) + ")...")
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
    plt.savefig("fig/" + type_exec + "/acp_correlation_circle_axes_" + str(x_axis) + "_" + str(y_axis))
    plt.close(fig)
    print("Done")

# Calcule la corvar de l'ACP
def get_corvar(X, acp):
    n = np.size(X, 0)
    p = np.size(X, 1)
    eigval = float(n - 1) / n * acp.explained_variance_
    sqrt_eigval = np.sqrt(eigval)
    corvar = np.zeros((p, p))

    for k in range(p):
        corvar[:, k] = acp.components_[k, :] * sqrt_eigval[k]
    
    return corvar, eigval, n, p

# Sauvegarde du graphe des eigval
def save_eigval_graph(eigval, p):
    print("Saving ACP eigenvalues graph...")
    fig = plt.figure()
    plt.plot(np.arange(1, p + 1), eigval)
    plt.title("Variance par composante")
    plt.ylabel("Eigen values")
    plt.xlabel("Factor number")
    plt.savefig("fig/" + type_exec + "/acp_eigen_values")
    plt.close(fig)
    print("Done")

# Sauvegarde de la heatmap de correlation des données df
def save_correlation(df, postfix = ""):
    print("Saving data correlation graph...")
    corr = df.corr()
    ax = sn.heatmap(corr, annot = True)
    plt.savefig("fig/" + type_exec + "/correlation" + postfix)
    plt.clf()
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required = True)
    group2 = parser.add_mutually_exclusive_group(required = True)

    group1.add_argument( "-f", "--full", help = "Execution du script sur tous les adhérents", action = "store_true", default = False)
    group1.add_argument( "-d", "--demissionnaire", help = "Execution du script sur les demissionnaires", action = "store_true", default = False)
    group1.add_argument( "-n", "--nondemissionnaire", help = "Execution du script sur les non demissionnaires", action = "store_true", default = False)

    group2.add_argument( "-c", "--clustering", help = "Clustering des données", action = "store_true", default = False)
    group2.add_argument( "-p", "--prediction", help = "Prédiction des données", action = "store_true", default = False)
    group2.add_argument( "-pa", "--predcluster", help = "Clustering et prédiction des données", action = "store_true", default = False)

    args = parser.parse_args()

    main(args)