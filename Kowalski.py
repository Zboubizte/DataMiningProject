# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
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
from scipy.spatial.distance import cdist


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

    # On ne garde que l'année de l'adhésion
    df["annee_adh"] = df["annee_adh"].str.slice(stop = 4).astype(int)

    save_correlation(df, "_original")

    # Filtrage des colonnes
    # df = df[["sexe", "nb_enfants", "situation_fam", "statut", "categorie", "annee_adh", "age", "duree", "demissionnaire"]]
    df = df[["sexe", "nb_enfants", "situation_fam", "categorie", "age", "duree", "demissionnaire"]]

    save_correlation(df, "_filtered")

    # Ligne 1: on ne discretise pas; Ligne 2: on discretise
    # df["situation_fam"] = df["situation_fam"].apply(lambda x: ord(x.lower()) - 96).astype(int)
    df = discretization(df, ["categorie", "situation_fam", "sexe"]) 
    
    # On prend autant de démissionnaires que de non démissionnaires
    mini = min([df[df["demissionnaire"] == True].count().iloc[0], df[df["demissionnaire"] == False].count().iloc[0]])
    df = df.groupby(["demissionnaire"]).apply(lambda grp: grp.sample(n = mini)).reset_index(level = [0, 1], drop = True)

    # Création des données X et Y
    X, Y = get_XY(df, "d")

    # Scale des données
    scaler = StandardScaler()
    X_cols = X.columns
    x_scaled = scaler.fit_transform(X.values)
    X = pd.DataFrame(data = x_scaled, columns = X_cols)
    Y = Y.reset_index()["demissionnaire"]

    # ACP sur les données X
    acp = PCA(svd_solver = "full")
    coord = acp.fit_transform(X)
    corvar, eigval, n, p = get_corvar(X, acp)

    save_eigval_graph(eigval, p)

    # Variance expliquée par composante principale
    # print(acp.explained_variance_ratio_)
    
    # correlation_circle(X, p, 0, 1, corvar)
    # correlation_circle(X, p, 2, 3, corvar)
    # save_acp_graph(acp, coord, X, Y, 0, 1)
    # save_acp_graph(acp, coord, X, Y, 2, 3)
    # test_predict(X, Y)
    # make_dendrogram(X)
    k = 5
    pca_components = pd.DataFrame(coord)
    make_elbow(X);
    make_Kmeans(k, X, pca_components.iloc[:, :6], Y);

def make_Kmeans(k, X_raw, X_pca, Y):
    
    model = KMeans(n_clusters=k, n_init = 20)

    # model.fit(X_pca)
    # labels = model.labels_
    # print(labels)
    #Compute cluster centers and predict cluster indices
    X_clustered = model.fit_predict(X_pca)

    # Plot the scatter digram
    plt.figure(figsize = (7,7))
    labels = model.labels_
    plt.scatter(X_pca.iloc[:,0],X_pca.iloc[:,1], c= labels.astype(np.float), alpha=0.5) 
    plt.savefig("fig/Kmeans_" + str(k) +"_pca")
    df_res = pd.DataFrame(Y.values, columns=['realData'])
    df_res["labels"] = labels
    print(df_res)
    model.fit(X_raw)
    X_raw['labels'] = labels
    # print(X.groupby(['labels','nb_enfants']).size())
    # print(X.groupby(['labels','categorie']).size())

# Function called to plot the elbow graph for choosing the kmeans number of cluster.
from R_square_clustering import r_square
def make_elbow(X):
 # Plot elbow graphs for KMeans using R square and purity scores
    lst_k=range(1,10)
    lst_rsq = []
    for k in lst_k:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        # lst_rsq.append(np.average(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        lst_rsq.append(r_square(X.values, kmeanModel.cluster_centers_,kmeanModel.labels_,k))

    fig = plt.figure()
    plt.plot(lst_k, lst_rsq, 'bx-')
    plt.xlabel('k')
    plt.ylabel('RSQ score')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig('fig/k-means_elbow_method')
    plt.close()


def make_dendrogram(X_norm):
    # hierarchical clustering
    # lst_labels = map(lambda pair: pair[0]+str(pair[1]), zip(fruits['fruit_name'].values,fruits.index))
    linkage_matrix = linkage(X_norm, 'ward')
    fig = plt.figure()
    dendrogram(
        linkage_matrix,
        color_threshold=0,
    )
    plt.title('Hierarchical Clustering Dendrogram (Ward)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.tight_layout()
    plt.savefig('fig/hierarchical-clustering')
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
def save_acp_graph(acp, coord, data, Y, cp1, cp2, fixed = False):
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
        plt.annotate(".", (coord[i, cp1], coord[i, cp2]), color = ("blue" if Y.loc[i] == 0 else "red"))

    plt.plot([xmin, xmax], [0, 0], color = "silver", linestyle = "-", linewidth = 1)
    plt.plot([0, 0], [ymin, ymax], color = "silver", linestyle = "-", linewidth = 1)
    axes.set_title("Composantes principales " + str(cp1) + " et " + str(cp2))
    axes.set_xlabel("CP " + str(cp1 + 1))
    axes.set_ylabel("CP " + str(cp2 + 1))
    plt.savefig("fig/acp_instances_plan_" + str(cp1) + "-" + str(cp2))
    print("ACP instance graph saved (" + str(cp1) + ", " + str(cp2) + ")")
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
    print("ACP correlation circle saved (" + str(x_axis) + ", " + str(y_axis) + ")")
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
    
    return corvar, eigval, n, p

# Sauvegarde du graphe des eigval
def save_eigval_graph(eigval, p):
    fig = plt.figure()
    plt.plot(np.arange(1, p + 1), eigval)
    plt.title("Variance par composante")
    plt.ylabel("Eigen values")
    plt.xlabel("Factor number")
    plt.savefig("fig/acp_eigen_values")
    print("ACP eigenvalues graph saved")
    plt.close(fig)

def save_correlation(df, postfix = ""):
    corr = df.corr()
    ax = sn.heatmap(corr, annot = True)
    plt.savefig("fig/correlation" + postfix)
    print("Data correlation graph saved")
    plt.clf()

if __name__ == "__main__":
    main()