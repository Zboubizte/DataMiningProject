Veuillez installer les packages indiqués dans le fichier requirement.txt

Voici comment utiliser le script (peut être obtenu en le lançant avec -h) :

usage: Kowalski.py [-h] (-f | -d | -n) (-c | -p | -pc) [-ng] [-dd]

required arguments:
      -f, --full                Execution du script sur tous les adhérents
      -d, --demissionnaire      Execution du script sur les demissionnaires
      -n, --nondemissionnaire   Execution du script sur les non demissionnaires

      -c,  --clustering    Clustering des données
      -p,  --prediction    Prédiction des données
      -pc, --predcluster   Clustering et prédiction des données

optional arguments:
      -h,  --help         show this help message and exit
      -ng, --nograph      Ne pas enregistrer les graphes de l'ACP
      -dd, --dendrogram   Creer le dendrogramme