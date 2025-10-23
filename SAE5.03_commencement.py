# Importation des librairies
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve, auc)
from sklearn import metrics

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.utils import resample

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#Les differentes librairies

## Jeu de données sur Titanic (objectif : prédire la survie)
import seaborn as sns
titanic = sns.load_dataset('titanic')
# [891 rows x 15 columns]
print("# Colonnes du dataset Titanic :")
print(titanic.columns.tolist())
# 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare','embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town','alive', 'alone'

## 4.Description rapide et qualité des données
# Aperçu des colonnes et des valeurs manquantes
print("\n--- Info dataset ---")
titanic.info()

# Calcul des % de NA
na_pct = titanic.isna().mean().round(3) * 100
print(na_pct[na_pct>0])


# Tableau récapitulatif des variables
variables_table = pd.DataFrame([
    ("survived","Indique si le passager a survécu (1) ou non (0)","Numérique binaire"),
    ("pclass","Classe du billet : 1,2,3 (statut social)","Numérique ordinale"),
    ("sex","Sexe du passager : male / female","Catégorielle"),
    ("age","Âge en années","Numérique continue"),
    ("sibsp","Nombre de frères/sœurs ou conjoints à bord","Numérique discrète"),
    ("parch","Nombre de parents/enfants à bord","Numérique discrète"),
    ("fare","Prix du billet (livres sterling)","Numérique continue"),
    ("embarked","Port d'embarquement : C/Q/S","Catégorielle"),
    ("class","Classe textuelle : First/Second/Third","Catégorielle"),
    ("who","man/woman/child","Catégorielle"),
    ("adult_male","True si homme adulte","Booléenne"),
    ("deck","Pont (A..G)","Catégorielle"),
    ("embark_town","Ville d'embarquement","Catégorielle"),
    ("alive","yes/no (survie)","Catégorielle"),
    ("alone","True si voyage seul","Booléenne"),
], columns=["Nom","Description","Type"])
# print(variables_table)
# variables non significatives selon l'article : 'sibsp','parch', 'embarked'

# on va utiliser : survived, pclass, sex, age, fare, who, deck, alive, embark_town, alone

# Sélection des variables pertinentes
data = titanic[['survived', 'pclass', 'sex', 'age','fare','who','deck','alive','embark_town','alone']].dropna()
print(data.head())
# [712 rows x 7 columns]



## Nettoyage et pipeline de preprocessing

X = data[['pclass', 'sex', 'age','fare','who','deck','alive','embark_town','alone']]  # juste les features
y = data['survived']

numeric_features = ['age','fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['pclass', 'sex', 'who','deck','alive','embark_town','alone']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
# preprocessor : C’est un objet qui combine toutes les transformations à appliquer aux données avant de les donner au modèle

# Petit test : transformer X pour voir les feature names
preprocessor.fit(X)
ohe_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(ohe_cols)
print("\nFeatures après transformation (extrait) :", feature_names)


## Exploration & dataviz (quelques graphiques essentiels)

# Histogramme des ages
plt.figure(figsize=(6,4))
sns.histplot(titanic['age'], kde=True)
plt.title('Distribution des âges (titanic)')
plt.xlabel('Age')
plt.show()

# Surive en fonction du sexe
plt.figure(figsize=(8,4))
sns.countplot(data=titanic, x='sex', hue='survived')
plt.title('Survie par sexe')
plt.show()

# Surive par classe
plt.figure(figsize=(8,4))
sns.countplot(data=titanic, x='pclass', hue='survived')
plt.title('Survie par classe')
plt.show()

# Nuage de points
plt.figure(figsize=(7,5))
# On remplace les NA de fare par median juste pour l'affichage si besoin
if 'fare' in titanic.columns:
    sns.scatterplot(data=titanic, x='age', y='fare', hue='survived', alpha=0.7)
    plt.ylim(0, 300)
    plt.title('Age vs Fare colored by Survived')
    plt.show()

