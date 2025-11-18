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



## Jeu de données sur Titanic (objectif : prédire la survie)
import seaborn as sns
titanic = sns.load_dataset('titanic')
# [891 rows x 15 columns]
print("# Colonnes du dataset Titanic :")
print(titanic.columns.tolist())
# 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare','embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town','alive', 'alone'

## 4.Description rapide et qualité des données (2)
# Aperçu des colonnes et des valeurs manquantes
print("\n--- Info dataset ---")
titanic.info()

# Calcul des % de NA (2.2-missing values)
na_pct = titanic.isna().mean().round(3) * 100
print(na_pct[na_pct>0])


# Tableau récapitulatif des variables (2.1)
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

# on va utiliser : survived, pclass, sex, age, fare, who, alive, embark_town, alone

# Sélection des variables pertinentes
data = titanic[['survived', 'pclass', 'sex', 'age','fare','who','alive','embark_town','alone']].dropna()
print(data.head())
# [712 rows x 9 columns]



## Nettoyage et pipeline de preprocessing

X = data[['pclass', 'sex', 'age','fare','who','alive','embark_town','alone']]  # juste les features
y = data['survived']

numeric_features = ['age','fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['pclass', 'sex', 'who','alive','embark_town','alone']
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


## Exploration & dataviz (quelques graphiques essentiels) (2.4)

# Histogramme des ages (2.4.1-tri a plat)
plt.figure(figsize=(6,4))
# sns.histplot(titanic['age'], kde=True)
sns.distplot(titanic['age'], kde=True)
plt.title('Distribution des âges (titanic)')
plt.xlabel('Age')
plt.show()

# Histogramme des prix des billets (2.4.1-tri a plat)
plt.figure(figsize=(6,4))
# sns.histplot(titanic['fare'], kde=True)
sns.distplot(titanic['fare'], kde=True)
plt.title('Distribution des Prix du billet (livre sterling)')
plt.xlabel('Prix du billet (livre sterling)')
plt.show()

# Surive en fonction du sexe (2.4.2-tri croisé)
plt.figure(figsize=(8,4))
sns.countplot(data=titanic, x='sex', hue='survived')
plt.title('Survie par sexe')
plt.show()

# Surive par classe (2.4.2-tri croisé)
plt.figure(figsize=(8,4))
sns.countplot(data=titanic, x='pclass', hue='survived')
plt.title('Survie par classe')
plt.show()

# Sélection des variables numériques et de la variable cible pour le pair plot
cols_for_pairplot = ['age', 'fare','survived']

# Conversion de 'survived' en type catégoriel pour une meilleure légende
# Copie du sous-ensemble de données pour éviter un SettingWithCopyWarning
data_pairplot = data[cols_for_pairplot].copy() 
data_pairplot['survived'] = data_pairplot['survived'].astype('category')
data_pairplot['survived'] = data_pairplot['survived'].cat.rename_categories({0: 'Non Survécu', 1: 'Survécu'})

# Génération du Pair Plot
# hue='survived' colore les points en fonction de la variable cible
# Les histogrammes sur la diagonale sont remplacés par des KDE (courbes de densité)
sns.pairplot(data_pairplot, hue='survived', diag_kind='kde')
plt.suptitle('Pair Plot des variables numériques coloré par la Survie', y=1.02)
plt.show()

# Nuage de points
plt.figure(figsize=(7,5))
# On remplace les NA de fare par median juste pour l'affichage si besoin
if 'fare' in titanic.columns:
    sns.scatterplot(data=titanic, x='age', y='fare', hue='survived', alpha=0.7)
    plt.ylim(0, 300)
    plt.title('Age vs Fare colored by Survived')
    plt.show()

# rajouter d'autres tris à plats, tri croisé(faire avec un pair plot)on doit faire des graphique original qui sorte de l’ordinaire()


## 6.Détection d’anomalies (2.3)
num_for_iso = titanic[['age','fare','sibsp','parch']].copy()
num_for_iso = num_for_iso.fillna(num_for_iso.median())

iso = IsolationForest(contamination=0.02, random_state=42) # IsolationForest est un algorithme pour détecter les valeurs atypiques.
iso.fit(num_for_iso)
outliers = iso.predict(num_for_iso) == -1
titanic['is_outlier'] = outliers

print("Nombre d'outliers détectés:", outliers.sum())
# Visualiser outliers sur fare
plt.figure(figsize=(6,4))
sns.boxplot(x=titanic['fare'])
plt.title('Fare boxplot (outliers visibles)')
plt.show()



## 7.Bootstrap : estimation d’un IC (exemple : proportion de survie, et moyenne d’âge)
"""Idée simple :
On a un échantillon (ici les passagers du Titanic).
On tire plein d’échantillons "simulés" à partir du même jeu de données, avec remise (un passager peut être choisi plusieurs fois).
Pour chaque échantillon, on calcule le paramètre qui nous intéresse (moyenne, proportion…).
À la fin, on regarde la distribution de ces valeurs pour construire un intervalle de confiance."""

# Bootstrap pour proportion de survie
n_boot = 10000
prop_star = []
for i in range(n_boot):
    samp = resample(y, replace=True, n_samples=len(y))
    prop_star.append(samp.mean())

# Afficher estimateur et IC
prop_hat = y.mean()
ic = np.percentile(prop_star, [2.5, 97.5])
print(f"Proportion survie: {prop_hat:.3f}, IC95 bootstrap: [{ic[0]:.3f}, {ic[1]:.3f}]")
# Proportion survie: 0.404, IC95 bootstrap: [0.368, 0.440]

# Bootstrap mean age (sans NA)
ages = titanic['age'].dropna()
age_boot = []
for _ in range(n_boot):
    samp = resample(ages, replace=True, n_samples=len(ages))
    age_boot.append(samp.mean())
ic_age = np.percentile(age_boot, [2.5, 97.5])
print(f"Moyenne age: {ages.mean():.2f}, IC95 bootstrap: [{ic_age[0]:.2f}, {ic_age[1]:.2f}]")
# Moyenne age: 29.70, IC95 bootstrap: [28.62, 30.76]


## 8. Problème de classification — prédire

# -------------------------------------------
# 8.i Sélection des prédicteurs (importance des variables)
# -------------------------------------------
log = LogisticRegression(max_iter=1000, random_state=42)
pipe_rfe = Pipeline(steps=[('pre', preprocessor), ('clf', log)])

# Transformation des données (encodage + normalisation)
X_processed = preprocessor.fit_transform(X)

# Importance des variables avec RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_processed, y)
importances = rf.feature_importances_

# Récupération des noms de colonnes après transformation
ohe_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(ohe_cols)

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Top 10 variables les plus importantes :")
print(feat_imp.head(10))


# -------------------------------------------
# 8.ii Construire plusieurs modèles (au moins 5)
# -------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Dictionnaire de modèles
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'NaiveBayes': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Évaluation des modèles
results = []
for name, model in models.items():
    pipe = Pipeline([('pre', preprocessor), ('clf', model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"{name} - ROC AUC moyen: {scores.mean():.3f} (écart-type: {scores.std():.3f})")

    # Entraînement sur tout le train pour évaluer sur test
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['clf'], "predict_proba") else pipe.decision_function(X_test)

    results.append({
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    })
"""
LogisticRegression - ROC AUC moyen: 0.849 (écart-type: 0.034)
NaiveBayes - ROC AUC moyen: 0.824 (écart-type: 0.053)
LDA - ROC AUC moyen: 0.850 (écart-type: 0.033)
KNN - ROC AUC moyen: 0.835 (écart-type: 0.047)
RandomForest - ROC AUC moyen: 0.845 (écart-type: 0.030)
"""

df_results = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
print("\nComparaison des modèles :")
print(df_results)
"""
Comparaison des modèles :
                model  accuracy  precision    recall        f1   roc_auc
3                 KNN  0.783217   0.728814  0.741379  0.735043  0.865720
0  LogisticRegression  0.811189   0.771930  0.758621  0.765217  0.858316
2                 LDA  0.776224   0.732143  0.706897  0.719298  0.854868
4        RandomForest  0.783217   0.721311  0.758621  0.739496  0.854564
1          NaiveBayes  0.762238   0.700000  0.724138  0.711864  0.828296
"""


# -------------------------------------------
# 8.iv Choisir le meilleur modèle
# -------------------------------------------
best_model_name = df_results.iloc[0]['model']
print(f"\n Meilleur modèle selon le ROC-AUC : {best_model_name}")
# Meilleur modèle selon le ROC-AUC : KNN

# Récupération du modèle choisi
best_model = models[best_model_name]
pipe_best = Pipeline([('pre', preprocessor), ('clf', best_model)])
pipe_best.fit(X_train, y_train)

"""
# -------------------------------------------
# 8.v (Optionnel) Ajustement léger (fine-tuning)
# -------------------------------------------
# Exemple simple : ajuster n_neighbors pour KNN si c’est lui le meilleur
if best_model_name == 'KNN':
    best_score = 0
    for k in [3, 5, 7, 9]:
        knn = KNeighborsClassifier(n_neighbors=k)
        pipe = Pipeline([('pre', preprocessor), ('clf', knn)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
        mean_score = scores.mean()
        print(f"KNN avec k={k} : ROC-AUC moyen = {mean_score:.3f}")
        if mean_score > best_score:
            best_score = mean_score
            pipe_best = pipe
"""
# -------------------------------------------
# 8.vi Prédiction sur données test
# -------------------------------------------
y_pred = pipe_best.predict(X_test)
y_proba = pipe_best.predict_proba(X_test)[:, 1] if hasattr(pipe_best.named_steps['clf'], "predict_proba") else pipe_best.decision_function(X_test)

print("\nRapport de classification du meilleur modèle :")
print(classification_report(y_test, y_pred))
print("ROC AUC sur le test :", roc_auc_score(y_test, y_proba))

"""
Rapport de classification du meilleur modèle :
              precision    recall  f1-score   support

           0       0.82      0.81      0.82        85
           1       0.73      0.74      0.74        58

    accuracy                           0.78       143
   macro avg       0.78      0.78      0.78       143
weighted avg       0.78      0.78      0.78       143

ROC AUC sur le test : 0.8657200811359027
"""

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title(f'Matrice de confusion — {best_model_name}')
plt.show()

"""
| Indicateur | Valeur approx.    | Interprétation |
| ---------- | ----------------- | -------------- |
| Accuracy   | 0.78              | Bon            |
| Precision  | 0.73              | Correct        |
| Recall     | 0.74              | Correct        |
| F1-score   | 0.73              | Bon équilibre  |
| ROC-AUC    | (si tu as ≈ 0.86) | Excellent      |
"""

## Le modèle fait un bon travail globalement.


