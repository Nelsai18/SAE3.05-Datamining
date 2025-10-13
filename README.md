# SAE3.05-Datamining

## 10. Conseils pratiques pour la rédaction du rapport (SAE)

Gestion de projet : inclure Gantt (tableau), dépôt (lien GitHub), répartition des rôles, flowchart.

Bibliographie : 1 page résumé de l’article + figure (copie/licence) — respecter droits.

Données : expliquer source, variables, unités, NA, transformations.

Qualité : liste des contrôles (NA, doublons, outliers, incohérences).

EDA : inclure 6–8 graphiques pertinents + commentaires analytiques courts.

Détection d’anomalies : expliquer méthode + décision.

Bootstrap : montrer code + résultats + interprétation.

Modélisation : détailler 5 modèles, hyperparamètres testés, scores CV, choix final (justifier par métriques).

Conclusion : limites, perspectives (ex : inclure variables textuelles, interactions, XGBoost, calibration probabiliste).


## 11. Checklist livrables (à rendre)

Notebook .ipynb complet commenté

Rapport PDF (max pages selon consignes) : incluant résumé article, Gantt, flowchart, résultats EDA, anomalie, bootstrap, modèles, conclusion

Présentation (slides) 8–10 diapos

Repo Git (avec README expliquant comment lancer le notebook)


# 12. Temps court (20 h) — priorités (si manque de temps)

S’assurer que pipeline fonctionne end-to-end (prétraitement → modèle → éval sur holdout).

Faire EDA minimal (3 graphiques clés) + 1 dataviz originale.

Bootstrap pour au moins 1 paramètre (survie ou âge).

Entraîner au moins 3 modèles + CV, puis 2 modèles supplémentaires si temps.

GridSearch sur 1 ou 2 modèles clés (RF/GB).
