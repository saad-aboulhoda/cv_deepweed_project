# Projet 5 — Classification de mauvaises herbes

## Fichiers

- `DeepWeeds_Rapport_Final.ipynb` : préparation des données, modèles, entraînements et résultats.
- `deepweeds_utils.py` : téléchargement, tableaux, métriques, figures et sauvegarde.
- `inference.py` : rechargement des modèles pour une application.
- `ressources/` : listes des images et photographie W2.

## Exécution

1. Décompresser cette archive et ouvrir le notebook dans le même dossier.
2. Exécuter la cellule d'installation. Redémarrer le noyau si les versions des bibliothèques ont changé.
3. Exécuter les cellules dans l'ordre. Un GPU est conseillé.

Dans Colab, ouvrir le notebook puis importer le ZIP du projet lorsque la première cellule le demande.

## Plan du notebook

1. **Travail demandé :** k-NN, Softmax linéaire et CNN sur 900 images, avec validation croisée et étude de l'augmentation.
2. **Expérimentations complémentaires :** 3000 images, EfficientNetB0, ResNet50, InceptionV3, DINOv2, fine-tuning, normalisation, initialisation, séparation par dates et évaluation externe.
3. **Sauvegarde :** rechargement d'un modèle et téléchargement de tous les modèles entraînés.

## Données

`images900.csv` fixe les 300 images par classe tirées aléatoirement avec la seed 42 parmi les 3205 images éligibles du rapport. `images3000.csv` conserve l'ordre des premières 1000 images par classe de TFDS DeepWeeds 3.0.0.
Ces fichiers contiennent uniquement des noms de JPEG et des étiquettes. Les JPEG originaux sont téléchargés depuis Zenodo et les étiquettes sont vérifiées avec le CSV public DeepWeeds.

`photos_externes.csv` définit les 36 photos du panel externe, leurs labels, leurs URL et leurs crédits. `w2.jpg` est la seconde photographie de l'analyse Grad-CAM.

Aucune caractéristique, prédiction ou tête entraînée précédemment n'est utilisée. Les caractéristiques sont calculées à partir des images et les classifieurs sont entraînés pendant l'exécution.

## Modèles

La dernière cellule crée et télécharge `DeepWeeds_All_Models.zip`. Cette archive contient les classifieurs de chaque expérience, les scalers, les extracteurs partagés, les classes, les résultats et le module d'inférence.
Conserver l'arborescence complète pour les modèles qui partagent un extracteur.

```python
from inference import Predictor
modele = Predictor('.', 'dino_softmax_baseline')
print(modele.predict_file('photo.jpg'))
```
