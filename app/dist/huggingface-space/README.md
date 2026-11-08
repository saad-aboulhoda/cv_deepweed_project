---
title: WeedSight
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Identification de mauvaises herbes avec DINOv2 + Softmax
---

# WeedSight

Identification de trois espèces de mauvaises herbes (**Chinee Apple**, **Lantana**,
**Snake Weed**) à partir d'une photo, avec **DINOv2 ViT-S/14 + Softmax**
(jeu DeepWeeds, 600 images de test).

1. L'image est redimensionnée en 224 × 224 et normalisée (ImageNet).
2. DINOv2 (gelé) en extrait un vecteur x de 384 caractéristiques.
3. Le StandardScaler appris sur le train centre et réduit x.
4. Une couche Dense(3) calcule z = W·x + b, et Softmax donne les probabilités.

Documentation de l'API : `/docs`.
