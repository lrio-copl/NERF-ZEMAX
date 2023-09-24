# NERF-ZEMAX

Pour rouler sur Serveur5

``` sh
python main2.py
```


Roule la génération d'image standard

``` sh
python main3.py
```


Roule la génération rapide d'image avec sliders XYZ


## Config

Pour main2.py (et aussi main3.py) on peut modifier les paramètres dans `data/config_test.yml` pour main2 et `data/config_test_simple.yml` pour main3

``` yaml
Ray file: data/mapping_simple_the_true_pinhole.txt #Fichier de rayon pour 1 micro-lentilles
Nerf file: C:\outputs\test_bureau_2\nerfacto\2023-09-16_153142\config.yml #Fichier avec le NERF pour nerfstudio
Output file: test_extended.sdf  #output
Extended ray file: data/microlensxperia9875.dat #Fichier pour étendre les positions des autres micro-lentilles
Working directory: C:\\ #ne pas toucher, important pour nerfstudio for some reasons
Scale: 0.0075 #Scale entre NERf et Zemax
Reference pose: '0276' #Pose de référence (comportement weird en ce moment)
Z translation: 3 #Translation en Z par rapport àla pose
Y translation: -25 #même chose en Y
X translation: -15 #Même chose en X
Z rotation: 0 #Désactivé pour le moment
Factor: 9945 #Pour accélérer calcul de rayon pour nerf studio, ça dépend du nombres de rayons
```