# Projet T-AIA-902

Ce projet comprend trois types d'algorithmes d'**apprentissage par renforcement**, entraînés sur des environnements issus de la **librairie Python Gymnasium** :
1. Un algorithme de **Q-Learning**, entraîné sur trois environnements différents : [`FrozenLake-v1`](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/), [`Taxi-v3`](https://www.gymlibrary.dev/environments/toy_text/taxi/) et [`CliffWalking-v0`](https://www.gymlibrary.dev/environments/toy_text/cliff_walking/).
2. Un algorithme **SARSA**, entraîné sur l'environnement [`CliffWalking-v0`](https://www.gymlibrary.dev/environments/toy_text/cliff_walking/).
3. Un algorithme de **Deep-Q-Learning**, implémentant un réseau de neurones convolutionnel (CNN) simple pour apprendre à jouer au jeu **Breakout** (environnement [`ALE/Breakout-v5`](https://gymnasium.farama.org/v0.26.3/environments/atari/breakout/)) à partir de 4 images consécutives.

## Système d'exploitation

Ce projet a été développé sur **Windows 11**.

## Python

Ce projet utilise la **version 3.13.2** de Python.

## Exécution des algorithmes dans VSCode

1. Pour les algorithmes **Q-Learning** et **SARSA** : ouvrir le fichier souhaité d'extension `.ipynb`, puis cliquer sur l'onglet `Run All`.
2. Pour l'algorithme **Deep-Q-Learning** : se référer aux instructions spécifiées dans le [README](./Deep-Q-Learning%20-%20ATARI%20Breakout/README.md) du dossier associé.