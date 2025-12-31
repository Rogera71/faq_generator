# Générateur de FAQ Professionnelle

Ce projet est une application web développée avec Streamlit qui génère automatiquement une Foire Aux Questions (FAQ) professionnelle à partir d'un fichier CSV fourni par l'utilisateur. L'application utilise des modèles de traitement du langage naturel (NLP) pour regrouper les questions par thèmes pertinents et générer des réponses claires et concises.

## Fonctionnalités

- **Génération automatique de FAQ** : Transformez instantanément un fichier CSV de questions/réponses en une FAQ bien structurée.
- **Clustering par thèmes** : Les questions sont automatiquement regroupées en thèmes cohérents à l'aide d'un modèle de clustering (KMeans) et d'embeddings de phrases (SentenceTransformer).
- **Personnalisation avancée** :
  - **Nombre de thèmes** : Ajustez le nombre maximum de thèmes à générer.
  - **Questions par thème** : Définissez le nombre de questions à inclure pour chaque thème.
  - **Filtrage par thèmes prioritaires** : Spécifiez une liste de thèmes pour affiner la génération de la FAQ.
  - **Ton de la FAQ** : Choisissez entre un style "Professionnel", "Amical" ou "Technique".
- **Interface intuitive** : Une interface utilisateur simple et conviviale grâce à Streamlit.

## Fonctionnement

Le processus de génération de la FAQ se déroule en plusieurs étapes :

1.  **Téléchargement du CSV** : L'utilisateur télécharge un fichier CSV contenant des colonnes `question` et `answer`.
2.  **Nettoyage des données** : Le texte est nettoyé (suppression des doublons, des URLs et des caractères non pertinents).
3.  **Génération des embeddings** : Le modèle `SentenceTransformer` convertit les questions en vecteurs numériques (embeddings).
4.  **Clustering** : Le modèle `KMeans` regroupe les questions similaires en se basant sur leurs embeddings.
5.  **Génération des noms de thèmes** : L'API Groq est utilisée pour générer des titres de thèmes pertinents en se basant sur les questions de chaque groupe.
6.  **Synthèse de la FAQ** : L'API Groq génère une FAQ professionnelle au format Markdown en se basant sur les thèmes et les paires de questions/réponses.

## Installation

Pour exécuter cette application localement, suivez ces étapes :

1.  **Clonez le dépôt** :
    ```bash
    git clone https://github.com/votre-utilisateur/votre-repo.git
    cd votre-repo
    ```

2.  **Créez un environnement virtuel** :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
    ```

3.  **Installez les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurez votre clé API Groq** :
    - Créez un fichier `secrets.toml` dans un dossier `.streamlit` à la racine du projet :
      ```
      .
      ├── .streamlit/
      │   └── secrets.toml
      ├── app.py
      └── ...
      ```
    - Ajoutez votre clé API Groq au fichier `secrets.toml` :
      ```toml
      GROQ_API_KEY="votre_cle_api_groq"
      ```

## Utilisation

Une fois l'installation terminée, vous pouvez lancer l'application avec la commande suivante :

```bash
streamlit run app.py
```

Ouvrez votre navigateur et accédez à l'URL locale affichée (généralement `http://localhost:8501`).

## Structure du projet

-   `app.py` : Le code principal de l'application Streamlit.
-   `requirements.txt` : La liste des dépendances Python nécessaires.
-   `model_training.ipynb` : Un notebook Jupyter détaillant le processus d'entraînement des modèles de clustering (`KMeans`) et de transformation de phrases (`SentenceTransformer`).
-   `webfaq_fr_artifacts/` : Ce répertoire contient les modèles pré-entraînés et d'autres artefacts nécessaires au fonctionnement de l'application.

## Entraînement du modèle

Si vous souhaitez comprendre comment les modèles ont été créés, ou si vous voulez les entraîner à nouveau avec vos propres données, veuillez consulter le notebook `model_training.ipynb`. Ce dernier contient toutes les étapes de nettoyage, de traitement et d'entraînement.
