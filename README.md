# NLP Project — Analyse de sentiment d’avis Trustpilot (Cdiscount)

## 🎯 Objectif du projet

Construire une **chaîne complète de traitement NLP** (de la collecte à l’évaluation) pour **classer automatiquement le sentiment** d’avis clients (positif / négatif) à partir de commentaires textuels.

Dans ce notebook, le cas d’usage est : **avis Trustpilot de Cdiscount**.

Objectifs opérationnels :
- **Collecter** des avis (commentaires + notes) depuis Trustpilot.
- **Nettoyer** et normaliser le texte.
- **Prétraiter NLP** (lemmatisation FR) et représenter les textes (TF-IDF).
- **Évaluer** des modèles de sentiment **pré-entraînés** (Transformers) sur plusieurs versions du texte.
- **Analyser les erreurs** (faux positifs / faux négatifs) et mesurer l’impact du nettoyage/lemmatisation.

---

## 🧩 Contenu du projet (ce qui existe aujourd’hui)

Le projet est actuellement centré sur un notebook unique :

- `nlp_project.ipynb` : pipeline end-to-end (scraping → nettoyage → NLP → benchmark modèles → analyse d’erreurs)

**Colonnes clés créées dans le notebook :**
- `comment` : avis brut récupéré
- `rating` : note (float)
- `label` : cible binaire dérivée de la note  
  - `1` si `rating >= 4` (positif)
  - `0` sinon (négatif)
- `clean_comment` : texte nettoyé (minuscules, ponctuation supprimée, chiffres/emoji/char spéciaux supprimés…)
- `lemmatized` : version lemmatisée via spaCy FR
- Colonnes de prédiction (sur un échantillon) du type `pred_<model_name>` et variantes selon la version texte.

---

## 🧠 Données & étiquetage

### Source
- Trustpilot (page FR) : avis de `www.cdiscount.com` (scraping HTML)

### Labeling (supervision faible)
Le label est dérivé de la note :
- **positif** si note ≥ 4
- **négatif** sinon

> Remarque : c’est un proxy (faible supervision). Un 3★ peut être “neutre” mais est rangé ici en négatif.

---

## 🔁 Pipeline actuel (dans le notebook)

### 1) Collecte (scraping)
- Requêtes `requests` vers Trustpilot avec `User-Agent`
- Extraction par expressions régulières :
  - `reviewBody` pour le texte
  - `ratingValue` pour la note
- Pagination : boucle sur `?page=...`

⚠️ **Attention** : Trustpilot peut limiter/bloquer le scraping. Utiliser des délais, headers, et rester raisonnable.

### 2) Nettoyage structurel
- `dropna`, `drop_duplicates`, suppression des commentaires vides
- Construction de la cible `label`

### 3) Nettoyage texte (`clean_text`)
- suppression HTML (BeautifulSoup)
- lowercasing
- suppression ponctuation
- option : suppression chiffres
- suppression emojis / caractères non-ASCII
- normalisation espaces

### 4) Prétraitement NLP
- téléchargement du modèle spaCy FR : `fr_core_news_sm`
- lemmatisation + suppression stopwords + tokens alpha

### 5) Vectorisation (baseline “classique”)
- `TfidfVectorizer(max_features=5000)`

### 6) Split & gestion déséquilibre
- `train_test_split(..., stratify=y)`
- calcul des `class_weight` (utile si entraînement d’un modèle supervisé)

### 7) Benchmark Transformers (pré-entraînés)
Test sur un **sous-échantillon** (100 avis) via `transformers.pipeline` :
- `tblard/tf-allocine`
- `nlptown/bert-base-multilingual-uncased-sentiment`
- `cardiffnlp/twitter-xlm-roberta-base-sentiment`

Évaluation :
- `classification_report` (precision/recall/f1)
- analyse des erreurs : faux positifs / faux négatifs
- comparaison de l’impact du **texte brut** vs **clean** vs **lemmatized** (selon le bloc).

---

## ✅ Où tu en es maintenant (état actuel)

À ce stade, le projet a déjà :
- ✅ récupéré des avis Trustpilot (texte + note) via scraping
- ✅ construit un dataset et une cible binaire (à partir de la note)
- ✅ implémenté un nettoyage robuste du texte
- ✅ ajouté une lemmatisation FR (spaCy)
- ✅ vectorisé en TF-IDF (préparation pour modèles ML “classiques”)
- ✅ benchmarké plusieurs modèles Transformers pré-entraînés
- ✅ lancé une analyse d’erreurs (FP/FN) et l’impact du nettoyage

Ce qui **n’est pas encore réellement finalisé** (prochaines étapes naturelles) :
- ⏳ entraîner un modèle supervisé (LogReg/SVM) sur TF-IDF
- ⏳ fine-tuner un modèle FR (CamemBERT/FlauBERT) sur tes labels
- ⏳ fiabiliser le scraping (gestion anti-bot, backoff, stockage incrémental)
- ⏳ persister les données (`.csv`/`.parquet`) et versionner les jeux de données
- ⏳ ajouter une vraie “classe neutre” (ou regression sur la note 1–5)

---

## ⚙️ Prérequis

### Environnement
- Python 3.9+ recommandé
- Jupyter Notebook / JupyterLab

### Dépendances principales
- `requests`
- `beautifulsoup4`
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib` (et éventuellement `seaborn`)
- `transformers`
- `torch`
- `spacy` + modèle `fr_core_news_sm`

Installation typique :
```bash
pip install -U requests beautifulsoup4 pandas numpy scikit-learn matplotlib seaborn transformers torch spacy
python -m spacy download fr_core_news_sm
```

> Si tu es sur Apple Silicon / CUDA / environnements spécifiques, adapte l’installation de `torch` selon ta plateforme.

---

## ▶️ Exécution

1. Ouvrir le notebook :
   - `nlp_project.ipynb`
2. Exécuter les cellules dans l’ordre :
   - Scraping → Nettoyage → Lemmatisation → Benchmark modèles
3. Ajuster si besoin :
   - `base_url`
   - le nombre de pages (boucle de pagination)
   - la taille de l’échantillon `sample_df = df.sample(...)`

---

## 📌 Bonnes pratiques & remarques

- **Respect des CGU** : le scraping peut être restreint par Trustpilot. Limiter le volume, ajouter des pauses, et envisager des sources alternatives/datasets publics si nécessaire.
- **Regex fragiles** : les patterns d’extraction peuvent casser si la structure HTML/JSON embarqué change.
- **Labels bruités** : la note n’est pas un sentiment parfait, surtout pour 3★.

---

## 🗺️ Roadmap (suggestion)

1. **Baseline supervisée** : Logistic Regression / Linear SVM sur TF-IDF + class_weight  
2. **Meilleure évaluation** : k-fold, courbe PR, matrice de confusion, calibration  
3. **Fine-tuning** : CamemBERT/FlauBERT sur dataset (après nettoyage)  
4. **Industrialisation** :
   - extraction robuste (retry/backoff)
   - export dataset en `parquet`
   - pipeline reproductible (scripts + config)
   - suivi d’expériences (MLflow)

---

## 🧑‍💻 Auteur

Adem

