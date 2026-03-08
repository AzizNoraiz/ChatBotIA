# 📚 ChatBot Bibliothèque Universitaire — Projet IA

Projet réalisé dans le cadre du cours **Introduction à l'IA** (L3 MIAGE).
Ce projet explore trois niveaux de complexité d'un chatbot,
allant d'un système déterministe simple jusqu'à un système
intelligent basé sur un LLM (GPT) avec architecture RAG.

---

---

## 🔰 Niveau 1 — Système Déterministe

### Description
Un chatbot basé sur un **dictionnaire de règles fixes**.
Si la question de l'utilisateur correspond exactement à une clé
du dictionnaire, le bot répond. Sinon, il ne comprend pas.

### Technologie
- Python pur (aucune bibliothèque externe)
- Dictionnaire `{question: réponse}`
- Matching exact sur l'entrée utilisateur

### Fonctionnement
Utilisateur tape → comparaison exacte avec le dictionnaire → réponse fixe


### Limitations
- Aucune tolérance aux fautes de frappe
- Aucune compréhension du contexte
- Si la question n'est pas dans le dictionnaire → réponse d'échec

⚙️ Niveau 2 — NLP Léger (spaCy + TF-IDF)
Description

Un chatbot qui utilise des techniques NLP pour comprendre
les requêtes de façon plus naturelle et rechercher des livres
dans la base de données JSON.

Technologie

spaCy : lemmatisation, suppression des stop words
TF-IDF (scikit-learn) : vectorisation du texte
Similarité cosinus : mesure de pertinence entre requête et livres
difflib : tolérance aux fautes de frappe (Niveau 1 amélioré)

Fonctionnement : 
Requête utilisateur
1 spaCy lemmatise
2 TF-IDF vectorise
3 Similarité cosinus avec chaque livre
4 Tri par score de pertinence (%)
5 Affichage des résultats

🤖 Niveau 3 — RAG + GPT (OpenAI)
Description

Un chatbot intelligent basé sur l'architecture RAG
(Retrieval-Augmented Generation). Il combine la recherche
TF-IDF du Niveau 2 avec la génération de langage naturel
de GPT pour produire des réponses conversationnelles.

Architecture RAG
R — Retrieval   : TF-IDF récupère les livres pertinents
A — Augmented   : les livres sont injectés comme contexte dans le prompt
G — Generation  : GPT génère une réponse naturelle en français

Technologie

spaCy : lemmatisation
TF-IDF (scikit-learn) : retrieval des livres pertinents
OpenAI API (gpt-4o-mini) : génération de réponses naturelles
Historique de conversation : mémoire entre les tours de dialogue
