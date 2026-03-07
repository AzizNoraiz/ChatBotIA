# =============================================================
# CHATBOT BIBLIOTHEQUE - NIVEAU 2 : NLP avec spaCy + TF-IDF
# =============================================================
# Bibliothèques nécessaires :
# pip install spacy scikit-learn
# python -m spacy download fr_core_news_sm
# =============================================================

import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------------------------------------------
# ETAPE 1 — Chargement du modèle de langue français spaCy
# -------------------------------------------------------------
# spaCy est une bibliothèque NLP. On charge ici un modèle
# pré-entraîné sur du texte français (fr_core_news_sm).
# Ce modèle sait : tokeniser, lemmatiser, reconnaître des entités.
# "sm" = small (modèle léger, suffisant pour notre usage)
nlp = spacy.load("fr_core_news_sm")

# -------------------------------------------------------------
# ETAPE 2 — Chargement des données JSON
# -------------------------------------------------------------
with open("bibliotheque.json", "r", encoding="utf-8") as f:
    bibliotheque = json.load(f)

# -------------------------------------------------------------
# ETAPE 3 — Lemmatisation avec spaCy
# -------------------------------------------------------------
# La lemmatisation consiste à ramener chaque mot à sa forme
# de base (lemme) :
#   "mangeais" → "manger"
#   "livres"   → "livre"
#   "écrits"   → "écrire"

# Cela permet de comparer "livres" et "livre" comme identiques.Sans sa on compterer livre et livres comme des mots différents
def lemmatiser(texte):
    doc = nlp(texte.lower())  # on parse le texte avec spaCy
    # On garde uniquement les mots significatifs :
    # on exclut la ponctuation et les "stop words"
    # Les stop words sont les mots très courants sans sens propre :
    # "le", "de", "est", "un", "dans"... spaCy les connaît déjà.
    #exemple 
    #doc = nlp("Les livres écrits par Mark")
    #Chaque token a plusieurs propriétés :
    #token.text → le mot original ("livres")
    #token.lemma_ → sa forme de base ("livre")
    #token.is_stop → True si c'est un mot vide ("le", "de", "est")
    #token.is_punct → True si c'est de la ponctuation
    # token par token :
    # "Les"   → lemme="le",    is_stop=True  ← ignoré
    # "livres"→ lemme="livre", is_stop=False ← gardé
    # "écrits"→ lemme="écrire",is_stop=False ← gardé
    # "par"   → lemme="par",   is_stop=True  ← ignoré
    # "Mark"  → lemme="mark",  is_stop=False ← gardé
    # Résultat lemmatisé : "livre écrire mark"

    lemmes = [token.lemma_ for token in doc
              if not token.is_stop and not token.is_punct]
    return " ".join(lemmes)

# -------------------------------------------------------------
# ETAPE 4 — Construction de la base de connaissances TF-IDF
# -------------------------------------------------------------
# Pour chaque livre, on crée une "description textuelle" qui
# regroupe toutes ses informations en une seule chaîne de texte.
# C'est ce texte que TF-IDF va vectoriser.
def construire_description(livre):
    return (
        f"{livre['Titre du livre']} "
        f"{livre['Auteur']} "
        f"{livre['Domaine']} "
        f"{livre['Année de publication']} "
        f"{livre['Disponibilité']}"
    )

# On construit la liste de toutes les descriptions lemmatisées
descriptions = [lemmatiser(construire_description(l)) for l in bibliotheque]

# -------------------------------------------------------------
# ETAPE 5 — Vectorisation TF-IDF
# -------------------------------------------------------------
# TF-IDF = Term Frequency - Inverse Document Frequency
# C'est une technique qui transforme du texte en vecteurs numériques.
#
# TF (Term Frequency) : mesure combien de fois un mot apparaît
#   dans un document. Plus un mot est fréquent → score élevé.
#
# IDF (Inverse Document Frequency) : pénalise les mots qui
#   apparaissent dans TOUS les documents (donc peu discriminants).
#   Ex : le mot "livre" est dans tous les livres → peu utile.
#
# Résultat : chaque livre devient un vecteur de nombres.
# Ex: ["python", "programmation", "disponible"] → [0.8, 0.6, 0.2]
#
# Cela permet ensuite de COMPARER mathématiquement une question
# avec tous les livres pour trouver le plus similaire.

vectoriseur = TfidfVectorizer()
# fit_transform : apprend le vocabulaire ET transforme les descriptions
matrice_tfidf = vectoriseur.fit_transform(descriptions)
# matrice_tfidf est une matrice où :
#   - chaque ligne = un livre
#   - chaque colonne = un mot du vocabulaire
#   - chaque valeur = le score TF-IDF de ce mot pour ce livre

# -------------------------------------------------------------
# ETAPE 6 — Détection d'intention avec spaCy
# -------------------------------------------------------------
# On analyse la requête pour détecter ce que l'utilisateur cherche.
# spaCy nous aide à extraire les entités nommées (NER) et
# à comprendre le sujet principal de la question.
def detecter_intention(texte):
    doc = nlp(texte.lower())
    tokens = [token.lemma_ for token in doc]

    if any(mot in tokens for mot in ["auteur", "écrire", "écrit", "par"]):
        return "auteur"
    elif any(mot in tokens for mot in ["disponible", "disponibilité", "emprunter"]):
        return "disponibilité"
    elif any(mot in tokens for mot in ["domaine", "sujet", "catégorie"]):
        return "domaine"
    elif any(mot in tokens for mot in ["année", "date", "publication", "sortir"]):
        return "année"
    else:
        return "general"  # recherche globale par similarité TF-IDF

# -------------------------------------------------------------
# ETAPE 7 — Recherche par similarité cosinus (TF-IDF)
# -------------------------------------------------------------
# La similarité cosinus mesure l'angle entre deux vecteurs.
# Si deux vecteurs pointent dans la même direction → angle = 0°
#   → similarité = 1 (très similaires)
# Si deux vecteurs sont perpendiculaires → angle = 90°
#   → similarité = 0 (rien en commun)
#
# Exemple visuel :
#   Requête : "python programmation"  → vecteur [0.9, 0.8, 0.0]
#   Livre 1 : "Apprendre Python"     → vecteur [0.8, 0.7, 0.1]
#   Livre 2 : "Cybersécurité"        → vecteur [0.0, 0.1, 0.9]
#   → Livre 1 est bien plus similaire à la requête que Livre 2

def recherche_tfidf(requete, seuil=0.1):
    # On lemmatise la requête de l'utilisateur
    requete_lemmatisee = lemmatiser(requete)
    # On transforme la requête en vecteur TF-IDF
    # transform (sans fit) : utilise le vocabulaire déjà appris
    vecteur_requete = vectoriseur.transform([requete_lemmatisee])
    # On calcule la similarité entre la requête et chaque livre
    similarites = cosine_similarity(vecteur_requete, matrice_tfidf).flatten()
    # On trie les livres du plus similaire au moins similaire
    indices_tries = np.argsort(similarites)[::-1]

    resultats = []
    for i in indices_tries:
        if similarites[i] >= seuil:
            resultats.append((bibliotheque[i], similarites[i]))
    return resultats

# -------------------------------------------------------------
# ETAPE 8 — Filtre par intention
# -------------------------------------------------------------
#Le but de cette fonction est d'affiner les résultats TF-IDF selon l'intention détectée. TF-IDF est puissant pour trouver des livres similaires, mais il ne sait pas distinguer "disponible" de "emprunté" — cette fonction compense cette limite.
def filtrer_par_intention(requete, intention, resultats):
    # requete => Ce que l'utilisateur à taper.
    #intention => detecter_intention()
    # resultats => res de recherche_tfidf(requete) càd les livres qui ce rapproche de notre recherche.
    doc = nlp(requete.lower())
    mots_cles = [token.lemma_ for token in doc
                 if not token.is_stop and not token.is_punct]

    if intention == "disponibilité":
        if any(m in requete.lower() for m in ["disponible"]):
            return [(l, s) for l, s in resultats if l["Disponibilité"] == "Disponible"] #(livre,score)
        else:
            return [(l, s) for l, s in resultats if l["Disponibilité"] == "Emprunté"]

    return resultats  # pour les autres intentions, TF-IDF suffit

# -------------------------------------------------------------
# ETAPE 9 — Affichage des résultats
# -------------------------------------------------------------
def afficher_resultats(resultats):
    if not resultats:
        print("\n❌ Aucun résultat trouvé. Essayez d'autres mots-clés.\n")
        return
    print(f"\n✅ {len(resultats)} résultat(s) trouvé(s) :\n")
    for i, (livre, score) in enumerate(resultats, 1):
        print(f"  [{i}] 📘 {livre['Titre du livre']}")
        print(f"       ✍️  Auteur       : {livre['Auteur']}")
        print(f"       📂 Domaine      : {livre['Domaine']}")
        print(f"       📅 Publication  : {livre['Année de publication']}")
        print(f"       ✅ Disponibilité: {livre['Disponibilité']}")
        print(f"       🎯 Pertinence   : {score:.0%}")  # score de similarité
        print()

# -------------------------------------------------------------
# ETAPE 10 — Fonction principale du chatbot
# -------------------------------------------------------------
def chatbot(user_input):
    # 1. Détecter l'intention
    intention = detecter_intention(user_input)
    # 2. Rechercher par similarité TF-IDF
    resultats = recherche_tfidf(user_input)
    # 3. Affiner selon l'intention si nécessaire
    resultats = filtrer_par_intention(user_input, intention, resultats)
    # 4. Afficher
    afficher_resultats(resultats)

# -------------------------------------------------------------
# MAIN — Boucle de dialogue
# -------------------------------------------------------------
print("=" * 55)
print("  📚 ChatBot Bibliothèque - Version NLP (spaCy + TF-IDF)")
print("=" * 55)
print("Exemples de questions :")
print("  → Livres écrits par Mark Lutz")
print("  → Livres disponibles sur la programmation")
print("  → Ouvrages sur l'intelligence artificielle")
print()

while True:
    print("Tapez \'quitter\' pour quitter")
    req = input("ChatBot > ").strip()
    if req.lower() == "quitter":
        print("Au revoir !")
        break
    elif req == "":
        continue
    else:
        chatbot(req)
